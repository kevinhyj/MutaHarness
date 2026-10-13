"""Label-free HIS3 prioritization; no measured fitness is loaded."""
import hashlib, json, math, re
from pathlib import Path
import numpy as np

def load_validate():
    case = json.loads(Path('case.json').read_text())
    manifest = json.loads(Path('evidence_manifest.json').read_text())
    evidence = json.loads(Path('msa_evidence.json').read_text())
    assert not manifest['measurements_present'] and manifest['support_count'] == 0
    assert case['context']['higher_is_better'] is True
    for name, desc in manifest['files'].items():
        if isinstance(desc, dict) and 'sha256' in desc:
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == desc['sha256'], name
    ref = case['reference_sequence']
    assert evidence['reference_sequences_sha256'] == {'__single__':hashlib.sha256(ref.encode()).hexdigest()}
    # Single enzyme sequence: there are no separately supplied binding partners.
    assert 'reference_chains' not in case
    idx = {e['id']:e for e in evidence['candidates']}
    expected = {f'M{i:03d}' for i in range(1,251)}
    assert {q['id'] for q in case['query']} == expected and len(case['query']) == 250
    for q in case['query']:
        s = list(ref); positions = set()
        for token in q['mutation'].split(':'):
            m = re.fullmatch(r'([A-Z])(\d+)([A-Z])',token); assert m
            wt, pos, alt = m.group(1), int(m.group(2)), m.group(3)
            assert pos not in positions and ref[pos-1] == wt and wt != alt
            positions.add(pos); s[pos-1] = alt
        assert ''.join(s) == q['sequence'] and len(positions) == q['mutation_order']
        e = idx[q['id']]; assert e['mutation'] == q['mutation']
        assert all(t['non_gap_depth'] > 0 for t in e['substitution_terms'])
        assert e['scores']['sum_log_frequency_ratio']['complete']
    return case, evidence, idx

def write_predictions(path, values, summary, case):
    predictions = [{'id':q['id'], 'score':round(float(values[q['id']]),8)} for q in case['query']]
    assert len(predictions) == 250 and len({p['id'] for p in predictions}) == 250
    assert all(math.isfinite(p['score']) and -1 <= p['score'] <= 1 for p in predictions)
    Path(path).write_text(json.dumps({'predictions':predictions,'summary':summary},indent=2)+'\n')

def baseline(case, idx):
    values = {q['id']:math.tanh(idx[q['id']]['scores']['sum_log_frequency_ratio']['score']/20) for q in case['query']}
    summary = ('Baseline: complete independent-site natural-homolog log-frequency ratios, mapped through a single tanh scale (20). Source and sequence hashes and every mutation verified; all mutated sites have homolog coverage (profile Neff 560). No assay labels, structure, or supplied fixed partners. Conservation is an uncertain growth prior and ignores epistasis; scores are relative, not calibrated assay units.')
    write_predictions('checkpoint_predictions.json',values,summary,case)

def refined(case, evidence, idx):
    from scipy.special import logsumexp
    from local_profile import local_profile, AA
    from amino_acids import BLOSUM62, ORDER
    import msa_tools
    assert msa_tools.score_case(case) == evidence, 'Fixed-score evidence must reproduce exactly'
    freq, mapping, diag, homologs, weights = local_profile(return_homologs=True)
    pos_idx = {int(p):i for i,p in enumerate(mapping)}
    # Bind the A2M columns to full, one-based WT positions and profile missingness.
    absent = [s['position'] for s in case['evidence']['sites'] if s['msa']['non_gap_depth'] == 0]
    assert set(mapping) == set(range(1,len(case['reference_sequence'])+1))-set(absent)
    ai={a:i for i,a in enumerate(AA)}
    bi={a:i for i,a in enumerate(ORDER)}
    b=BLOSUM62[np.ix_([bi[a] for a in AA],[bi[a] for a in AA])]
    d=(np.diag(b)[:,None]+np.diag(b)[None,:])/2-b
    assert (d>=0).all()
    distance=np.full((21,21),6.0);distance[:20,:20]=d
    assert np.array_equal(distance,distance.T)
    ref=np.array([ai[case['reference_sequence'][int(p)-1]] for p in mapping])
    logweights=np.log(weights/weights.sum())
    values={};rows=[]
    for q in case['query']:
        terms=idx[q['id']]['substitution_terms']
        effects=[];chemistry=0
        for t in terms:
            i=pos_idx[t['position']];wt,alt=ai[t['WT']],ai[t['ALT']]
            foreground_lr=math.log(freq[i,alt]/freq[i,wt])
            # A geometric ensemble of broad and WT-neighborhood evidence.
            effects.append(.4*t['sum_log_frequency_ratio']+.6*foreground_lr)
            chemistry+=.15*max(0,-b[wt,alt])
        losses=np.maximum(0,-np.asarray(effects));gains=np.maximum(0,effects)
        # Limit consensus gains; preserve sensitivity to a single severe lesion.
        marginal=-.8*losses.sum()-.6*losses.max()+.25*np.minimum(gains,2).sum()-.15*len(terms)-chemistry
        # Natural block compatibility: a fixed mixture of observed homologs,
        # smoothed with a BLOSUM distance kernel. It supplies statistical
        # co-occurrence evidence, not an experimentally fitted epistasis model.
        low=min(t['position'] for t in terms)-4
        high=max(t['position'] for t in terms)+4
        window=(mapping>=low)&(mapping<=high)
        target=np.array([ai[q['sequence'][int(p)-1]] for p in mapping])
        h=homologs[:,window]
        wt_energy=distance[ref[window][None,:],h].sum(axis=1)/3
        mut_energy=distance[target[window][None,:],h].sum(axis=1)/3
        block=float(logsumexp(logweights-mut_energy)-logsumexp(logweights-wt_energy))
        latent=.75*marginal+.25*block
        values[q['id']]=math.tanh((latent+10)/12)
        rows.append({'id':q['id'],'mutation':q['mutation'],'complete_broad_log_ratio':idx[q['id']]['scores']['sum_log_frequency_ratio']['score'],'foreground_log_ratio':float(sum(effects)),'marginal_robust_score':float(marginal),'natural_block_score':block,'latent_score':float(latent),'msa_coverage':1.0,'window':[int(low),int(high)]})
    summary = ('Label-free growth ranking combines supplied marginal MSA evidence (40%) with a redundancy-weighted WT-neighborhood profile (60%), plus a modest BLOSUM-smoothed natural-block compatibility term. Severe individual losses outweigh consensus gains; one global tanh scale. Raw/profile/WT hashes, residue mapping, mutation sequences and all 250 IDs verified. All mutation sites have homolog coverage; 28 other WT positions lack evidence. Broad Neff 560; neighborhood 938 unique homologs. No measured outcomes, structures or supplied fixed partners. Natural co-occurrence is not validated assay epistasis; scores are uncertain relative priorities, not calibrated growth.')
    write_predictions('final_predictions.json',values,summary,case)
    audit={'foreground_profile':diag,'broad_profile_neff':evidence['profile_neff'],'absent_reference_positions':absent,'complete_candidate_msa_scores':250,'mutated_missing_sites':0,'hashes_verified':True,'mutation_identity_verified':True,'fixed_partners':'None separately supplied; single enzyme sequence','score_direction':'higher is predicted better growth','revisions':['Baseline checkpoint uses exact complete supplied LR sums.','Raw homologs are predominantly distant; use a WT-neighborhood profile with five broad-prior sequence equivalents to reduce phylogenetic/context mismatch.','Candidates have 3-11 clustered substitutions; a low-weight natural block kernel retains observed co-occurrence without fitting free interaction parameters.','Growth can fail through one severe lesion; cap positive consensus compensation and include a worst-site penalty. These are scientific priors, not outcome-validated improvements.'],'candidates':rows}
    Path('prediction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'candidates':len(values),'score_range':[min(values.values()),max(values.values())],'foreground':diag,'top_ids':sorted(values,key=values.get,reverse=True)[:10],'bottom_ids':sorted(values,key=values.get)[:10]}))

if __name__ == '__main__':
    import sys
    case,evidence,idx=load_validate()
    if '--checkpoint' in sys.argv:
        baseline(case,idx)
        print('Validated baseline checkpoint: 250 finite scores; all mutation terms covered.')
    else:
        refined(case,evidence,idx)
