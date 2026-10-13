"""Label-free GFP fluorescence ranking; initial checkpoint model.
No assay labels are read. Missing MSA terms receive a mutation-burden prior,
not a fabricated complete evolutionary score. Higher scores are better.
"""
import json, math, hashlib, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read_and_validate():
    case=json.loads((ROOT/'case.json').read_text())
    manifest=json.loads((ROOT/'evidence_manifest.json').read_text())
    msa=json.loads((ROOT/'msa_evidence.json').read_text())
    assert manifest['support_count']==0 and not manifest['measurements_present']
    assert case['context']['higher_is_better'] is True
    for name,meta in manifest['files'].items():
        if isinstance(meta,dict) and 'sha256' in meta:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==meta['sha256'],name
    wt=case['reference_sequence']
    assert case['domain']=='protein' and not manifest['allowed_modalities']['structure']
    assert not case.get('reference_chains') and not case.get('fixed_partners')
    profile=json.loads((ROOT/'raw_evidence/msa_profile.json').read_text())
    assert profile['msa_sha256']==manifest['files']['raw_evidence/GFP_AEQVI_full_04-29-2022_b08.a2m']['sha256']
    raw_query=[]
    with (ROOT/'raw_evidence/GFP_AEQVI_full_04-29-2022_b08.a2m').open() as handle:
        header=next(handle)
        assert header.startswith('>GFP_AEQVI/1-238')
        for line in handle:
            if line.startswith('>'):break
            raw_query.append(line.strip())
    query=''.join(raw_query)
    assert query.upper()==wt
    unaligned={i for i,a in enumerate(query,1) if a.islower()}
    assert unaligned=={1,234,235,236,237,238}
    for row in case['evidence']['sites']:
        pos=row['position']; pp=profile['positions'][str(pos)]
        assert pp['wt']==row['WT']==wt[pos-1]
        assert pp['frequency']==row['msa']['frequencies']
        assert (row['msa']['non_gap_depth']==0)==(pos in unaligned)
    assert hashlib.sha256(wt.encode()).hexdigest()==msa['reference_sequences_sha256']['__single__']
    assert len(case['query'])==250
    assert {q['id'] for q in case['query']}=={f'M{i:03d}' for i in range(1,251)}
    assert msa['candidate_order']==[q['id'] for q in case['query']]
    byid={e['id']:e for e in msa['candidates']}
    for q in case['query']:
        muts=[]; seq=list(wt)
        for token in q['mutation'].split(':'):
            w,pos,a=re.fullmatch(r'([A-Z])(\d+)([A-Z])',token).groups();pos=int(pos)
            assert wt[pos-1]==w and seq[pos-1]==w and a!=w
            seq[pos-1]=a;muts.append((w,pos,a))
        assert ''.join(seq)==q['sequence'] and len(muts)==q['mutation_order']
        assert [(t['WT'],t['position'],t['ALT']) for t in byid[q['id']]['substitution_terms']]==sorted(muts,key=lambda t:t[1])
    return case,manifest,msa

def main():
    case,manifest,msa=read_and_validate()
    byid={e['id']:e for e in msa['candidates']}
    preds=[]
    for q in case['query']:
        terms=byid[q['id']]['substitution_terms']
        burden=.65*len(terms)
        for t in terms:
            x=t['sum_log_frequency_ratio']
            if x is not None and t['non_gap_depth']>0:
                burden+=.32*max(-x,0)-.08*min(max(x,0),2)
        score=2*math.exp(-burden/10)-1
        preds.append({'id':q['id'],'score':round(score,8)})
    summary=('Initial label-free checkpoint: independent-site supplied MSA log-frequency ratios plus mutation burden, on one global monotone scale (higher is better). Source and reference hashes and all mutation identities verified. No measured supports or structure; shallow homolog evidence is an uncertain fluorescence prior; missing terms are excluded, not treated as neutral.')
    out={'predictions':preds,'summary':summary}
    (ROOT/'checkpoint_predictions.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Saved usable checkpoint:',len(preds),'finite predictions. MSA Neff:',msa['profile_neff'])


# Assay-aware revision. These are coarse expert priors, not supplied geometry.
# No weights are fitted: the MSA is shallow and includes fluorescent-protein
# lineages whose emission spectra and sequence backgrounds differ from avGFP.
HYDRO=set('AVILMFWYC')
CHARGE=set('DEKR')
CONSERVATIVE=[set(g) for g in ['AVLIM','FYWH','ST','NQ','DE','KR','AG','CS']]
# Approximate native GFP regular-secondary-structure intervals. No measured
# accessibility, contacts, relaxed mutant coordinates, or epistasis is claimed.
BETA_RANGES=[(11,22),(25,37),(40,50),(94,104),(117,124),(132,142),
             (145,153),(161,171),(176,186),(198,208),(216,227)]
HELIX_RANGES=[(58,74),(80,88)]
# Coarse core/chromophore packing sites from standard GFP fold knowledge;
# these are explicitly model priors and are not verified RSA observations.
PACKING_SITES=set([12,14,16,18,22,27,29,42,44,46,48,55,57,60,61,64,
                  68,70,71,74,83,84,88,98,100,106,114,119,123,125,130,
                  136,137,141,143,150,152,161,163,165,178,188,199,201,
                  207,218,220,224])
# Specific mechanistic concerns: chromophore identity/maturation/proton relay.
# A green-channel assay may penalize spectral changes without loss of folding.
MECHANISTIC={
    (65,'P'):7.5,(66,'C'):9.0,(66,'D'):10.0,
    (57,'R'):6.0,(70,'R'):5.0,(96,'C'):7.0,(96,'H'):4.5,
    (94,'L'):2.0,(222,'K'):3.2,(205,'F'):2.0,(205,'P'):3.0,
    (203,'I'):1.0,(203,'A'):.45,(183,'R'):2.5,(183,'L'):1.4,
}
# Established engineering substitutions are modest transferable priors only.
# Benefits remain uncertain without the specific assay/sequence background.
ENGINEERING_PRIORS={
    (30,'R'):-.45,(39,'N'):-.65,(99,'S'):-.65,(99,'C'):-.25,
    (105,'T'):-.45,(105,'S'):-.25,(145,'F'):-.5,(153,'T'):-.6,
    (163,'A'):-.45,(171,'V'):-.45,(206,'V'):-.25,
    (46,'L'):-.08,(72,'A'):-.15,(175,'G'):-.2,
}

def in_ranges(pos,ranges):
    return any(a<=pos<=b for a,b in ranges)

def substitution_cost(t):
    w,p,a=t['WT'],t['position'],t['ALT']
    conservative=any(w in g and a in g for g in CONSERVATIVE)
    cost=.20 if conservative else .48
    # Zero depth/null is missing evidence. The biochemical prior still applies;
    # the observed partial MSA sum is never relabeled as a complete score.
    observed=(t['sum_log_frequency_ratio'] is not None and
              t['non_gap_depth'] is not None and t['non_gap_depth']>0)
    evo=0.0
    if observed:
        lr=t['sum_log_frequency_ratio']
        # Retain the exact supplied LR as input. Shallow-family evidence is
        # only one component, with less influence for chemically similar AAs.
        reliability=t['non_gap_depth']/(t['non_gap_depth']+10.0)
        evo=.28*reliability*(.55 if conservative else 1.0)*min(max(-lr,0),6)
        evo-=.09*reliability*min(max(lr,0),2)
    cost+=evo
    # Backbone disruption depends on an approximate fold prior, not RSA.
    if a=='P':
        if in_ranges(p,BETA_RANGES):cost+=3.5
        elif in_ranges(p,HELIX_RANGES):cost+=4.3
        elif p in [7,53,54,56,58,75,89,192,196]:cost+=.7
        else:cost+=.6
        # Proline is common here in natural homologs: avoid an inconsistent
        # blanket secondary-structure penalty at the flexible strand entrance.
        if p==146:cost-=2.5
    if w=='P':
        cost+=.9 if p in [54,56,58,75,89] else .45
    if w=='G' and a not in 'AS':cost+=.45
    if p in PACKING_SITES and w in HYDRO and a!='P':
        if a in CHARGE:cost+=2.7
        elif a not in HYDRO:cost+=1.35
        elif w in 'FWY' and a not in 'FWY':
            cost+=.40
            # Cross-family retention of a buried aromatic ring is stronger
            # evidence than conservation at an arbitrary surface position.
            if observed and t['frequency_WT']>.85:cost+=.60
        elif w in 'ILVM' and a=='A':cost+=.10
    # Conservative polar changes on the known chromophore surroundings can
    # alter spectra, but do not receive a generic catastrophic folding penalty.
    if p==148 and a not in 'HQNST':cost+=.65
    if p==143 and a not in 'YF':cost+=.9
    if p==146 and a not in 'NQSTP':cost+=.75
    if p==159 and a not in 'NQDE':cost+=.35
    if p==77 and a not in 'HNQDE':cost+=.65
    if p==78 and a in CHARGE:cost+=1.35
    if p==165 and a not in HYDRO:cost+=.45
    if p==167 and a in 'NSTQDEKR':cost+=.40
    # Terminal substitutions are outside the aligned core; absence of homologs
    # is not taken as evidence of neutrality. Use a weak terminal-tolerance prior.
    if p>=230 or p<=9:cost*=.5
    if p in [10,11,128,176,182,184,194,195,198,200,202,204,206,208,209,210,211,212,213,214,215,219,221,223,228,229]:
        cost*=.75
    cost+=MECHANISTIC.get((p,a),0)
    if (p,a) in ENGINEERING_PRIORS:
        cost=ENGINEERING_PRIORS[p,a]
    return cost,{'msa_observed':observed,'evolution_component':evo,'cost':cost,
                 'prior_core':p in PACKING_SITES}

def refined_main():
    case,manifest,msa=read_and_validate()
    # Deterministic scorer is re-run and compared to supplied terms; this tests
    # bookkeeping only, and cannot validate the phenotype ranking.
    import msa_tools
    recalculated=msa_tools.score_case(case)
    assert recalculated==msa
    byid={e['id']:e for e in msa['candidates']}
    predictions=[];diagnostics=[]
    for q in case['query']:
        terms=byid[q['id']]['substitution_terms']
        evaluated=[substitution_cost(t) for t in terms]
        costs=[x[0] for x in evaluated];n=len(costs)
        # Combine aggregate mutational load with a modest weakest-link prior.
        # No pair coefficients or measured epistasis are estimated.
        burden=sum(costs)+.30*max([0]+costs)+.035*max(0,n-2)**2
        # One global strictly monotone transformation, not block normalization.
        score=2/(1+math.exp((burden-3.3)/3.2))-1
        predictions.append({'id':q['id'],'score':round(score,8)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],'burden':burden,
            'msa_coverage':byid[q['id']]['scores']['sum_log_frequency_ratio']['coverage'],
            'complete_msa_lr':byid[q['id']]['scores']['sum_log_frequency_ratio']['score'],
            'terms':[dict(mutation=f"{t['WT']}{t['position']}{t['ALT']}",**d)
                     for t,(_,d) in zip(terms,evaluated)]})
    assert len(predictions)==250 and len({x['id'] for x in predictions})==250
    assert all(math.isfinite(x['score']) and -1<=x['score']<=1 for x in predictions)
    summary=('Label-free GFP fluorescence ranking combines supplied independent-site MSA log ratios with conservative-substitution, GFP packing/backbone, chromophore-maturation and modest established engineering priors; aggregate load and a weakest-link term share one global scale, higher better. Raw/profile/reference hashes and 1-based mutation mapping verified. MSA Neff 13.88; 17 candidates have incomplete evolutionary evidence at unaligned tail positions, handled with weak biochemical priors rather than neutral or complete MSA scores. No structure or fixed partners are supplied. Scores are uncalibrated: shallow homolog conservation, spectral shifts, folding thresholds and epistasis make ordering uncertain; no assay outcomes were used.')
    output={'predictions':predictions,'summary':summary}
    (ROOT/'final_predictions.json').write_text(json.dumps(output,indent=2)+'\n')
    (ROOT/'prediction_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print('Validated',len(predictions),'predictions; range',min(x['score'] for x in predictions),max(x['score'] for x in predictions))
    for d in sorted(diagnostics,key=lambda d:d['burden'])[:25]:
        print('TOP',d['id'],round(d['burden'],2),d['mutation'])
    for d in sorted(diagnostics,key=lambda d:d['burden'])[-12:]:
        print('LOW',d['id'],round(d['burden'],2),d['mutation'])

if __name__=='__main__':
    import sys
    if '--checkpoint' in sys.argv:main()
    else:refined_main()
