"""Label-free KRAS abundance ranking. No measured outcomes are used.

Baseline checkpoint: observed MSA log ratios plus a physicochemical fallback.
Final model extensions and their scientific rationale are recorded in model_notes.md.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

AA = 'ACDEFGHIKLMNPQRSTVWY'
HYDRO = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
GROUPS = ['AVLIM','FYW','STNQ','DE','KRH','CGP']
# Qualitative knowledge of the Ras fold, NOT coordinates, SASA, or measured RSA.
# Used as a weak packing prior; the switch regions can change conformation.
PACKING = {
    2:0.02,6:0.85,7:0.75,8:0.65,9:0.85,10:0.15,11:0.3,
    12:0.05,13:0.10,14:0.70,15:0.15,16:0.10,17:0.15,
    18:0.60,19:0.85,21:0.80,22:0.30,23:0.65,24:0.60,25:0.10,
    26:0.10,27:0.15,29:0.25,30:0.05,31:0.05,32:0.10,33:0.05,
    35:0.05,36:0.45,37:0.05,38:0.05,39:0.10,40:0.15,41:0.10,42:0.15,
    44:0.80,45:0.25,46:0.85,47:0.10,48:0.10,49:0.10,50:0.10,
    51:0.60,52:0.30,53:0.90,54:0.10,55:0.90,56:0.65,57:0.15,
    58:0.10,59:0.10,60:0.05,61:0.05,62:0.10,63:0.10,64:0.45,
    65:0.10,66:0.15,67:0.55,68:0.10,70:0.15,71:0.75,72:0.50,
    75:0.10,76:0.10,78:0.80,79:0.85,80:0.80,81:0.75,82:0.85,
    83:0.20,84:0.85,86:0.15,88:0.10,97:0.15,98:0.10,101:0.10,
    102:0.10,103:0.30,106:0.10,107:0.10,108:0.10,110:0.25,
    111:0.55,112:0.90,113:0.90,114:0.70,116:0.25,117:0.10,
    118:0.15,119:0.15,120:0.45,123:0.15,125:0.65,126:0.10,
    127:0.10,128:0.10,130:0.45,131:0.10,133:0.80,136:0.10,
    137:0.55,146:0.15,149:0.10,153:0.10,155:0.55,157:0.35,
    159:0.85,162:0.10,163:0.75,164:0.10,165:0.05,166:0.05,
}
HELIX = set(range(16,26)) | set(range(67,75)) | set(range(87,105)) | set(range(127,138)) | set(range(152,167))
STRAND = set(range(5,10)) | set(range(40,47)) | set(range(51,57)) | set(range(79,85)) | set(range(116,120)) | set(range(140,145))
HYDROPHOBIC = dict(zip(AA,[0.5,0.6,0,0,1.0,0.05,0.25,1.0,0,1.0,0.9,0,0.2,0,0,0.1,0.2,0.95,1.0,0.8]))
VOLUME = dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))

def abundance_term(term):
    w,a,p = term['WT'],term['ALT'],term['position']
    b = PACKING.get(p, 0.02 if p >= 167 else 0.20)
    # Penalize lost core hydrophobicity, packing cavities/clashes, and new buried charge.
    hw,ha = HYDROPHOBIC[w],HYDROPHOBIC[a]
    physical = b * (1.80*max(0,hw-ha) + 0.65*abs(VOLUME[a]-VOLUME[w])/100)
    if a in 'DEKR' and w not in 'DEKR': physical += 0.55*b
    if w not in 'AVLIMFYWCP' and ha > 0.45:
        physical += 0.40*(1-b)*(ha-0.35)
    # Modest cost even for conservative substitutions; changes of charge are mostly tolerated on surface.
    same = any(w in g and a in g for g in GROUPS)
    physical += 0.015 if same else 0.045
    if a == 'P':
        physical += 1.75 if p in HELIX or p in STRAND else 0.20
        if p == 66: physical -= 0.13  # switch-II helix N-cap, not its interior
    if a == 'G' and p in HELIX: physical += 0.40
    if w == 'P' and a != 'P': physical += 0.20
    if w == 'G' and p < 167:
        physical += 0.18 if a in 'AS' else 0.45
    # Nucleotide occupancy stabilizes Ras. These are chemistry priors, not function scores.
    nucleotide = 0.0
    if p == 12:
        # Common small/medium activating substitutions need not destabilize the fold.
        physical = {'C':0.12,'D':0.20,'V':0.22,'P':1.10,'Y':1.20}.get(a,0.45)
    elif p == 13: nucleotide = 0.25 if a in 'KRFWY' else 0.10
    elif p == 15: nucleotide = 0.65 if a in 'FYWKR' else 0.25
    elif p == 16 and a not in 'KR': nucleotide = 0.70
    elif p == 17 and a != 'T': nucleotide = 0.55 if a == 'N' else 0.70
    elif p == 57 and a != 'E': nucleotide = 0.75
    elif p == 58 and a not in 'ST': nucleotide = 0.12
    elif p == 59 and a in 'KRH': nucleotide = 0.12
    elif p == 60 and a not in 'GAS': nucleotide = 0.10
    elif p == 116 and a not in 'NQT': nucleotide = 0.40 if a != 'W' else 0.65
    elif p == 117 and a not in 'KR': nucleotide = 0.15 if a in 'NQ' else 0.40
    elif p == 119 and a != 'E': nucleotide = 0.70
    elif p == 146 and a in 'FYW': nucleotide = 0.55
    if p == 2: physical = 0.01 if a in 'ASC' else 0.08
    if p >= 167:
        physical *= 0.35
        nucleotide = 0.0
    # Keep supplied MSA marginal intact, but modulate its predictive contribution.
    # Shrink shallow columns and functional conservation for an abundance endpoint.
    lr = term['sum_log_frequency_ratio']
    depth = term['non_gap_depth'] or 0
    evolution = 0.0
    reliability = 0.0
    if lr is not None and depth > 0:
        reliability = depth/(depth+50.0)
        specificity = 0.45 + 0.70*b
        if p in {35,37,38,40,59,60,61,62}: specificity *= 0.35
        if p in {57,116,117,119,146}: specificity *= 0.60
        if p >= 167: specificity *= 0.30
        evolution = 0.12*reliability*specificity*max(-1.0,min(7.0,-lr))
    cost = 0.75*physical + nucleotide + evolution
    return -cost, {'packing_prior':b,'physical_cost':physical,'nucleotide_cost':nucleotide,
                   'evolution_cost':evolution,'msa_reliability':reliability,
                   'msa_term_observed':lr is not None and depth>0}

def load_inputs():
    case = json.loads(Path('case.json').read_text())
    manifest = json.loads(Path('evidence_manifest.json').read_text())
    ev = json.loads(Path('msa_evidence.json').read_text())
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better'] is True
    assert not manifest['allowed_modalities']['structure']
    for path, spec in manifest['files'].items():
        if isinstance(spec, dict) and 'sha256' in spec:
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == spec['sha256']
    wt = case['reference_sequence']
    assert hashlib.sha256(wt.encode()).hexdigest() == ev['reference_sequences_sha256']['__single__']
    profile = json.loads(Path('raw_evidence/msa_profile.json').read_text())
    assert profile['msa_sha256'] == manifest['files']['raw_evidence/RASK_HUMAN_2023-08-07_b03.a2m']['sha256']
    for site in case['evidence']['sites']:
        row = profile['positions'][str(site['position'])]
        assert row['wt'] == site['WT'] and row['frequency'] == site['msa']['frequencies']
    from msa_tools import score_case
    assert score_case(case) == ev
    assert 'reference_chains' not in case  # single protein, no supplied fixed partners
    assert len(case['query']) == 250
    assert {q['id'] for q in case['query']} == {f'M{i:03}' for i in range(1,251)}
    assert ev['candidate_order'] == [q['id'] for q in case['query']]
    for site in case['evidence']['sites']:
        assert site['chain'] is None and site['WT'] == wt[site['position']-1]
    index = {c['id']: c for c in ev['candidates']}
    for q in case['query']:
        actual = [f'{w}{i}{a}' for i,(w,a) in enumerate(zip(wt,q['sequence']),1) if w != a]
        assert len(q['sequence']) == len(wt)
        assert sorted(actual) == sorted(q['mutation'].split(':'))
        assert len(actual) == q['mutation_order']
        assert index[q['id']]['mutation'] == q['mutation']
    return case, ev

def baseline_term(term):
    w, a, p = term['WT'], term['ALT'], term['position']
    same = any(w in g and a in g for g in GROUPS)
    penalty = 0.04 if same else 0.15
    if a == 'P' and p < 167: penalty += 0.35
    if w == 'G' and a not in 'ASG' and p < 167: penalty += 0.15
    lr = term['sum_log_frequency_ratio']
    # None is missing information, never evolutionary neutrality.
    observed = lr is not None and (term['non_gap_depth'] or 0) > 0
    if observed:
        penalty += 0.075 * (-lr)
    if p >= 167: penalty *= 0.25
    return -penalty

def run(mode='baseline', output='checkpoint_predictions.json'):
    case, ev = load_inputs()
    predictions = []
    details = []
    for q, e in zip(case['query'], ev['candidates']):
        if mode == 'baseline':
            terms = [baseline_term(t) for t in e['substitution_terms']]
            features = []
        else:
            calculated = [abundance_term(t) for t in e['substitution_terms']]
            terms = [x[0] for x in calculated]
            features = [x[1] for x in calculated]
        latent = sum(terms)
        score = math.tanh((latent + 0.40) / (0.85 if mode=='baseline' else 1.75))
        predictions.append({'id': q['id'], 'score': round(score,7)})
        details.append({'id':q['id'], 'mutation':q['mutation'], 'latent':latent,
                        'term_effects':terms,'features':features,'msa_coverage':e['scores']['sum_log_frequency_ratio']['coverage']})
    summary = ('Unsupervised abundance prior combining observed natural-homolog log-frequency ratios '
               'with conservative substitution penalties; missing MSA terms use biochemical priors. '
               'One global monotonic scale; no measured calibration, mutant structure or validated epistasis.')
    if mode != 'baseline':
        summary = ('Label-free KRAS abundance ranking combines conservative packing/backbone and nucleotide-stability priors '
                   'with depth-shrunk MSA ratios and reduced switch-function penalties. Hashes and residue identities verified; '
                   'MSA covers 35–170, with 117 incomplete candidates supplemented by biochemical priors. '
                   'No supplied partners or structure; packing priors are qualitative. Additive effects on one global scale. '
                   'No measured calibration or validated epistasis; substantial ranking uncertainty.')
    result = {'predictions': predictions, 'summary':summary}
    assert len(predictions)==250 and all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    Path(output.replace('.json','_diagnostics.json')).write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps({'output':output,'count':len(predictions),'range':[min(p['score'] for p in predictions),max(p['score'] for p in predictions)],
                      'complete_msa_candidates':sum(e['scores']['sum_log_frequency_ratio']['complete'] for e in ev['candidates'])}))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',default='baseline',choices=['baseline','abundance'])
    parser.add_argument('--output',default='checkpoint_predictions.json')
    args=parser.parse_args()
    run(args.mode,args.output)
