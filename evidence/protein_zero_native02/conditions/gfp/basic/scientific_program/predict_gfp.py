"""Unsupervised GFP fluorescence ranking; uses supplied sequences and biochemical priors only."""
import json, math, re
from pathlib import Path

AA='ARNDCQEGHILKMFPSTWYV'
ROWS='''
4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4
'''
BLOSUM={(a,b):float(v) for a,row in zip(AA,ROWS.strip().splitlines()) for b,v in zip(AA,row.split())}

def load_validate():
    case=json.load(open('case.json')); manifest=json.load(open('evidence_manifest.json'))
    ref=case['reference_sequence']; queries=case['query']
    assert len(queries)==250 and len({q['id'] for q in queries})==250
    assert {q['id'] for q in queries}=={f'M{i:03d}' for i in range(1,251)}
    assert case['context']['higher_is_better'] is True
    assert manifest['support_count']==0 and manifest['measurements_present'] is False
    parsed={}
    for q in queries:
        seq=list(ref); muts=[]; positions=set()
        for m in q['mutation'].split(':'):
            match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',m); assert match,m
            a,p,b=match.groups(); p=int(p)
            assert p not in positions and ref[p-1]==a and a!=b and b in AA,(q['id'],m)
            positions.add(p); seq[p-1]=b; muts.append((a,p,b))
        assert ''.join(seq)==q['sequence'] and len(muts)==q['mutation_order'],q['id']
        parsed[q['id']]=muts
    # GFP has no fixed partner sequences in this case.
    assert not any(k in case for k in ['partners','fixed_partners'])
    return case,manifest,parsed

def baseline_cost(a,p,b):
    cost=.22+.14*max(0,-BLOSUM[a,b])
    if b=='P' and a!='P': cost+=.8
    if a in 'GP' and b!=a: cost+=.3
    if p in (66,67):cost+=6
    if p==65:cost+=0 if b=='T' else 3
    if p in (96,222):cost+=2
    return cost

def score(cost):
    # A single monotone bounded map for all candidates; no per-block scaling.
    return math.tanh((1.25-cost)/3.5)

def run():
    case,manifest,parsed=load_validate()
    predictions=[{'id':q['id'],'score':score(sum(baseline_cost(*m) for m in parsed[q['id']]))} for q in case['query']]
    result={'predictions':predictions,'summary':'Unsupervised sequence-only checkpoint: mutation burden, substitution compatibility, proline/glycine disruption, and GFP chromophore/catalytic-residue priors. No measured outcomes, MSA, or structural evidence are supplied; scores are relative heuristics on one global scale, with high uncertainty.'}
    assert all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in predictions)
    Path('checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved and validated checkpoint_predictions.json:',len(predictions),'unique candidates.')
if __name__=='__main__':run()
