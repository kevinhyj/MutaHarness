"""Unsupervised sequence-based stability prior; no assay outcomes are used."""
import json, math, re, hashlib
from pathlib import Path
import numpy as np

AA='ARNDCQEGHILKMFPSTWYV'
BLOSUM=np.array([
[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]], dtype=float)
HYD=dict(zip(AA,[1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-.8,-.7,-.9,-1.3,4.2]))
VOL=dict(zip(AA,[88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.,116.1,227.8,193.6,140.]))
CHARGE={a: (1.0 if a in 'KR' else .2 if a=='H' else -1.0 if a in 'DE' else 0.0) for a in AA}

def parse_and_check(case):
    wt=case['reference_sequence']; candidates=case['query']
    assert len(candidates)==250
    assert set(c['id'] for c in candidates)=={f'M{i:03d}' for i in range(1,251)}
    parsed={}
    for c in candidates:
        m=[]; seq=list(wt)
        for label in c['mutation'].split(':'):
            match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',label)
            assert match,label
            old,pos,new=match.groups(); pos=int(pos)
            assert 1<=pos<=len(wt) and wt[pos-1]==old and new in AA and new!=old
            assert pos not in [t[1] for t in m]
            seq[pos-1]=new; m.append((old,pos,new))
        assert len(m)==c['mutation_order'] and ''.join(seq)==c['sequence'],c['id']
        parsed[c['id']]=m
    assert case['context']['higher_is_better'] is True
    return wt,candidates,parsed

def baseline_cost(old,pos,new,wt):
    i,j=AA.index(old),AA.index(new)
    distance=(BLOSUM[i,i]+BLOSUM[j,j])/2-BLOSUM[i,j]
    cost=.19*distance
    # Local aliphatic/aromatic density is an uncertain core indicator, not RSA.
    near=wt[max(0,pos-5):min(len(wt),pos+4)]
    density=sum(a in 'AVILMFWY' for a in near)/len(near)
    core=max(0.,min(1.,(density-.25)/.55))
    if old in 'VILMFWY' and new not in 'VILMFWYC':
        cost+=core*max(0.,HYD[old]-HYD[new])*.17
    if old=='G':cost+=.65
    if old=='P':cost+=.55
    if new=='P':cost+=.65
    if new=='G' and old not in 'G':cost+=.25
    return float(cost)

def main():
    case=json.load(open('case.json'));manifest=json.load(open('evidence_manifest.json'))
    wt,candidates,parsed=parse_and_check(case)
    assert manifest['support_count']==0 and manifest['measurements_present'] is False
    scores=[]
    for c in candidates:
        cost=sum(baseline_cost(*m,wt) for m in parsed[c['id']])
        scores.append({'id':c['id'],'score':float(np.tanh(-cost/4.))})
    result={'predictions':scores,'summary':'Sequence-only stability prior: general substitution compatibility, local hydrophobic density, and glycine/proline penalties; costs summed across substitutions and mapped monotonically to one global [-1,1] scale. No supports, MSA, structure, partners, or measured WT were supplied. Scores are uncalibrated relative priorities; residue burial and epistasis are uncertain.'}
    assert len(scores)==250 and all(math.isfinite(p['score']) and -1<=p['score']<=1 for p in scores)
    Path('checkpoint_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    Path('input_audit.json').write_text(json.dumps({'sources':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['case.json','evidence_manifest.json']},'candidate_count':250,'reference_length':len(wt),'all_mutation_identities_verified':True,'msa_present':False,'structure_present':False,'fixed_partners_present':False,'higher_is_better':True},indent=2)+'\n')
    print('Checkpoint saved:',len(scores),'validated predictions; score range',min(p['score'] for p in scores),max(p['score'] for p in scores))
if __name__=='__main__': main()
