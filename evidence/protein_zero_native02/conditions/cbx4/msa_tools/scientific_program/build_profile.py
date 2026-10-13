"""Reproducible coverage-unbiased natural-homolog profile from supplied A2M.
Uses a single 4096-row uniform sample of deduplicated natural rows, inverse
80%-identity neighbor weights, and total-one uniform Dirichlet pseudocount.
No candidate-dependent selection, outcomes, structure, or epistasis.
"""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
AA='ACDEFGHIKLMNPQRSTVWY'

def build():
    path=ROOT/'raw_evidence/CBX4_HUMAN_2023-08-07_b03.a2m'
    records=[];sequence=[]
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            if sequence:records.append(''.join(sequence));sequence=[]
        else:sequence.append(line.strip())
    if sequence:records.append(''.join(sequence))
    ref=json.loads((ROOT/'case.json').read_text())['reference_sequence']
    assert records[0].upper()==ref and len(ref)==50
    assert all(len(s)==50 for s in records)
    # Positions 1 and 50 are query lowercase insertions, not homolog columns.
    assert records[0][0].islower() and records[0][-1].islower()
    assert records[0][1:-1].isupper()
    unique=sorted(set(s[1:-1].upper().replace('.','-') for s in records[1:]))
    rng=np.random.default_rng(20261013)
    indices=np.sort(rng.choice(len(unique),min(4096,len(unique)),replace=False))
    seq=[unique[i] for i in indices]
    x=np.array([[AA.find(a) if a in AA else 20 for a in s] for s in seq],dtype=np.int8)
    valid=x<20
    n=len(x);neighbors=np.zeros(n,dtype=np.int32)
    # Missing and ambiguous amino acids cannot create similarity evidence.
    for start in range(0,n,128):
        block=x[start:start+128]
        pairvalid=(block[:,None,:]<20)&valid[None,:,:]
        shared=pairvalid.sum(2)
        matches=((block[:,None,:]==x[None,:,:])&pairvalid).sum(2)
        near=(shared>=24)&(matches>=0.8*shared)
        neighbors[start:start+len(block)]=near.sum(1)
    assert np.all(neighbors>=1)
    weights=1.0/neighbors
    depth=np.sum(valid*weights[:,None],axis=0)
    positions={}
    for j in range(48):
        counts=np.bincount(x[:,j],weights=weights,minlength=21)[:20]
        frequencies=(counts+0.05)/(depth[j]+1.0)
        assert np.isclose(frequencies.sum(),1)
        positions[str(j+2)]={'wt':ref[j+1],'frequency':dict(zip(AA,frequencies.tolist())),
           'non_gap_effective_depth':float(depth[j]),'gap_fraction':float(1-depth[j]/weights.sum()),
           'entropy_bits':float(-np.sum(frequencies*np.log2(frequencies))),
           'homolog_column_1based':j+1}
    profile={'format':'resampled_homolog_profile_v1','msa_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'reference_sha256':hashlib.sha256(ref.encode()).hexdigest(),
      'method':{'natural_rows':len(records)-1,'unique_rows':len(unique),'sample_rows':n,'seed':20261013,
       'selection':'uniform sample without replacement of all sorted unique natural rows, query excluded',
       'weighting':'inverse number of >=80% identical sampled neighbors at >=24 shared canonical AA columns',
       'pseudocount_total':1.0,'neff':float(weights.sum()),
       'mapping':'WT positions 2..49 correspond exactly to match-state columns 1..48; WT 1,50 are unsupported query insertions'},
      'positions':positions}
    (ROOT/'resampled_msa_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print(json.dumps(profile['method'],indent=2))
    for pos in [2,3,4,5,6,7,8,9,10,12,16,21,23,30,36,43,46]:
        r=positions[str(pos)];f=r['frequency']
        print(pos,r['wt'],round(r['non_gap_effective_depth'],1),round(r['entropy_bits'],2),
              ' '.join(f'{a}:{v:.3f}' for a,v in sorted(f.items(),key=lambda z:-z[1])[:6]))
if __name__=='__main__':build()
