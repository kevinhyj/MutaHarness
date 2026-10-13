from pathlib import Path
import json,collections
import numpy as np

def read_alignment():
    seqs=[]; names=[]; parts=[]
    for line in Path('raw_evidence/HIS7_YEAST_full_11-26-2021_b09.a2m').read_text().splitlines():
        if line.startswith('>'):
            if parts: seqs.append(''.join(parts)); parts=[]
            names.append(line[1:])
        else: parts.append(line.strip())
    if parts: seqs.append(''.join(parts))
    rawref=seqs[0]; case=json.loads(Path('case.json').read_text());ref=case['reference_sequence']
    assert rawref.upper().replace('-','').replace('.','')==ref
    mapping=[];pos=0
    for x in rawref:
        if x!='-' and x!='.':pos+=1
        if x.isupper() or x=='-':mapping.append(pos if x!='-' else None)
    seqs=[''.join(x for x in s if x.isupper() or x=='-') for s in seqs]
    assert len(set(map(len,seqs)))==1
    rows=sorted(set(seqs[1:]))
    arr=np.array([list(s) for s in rows],dtype='U1');wt=np.array(list(seqs[0]))
    return arr,wt,np.array(mapping),rows

if __name__=='__main__':
    a,w,m,rows=read_alignment();valid=(a!='-');n=valid.sum(axis=1);ident=(a==w).sum(axis=1)/np.maximum(n,1)
    print('unique rows',a.shape,'identity quantiles',np.quantile(ident,[0,.25,.5,.75,.9,.95,.99,1]));print('coverage quantiles',np.quantile(n/a.shape[1],[0,.1,.5,.9,1]))
    for threshold in [.4,.5,.6,.7,.8,.9]:print('identity>=',threshold,int(((ident>=threshold)&(n>=.8*a.shape[1])).sum()))
    first=ident[:2048];print('selected first2048 identity',np.quantile(first,[0,.5,.9,.99,1]))
    print('query aligned positions',len(m),'first',m[:30].tolist())
