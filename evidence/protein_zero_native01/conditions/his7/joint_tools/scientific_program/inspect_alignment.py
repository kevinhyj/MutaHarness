import json,collections,re,numpy as np
from pathlib import Path

def read_alignment():
    records=[];name=None;seq=[]
    for line in open('raw_evidence/HIS7_YEAST_full_11-26-2021_b09.a2m'):
        line=line.strip()
        if line.startswith('>'):
            if name is not None:records.append((name,''.join(seq)))
            name=line[1:];seq=[]
        else:seq.append(line)
    records.append((name,''.join(seq)))
    q=records[0][1];wt=json.load(open('case.json'))['reference_sequence']
    assert q.replace('-','').replace('.','').upper()==wt
    alignment=[];positions=[];p=0; col=0
    for aa in q:
        if aa in '.-':col+=int(aa=='-');continue
        p+=1
        if aa.isupper():positions.append(p);col+=1
    keep=lambda s: ''.join(x for x in s if x.isupper() or x=='-')
    seqs=[keep(s) for n,s in records]
    assert set(map(len,seqs))=={len(positions)}
    amino='ACDEFGHIKLMNPQRSTVWY-';amap={a:i for i,a in enumerate(amino)}
    a=np.array([[amap.get(x,20) for x in s] for s in seqs],dtype=np.int8)
    full=np.full((len(a),len(wt)),20,dtype=np.int8);full[:,np.array(positions)-1]=a
    return records,full,positions

if __name__=='__main__':
    records,a,positions=read_alignment(); non=(a[1:]<20)&(a[0]<20);matches=(a[1:]==a[0])&non
    ident=matches.sum(1)/np.maximum(non.sum(1),1);cov=non.sum(1)/(a[0]<20).sum()
    print('MSA records',len(a),'query aligned positions',len(positions),'query insertion positions',sorted(set(range(1,221))-set(positions)))
    print('IDs quantiles',np.quantile(ident,[0,.25,.5,.75,.9,.95,.99,1]));print('COV quantiles',np.quantile(cov,[0,.25,.5,.75,.9,.99,1]))
    for cut in [.4,.5,.6,.7,.8,.9,.95]:print('identity>=',cut,'cov>=0.8',int(((ident>=cut)&(cov>=.8)).sum()))
    print('Closest homologs',[(records[i+1][0],round(ident[i],3),round(cov[i],3)) for i in np.argsort(-ident)[:12]])
    np.savez_compressed('alignment_parsed.npz',alignment=a,positions=positions,identities=ident,coverage=cov)
