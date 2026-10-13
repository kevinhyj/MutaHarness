"""Support-only deterministic protein stability ranking.
All fits are rebuilt from the supplied support; no persisted training artifacts.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Standard amino-acid hydropathy, side-chain volume, charge, and helix propensity.
HYDRO = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOLUME = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]
HELIX = [1.45,.77,.98,1.53,1.12,.53,1.24,1.,1.07,1.34,1.2,.73,.59,1.17,.79,.79,.82,1.14,1.14,.61]
def properties():
    a=np.array([[HYDRO[i],VOLUME[i],HELIX[i],float(c in 'KR')-float(c in 'DE')+.15*(c=='H'),float(c in 'FWY'),float(c=='P'),float(c=='G'),float(c in 'STNQ'),float(c in 'IVT'),float(c=='C')] for i,c in enumerate(AA)])
    return (a-a.mean(0))/a.std(0)

def features(records,ref):
    p=properties(); n=len(records); L=len(ref)
    ident=np.zeros((n,L*20)); phys=np.zeros((n,L*p.shape[1])); pos=np.zeros((n,L)); glob=np.zeros((n,p.shape[1]))
    for i,r in enumerate(records):
        seq=r.get('sequence')
        if not seq:
            seq=list(ref)
            for m in r['mutation'].split(':'):seq[int(m[1:-1])-1]=m[-1]
        for j,(a,b) in enumerate(zip(ref,seq)):
            if a!=b:
                u=AA.index(a); v=AA.index(b); d=p[v]-p[u]
                ident[i,j*20+v]=1; pos[i,j]=1
                phys[i,j*p.shape[1]:(j+1)*p.shape[1]]=d; glob[i]+=d
    return ident,phys,pos,glob

def propose(case,assignment):
    train=case['support']; query=case['query']
    if not query:return {'scores':{},'selected_ids':[]}
    if not train:return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    f=features(train+query,case['reference_sequence']); n=len(train)
    X=np.concatenate([f[0],f[1]*.2,f[2]*.7,f[3]*.12,np.ones((len(train)+len(query),1))],axis=1)
    y=np.array([r['fitness'] for r in train],float)
    K=X@X.T
    pred=K[n:,:n]@solve(K[:n,:n]+.3*np.eye(n),y,assume_a='pos')
    scores=2*(rankdata(pred)-.5)/len(pred)-1
    return {'scores':{r['id']:float(s) for r,s in zip(query,scores)},'selected_ids':[]}
