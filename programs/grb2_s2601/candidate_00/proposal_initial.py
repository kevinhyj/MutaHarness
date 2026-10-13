"""Support-only deterministic sequence regression; scores are ordinal, not probabilities."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Published-style physical descriptors (hydropathy, side-chain volume, charge).
HYDRO = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
VOLUME = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]
def aa_descriptors():
    z=np.array([[HYDRO[i],VOLUME[i],float(a in 'KR')-float(a in 'DE'),float(a in 'FWY'),float(a=='G'),float(a=='P'),float(a in 'STNQ'),float(a in 'IVT')] for i,a in enumerate(AA)])
    z[:,:2]=(z[:,:2]-z[:,:2].mean(0))/z[:,:2].std(0)
    return z
DESC=aa_descriptors()

def mutations(record, reference):
    seq=record.get('sequence')
    if seq is not None and len(seq)==len(reference):
        return [(i,a,b) for i,(a,b) in enumerate(zip(reference,seq)) if a!=b]
    result=[]
    for token in record.get('mutation','').split(':'):
        if token: result.append((int(token[1:-1])-1,token[0],token[-1]))
    return result

def features(records,reference):
    # All coordinates are fixed by the supplied reference; no query labels used.
    n=len(records); L=len(reference)
    site=np.zeros((n,L)); exact=np.zeros((n,L*20)); prop=np.zeros((n,L*len(DESC[0])))
    global_prop=np.zeros((n,16)); aa=np.zeros((n,20)); pair=np.zeros((n,400)); order=np.zeros((n,1))
    for j,r in enumerate(records):
        ms=mutations(r,reference);order[j,0]=len(ms)
        for p,w,m in ms:
            if w not in AA or m not in AA or not 0<=p<L:continue
            wi=AA.index(w);mi=AA.index(m);d=DESC[mi]-DESC[wi]
            site[j,p]=1;exact[j,p*20+mi]+=1;prop[j,p*8:p*8+8]+=d
            global_prop[j,:8]+=d;global_prop[j,8:]+=abs(d)
            aa[j,mi]+=1;pair[j,wi*20+mi]+=1
    return {'site':site,'exact':exact,'prop':prop,'global':global_prop,'aa':aa,'pair':pair,'order':order}

def kernel(f,g,config=(1.,0.4,0.15,0.04,0.1,0.1)):
    return sum(w*(f[k]@g[k].T) for k,w in zip(['site','exact','prop','global','aa','pair'],config)) + .2*(f['order']@g['order'].T)

def propose(case,assignment):
    supports=case.get('support',[]); query=case.get('query',[])
    if not supports: return {'scores':{str(q['id']):0.0 for q in query},'selected_ids':[]}
    f=features(supports,case['reference_sequence']);g=features(query,case['reference_sequence'])
    y=np.array([float(r['fitness']) for r in supports]);K=kernel(f,f);Q=kernel(g,f)
    yc=y-y.mean();pred=Q@solve(K+np.eye(len(y))*.7,yc,assume_a='pos')
    ranks=rankdata(pred);scores=2*(ranks-.5)/max(1,len(pred))-1
    return {'scores':{str(q['id']):float(s) for q,s in zip(query,scores)},'selected_ids':[]}
