"""Reusable sequence features and small-sample kernel models; no data reads."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata, norm
from proposal import AA, RAW, mutations

# Putative WW-domain regions from the local sequence motif. Hypotheses only:
# beta1 176--181; beta2 186--190; beta3 196--201; intervening loops.
def region(p):
    if 176<=p<=181:return 1
    if 186<=p<=190:return 3
    if 196<=p<=201:return 5
    if 182<=p<=185:return 2
    if 191<=p<=195:return 4
    return 0

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
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]],float)

def kernels(records, reference):
    rows=[mutations(r,reference) for r in records]
    keys=sorted(set(m for row in rows for m in row))
    if not keys:return {'exact':np.zeros((len(records),len(records)))}
    n=len(rows);t=len(keys); lookup={m:i for i,m in enumerate(keys)}
    M=np.zeros((n,t))
    for i,row in enumerate(rows):
        for m in row:M[i,lookup[m]]=1
    p=np.array([m[0] for m in keys]);w=np.array([AA.index(m[1]) for m in keys]);a=np.array([AA.index(m[2]) for m in keys])
    props=np.array([RAW[x] for x in AA]);props=(props-props.mean(0))/props.std(0)
    # BLOSUM rows used as similarity descriptors, no external sequences.
    bz=(BLOSUM-BLOSUM.mean(0))/BLOSUM.std(0)
    diff=props[a]-props[w];bd=bz[a]-bz[w]
    same=p[:,None]==p[None,:]
    reg=np.array([region(x) for x in p]);samereg=reg[:,None]==reg[None,:]
    rbf=np.exp(-np.mean((props[a,None,:]-props[None,a,:])**2,axis=2)/2)
    brbf=np.exp(-np.mean((bz[a,None,:]-bz[None,a,:])**2,axis=2)/2)
    mats={
      'exact':np.eye(t), 'pos':same.astype(float),
      'local':same*(diff@diff.T), 'rbf':same*rbf,
      'blosum':same*(bd@bd.T), 'brbf':same*brbf,
      'target':(a[:,None]==a[None,:]).astype(float),
      'sub':(w[:,None]==w[None,:])*(a[:,None]==a[None,:]),
      'chem':diff@diff.T,
      'regional':samereg*(diff@diff.T),
      'regtarget':samereg*(a[:,None]==a[None,:]),
      'wtchem':(w[:,None]==w[None,:])*(diff@diff.T),
      'wtblosum':(w[:,None]==w[None,:])*(bd@bd.T),
    }
    # Shared mutation severity indicators; fitted signs, not assumed outcomes.
    f=[]
    for pos,wt,mut in keys:
        x=props[AA.index(mut)]-props[AA.index(wt)]
        f.append([float(mut=='P'),float(wt=='P'),float(mut=='G'),float(wt=='G'),
          float(wt in 'WFY' and mut not in 'WFY'),float(wt in 'WFY' and mut in 'WFY'),
          float(wt in 'ILVM' and mut not in 'ILVM'),float(wt in 'DE'),float(mut in 'KR'),
          float(mut in 'DE'),float(mut=='P' and region(pos) in [1,3,5]),
          BLOSUM[AA.index(wt),AA.index(mut)],abs(x[0]),abs(x[1]),x[2]])
    f=np.array(f,float)
    f=f/np.maximum(np.sqrt(np.mean(f*f,axis=0)),1e-6)
    mats['severity']=f@f.T
    out={}
    for name,mat in mats.items():
        mat=mat/max(np.mean(np.diag(mat)),1e-6)
        out[name]=M@mat@M.T
    counts=M.sum(1)
    out['count']=np.outer(counts,counts)
    out['constant']=np.ones((n,n))
    return out

def target_transform(y,kind):
    y=np.asarray(y,float)
    if kind=='log':return np.log(np.maximum(y,1e-6))
    if kind=='raw':return y
    if kind=='sqrt':return np.sqrt(np.maximum(y,0))
    if kind=='quarter':return np.maximum(y,0)**.25
    if kind=='rank':return norm.ppf((rankdata(y)-.5)/len(y))
    if kind=='asinh':return np.arcsinh(y/.3)
    raise ValueError(kind)

def combine(ks, weights, epi=0):
    K=sum(ks[name]*v for name,v in weights.items() if v)
    if epi:
        # Degree two covariance: regularized context-dependent combinations.
        d=max(np.mean(np.diag(K)),1e-9)
        K=K+epi*K*K/d
    return K

def predict_kernel(K,train,test,y,alpha, intercept=2):
    A=K[np.ix_(train,train)]+intercept+alpha*np.eye(len(train))
    coef=cho_solve(cho_factor(A,lower=True,check_finite=False),y,check_finite=False)
    return (K[np.ix_(test,train)]+intercept)@coef
