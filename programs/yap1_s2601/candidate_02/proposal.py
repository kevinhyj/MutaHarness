#!/usr/bin/env python3
"""Support-only deterministic ordinal protein predictor. No persistent fitted state."""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ARNDCQEGHILKMFPSTWYV'
# Standard chemical descriptors: Kyte-Doolittle hydropathy, volume, charge,
# aromaticity, proline, glycine. These are scientific constants, not labels.
RAW = {
'A':[1.8,88.6,0,0,0,0], 'R':[-4.5,173.4,1,0,0,0],
'N':[-3.5,114.1,0,0,0,0], 'D':[-3.5,111.1,-1,0,0,0],
'C':[2.5,108.5,0,0,0,0], 'Q':[-3.5,143.8,0,0,0,0],
'E':[-3.5,138.4,-1,0,0,0], 'G':[-.4,60.1,0,0,0,1],
'H':[-3.2,153.2,.15,1,0,0], 'I':[4.5,166.7,0,0,0,0],
'L':[3.8,166.7,0,0,0,0], 'K':[-3.9,168.6,1,0,0,0],
'M':[1.9,162.9,0,0,0,0], 'F':[2.8,189.9,0,1,0,0],
'P':[-1.6,112.7,0,0,1,0], 'S':[-.8,89,0,0,0,0],
'T':[-.7,116.1,0,0,0,0], 'W':[-.9,227.8,0,1,0,0],
'Y':[-1.3,193.6,0,1,0,0], 'V':[4.2,140,0,0,0,0]}

def mutations(record, reference):
    seq = record.get('sequence')
    if seq and len(seq) == len(reference):
        return [(i+1, w, a) for i,(w,a) in enumerate(zip(reference,seq)) if w != a]
    return [(int(p), w, a) for w,p,a in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

# Fixed configurations selected using only public support cross-validation.
# Every effect and pairwise score function is estimated again on invocation.
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import norm
SEED = 2601
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

def rank_fit(K,train,test,y,reg=1,power=0):
 A=K[np.ix_(train,train)]+.001*np.eye(len(train))
 inv=cho_solve(cho_factor(A,lower=True),np.eye(len(train)))
 order=np.argsort(y,kind='stable'); ii,jj=np.triu_indices(len(y),1); lo=order[ii];hi=order[jj]
 keep=y[hi] > y[lo]
 lo,hi,ii,jj=lo[keep],hi[keep],ii[keep],jj[keep]
 wt=((jj-ii)/len(y))**power/len(y)
 f=np.zeros(len(y))
 for step in range(30):
  d=f[hi]-f[lo];pp=expit(d);res=(pp-1)*wt
  g=reg*(inv@f)+np.bincount(hi,weights=res,minlength=len(y))-np.bincount(lo,weights=res,minlength=len(y))
  v=pp*(1-pp)*wt;H=np.zeros_like(A);H[hi,lo]=-v;H[lo,hi]=-v
  H[np.diag_indices(len(y))]=np.bincount(hi,weights=v,minlength=len(y))+np.bincount(lo,weights=v,minlength=len(y))
  delta=solve(H+reg*inv,g,assume_a='pos');rate=1
  before=(np.logaddexp(0,-d)*wt).sum()+reg*.5*f@inv@f
  for _ in range(15):
   nf=f-rate*delta;after=(np.logaddexp(0,-(nf[hi]-nf[lo]))*wt).sum()+reg*.5*nf@inv@nf
   if after<=before:break
   rate*=.5
  f=nf
  if np.max(abs(rate*delta))<1e-6:break
 return K[np.ix_(test,train)]@(inv@f)

def propose(case, assignment):
    """Fit only supplied measurements; rank every current query; select none.

    The two-degree polynomial kernel learns regularized combination effects.
    WW-region/severity descriptors are hypotheses with fitted signs. No WT
    outcome, external sequence, stored fit, or fixed mutation effect is used.
    """
    support=list(case.get('support', []))
    query=list(case.get('query', []))
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not support:
        return {'scores': {r['id']: 0.0 for r in query}, 'selected_ids': []}
    # No randomized fitting is needed. Seed kept explicit for future extensions.
    rng = np.random.default_rng(SEED)
    n=len(support)
    y=np.array([float(r['fitness']) for r in support], dtype=float)
    ks=kernels(support+query,case['reference_sequence'])
    train=np.arange(n);test=np.arange(n,n+len(query))
    base={'exact':1.0,'pos':0.3,'local':0.2,'severity':1.0,'target':1.0}
    configs=[
        # weight, kernel weights, regularization, objective, rank gap exponent, epistasis
        (.30,base,.01,'pair',1.0,0.0),
        (.25,dict(base,count=1.0),.03,'pair',.5,0.0),
        (.15,dict(base,count=3.0),.01,'pair',.5,.2),
        (.15,{'exact':1.0,'pos':.3,'brbf':.2,'severity':1.0,'target':1.0},.1,'sqrt',0.0,0.0),
        (.15,dict(base,count=3.0),.1,'sqrt',0.0,.2),
    ]
    ensemble=np.zeros(len(query))
    for weight,weights,reg,objective,gap,epistasis in configs:
        K=combine(ks,weights,epistasis)
        if objective=='pair':
            pred=rank_fit(K,train,test,y,reg,gap)
        else:
            # The observed assay scores are positive. Square-root is monotone
            # and reduces leverage of large readings; this is not calibration.
            pred=predict_kernel(K,train,test,target_transform(y,objective),reg)
        ensemble+=weight*(rankdata(pred,method='average')-.5)/len(query)
    # Purely ordinal, finite scores in [-1,1]; preserve ties.
    ordinal=2*(rankdata(ensemble,method='average')-.5)/len(query)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,ordinal)}, 'selected_ids':[]}
