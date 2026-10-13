"""Support-only deterministic ordinal protein predictor.

Amino-acid constants are ordinary biochemical descriptors, not trained scores.
All fitted effects are estimated anew from the supplied support records.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Hydropathy, side-chain volume, charge, aromatic, H-bond donor/acceptor,
# and special backbone geometry indicators. Fixed scientific descriptors.
HYD = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]
def _props():
    rows=[]
    for a,h,v in zip(AA,HYD,VOL):
        rows.append([h/4.5,(v-140)/65, float(a in 'KR')-float(a in 'DE'),
                     float(a in 'FWY'),float(a in 'STNQH'),float(a=='G'),
                     float(a=='P'),float(a=='C'),float(a in 'ILV')])
    return np.array(rows)
PROPS=_props()

def mutations(r, ref):
    if r.get('sequence') and len(r['sequence']) == len(ref):
        return [(p,a,b) for p,(a,b) in enumerate(zip(ref,r['sequence'])) if a!=b]
    return [(int(p)-1,a,b) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',r.get('mutation',''))]

def features(records, ref):
    n=len(records); L=len(ref)
    site=np.zeros((n,L)); exact=np.zeros((n,L*20)); chem=np.zeros((n,L*len(PROPS[0])))
    globalchem=np.zeros((n,20*20)); count=np.zeros((n,1))
    for i,r in enumerate(records):
        mm=mutations(r,ref); count[i]=len(mm)
        for p,a,b in mm:
            if b not in AA or a not in AA: continue
            j=AA.index(b); k=AA.index(a)
            site[i,p]=1; exact[i,p*20+j]=1
            # Destination chemistry (site effect handles WT-relative offset).
            chem[i,p*PROPS.shape[1]:(p+1)*PROPS.shape[1]]=PROPS[j]/np.sqrt(PROPS.shape[1])
            globalchem[i,k*20+j]+=1
    return site,exact,chem,globalchem,count

def predict_records(support, query, ref, config=None):
    if not support: return np.zeros(len(query))
    cfg=config or (1.,1.,0.5,0.1,1.,0.3,0.)
    ws,we,wc,wg,wn,alpha,degree=cfg
    allr=list(support)+list(query); fs=features(allr,ref); n=len(support)
    X=np.concatenate([fs[0]*np.sqrt(ws),fs[1]*np.sqrt(we),fs[2]*np.sqrt(wc),
                      fs[3]*np.sqrt(wg),fs[4]*np.sqrt(wn),np.ones((len(allr),1))],axis=1)
    K=X@X.T
    if degree: K=K+degree*K*K/4
    y=np.array([float(r['fitness']) for r in support])
    coef=solve(K[:n,:n]+alpha*np.eye(n),y,assume_a='pos')
    return K[n:,:n]@coef


from scipy.linalg import cholesky
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

def mutation_features(tokens,ref,ws=8.,we=.1,wc=.25):
 L=len(ref);X=np.zeros((len(tokens),L*(21+PROPS.shape[1])))
 for i,(p,a,b) in enumerate(tokens):
  j=AA.index(b); X[i,p]=np.sqrt(ws)
  X[i,L+p*20+j]=np.sqrt(we)
  X[i,L*21+p*PROPS.shape[1]:L*21+(p+1)*PROPS.shape[1]]=np.sqrt(wc/PROPS.shape[1])*PROPS[j]
 return X

def pooled_predict(rs, qs, ref, ws=8.,we=.1,wc=.25,alpha=.5,t=.1,mix=1):
 muts=[mutations(r,ref) for r in rs+qs]
 tokens=sorted(set(m for ms in muts[:len(rs)] for m in ms)); ids={m:i for i,m in enumerate(tokens)}
 X=mutation_features(tokens,ref,ws,we,wc); K=X@X.T
 C=cholesky(K+1e-8*np.eye(len(K)),lower=True)
 mmax=max([len(ms) for ms in muts]+[1]); idx=np.zeros((len(rs),mmax),int);mask=np.zeros_like(idx,dtype=bool)
 for i,ms in enumerate(muts[:len(rs)]):
  for j,m in enumerate(ms):idx[i,j]=ids[m];mask[i,j]=True
 y=np.array([r['fitness'] for r in rs]); counts=mask.sum(1)
 # Ridge additive start, same regularizer in the whitened mutation space.
 A=np.zeros((len(rs),len(tokens)))
 for i in range(len(rs)):
  for j in idx[i,mask[i]]: A[i,j]+=1
 H=np.c_[A@C,np.ones(len(rs)),counts-1]
 reg=np.r_[np.full(len(tokens),alpha),.01,.3]
 initial=solve(H.T@H+np.diag(reg),H.T@y,assume_a='pos')
 def objective(v):
  z=C@v[:-2]; zv=z[idx]; logits=np.where(mask,-zv/t,-1e6)
  probs=softmax(logits,axis=1)
  sm=np.where(counts>0,-t*(logsumexp(logits,axis=1)-np.log(np.maximum(counts,1))),0.)
  pred=(1-mix)*np.sum(zv*mask,axis=1)+mix*sm+v[-2]+v[-1]*(counts-1)
  residual=pred-y
  gradz=np.zeros(len(tokens)); g=((1-mix)*mask+mix*probs)*mask
  np.add.at(gradz,idx.ravel(),(residual[:,None]*g).ravel())
  grad=np.r_[C.T@gradz,residual.sum(),residual@(counts-1)]+reg*v
  return .5*(residual@residual+np.sum(reg*v*v)),grad
 fit=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':250,'ftol':1e-9,'gtol':1e-6})
 # Cross-kernel extension of latent mutation values. No labels from queries.
 qt=sorted(set(m for ms in muts[len(rs):] for m in ms)); qids={m:i for i,m in enumerate(qt)}
 Q=mutation_features(qt,ref,ws,we,wc)
 z=(Q@X.T)@solve(C.T,fit.x[:-2],assume_a='gen')
 out=[]
 for ms in muts[len(rs):]:
  zz=np.array([z[qids[m]] for m in ms]); n=len(zz)
  sm=-t*(logsumexp(-zz/t)-np.log(n)) if n else 0
  out.append((1-mix)*zz.sum()+mix*sm+fit.x[-2]+fit.x[-1]*(n-1))
 return np.array(out)


def _position_kernel(records, ref, wt_weight=8., degree=.3):
    """PSD site, WT-identity, mutation-count and degree-two interaction kernel.

    The quadratic term is a shrinkage prior on interactions, not a claim of
    a physical contact or identified molecular epistasis.
    """
    site=np.zeros((len(records),len(ref)))
    for i,r in enumerate(records):
        for p,a,b in mutations(r,ref):
            site[i,p]+=1
    wt=np.zeros((len(ref),len(AA)))
    for p,a in enumerate(ref):
        if a in AA: wt[p,AA.index(a)]=1
    wts=site@wt
    counts=site.sum(axis=1)
    K=4*(site@site.T)+wt_weight*(wts@wts.T)+np.outer(counts,counts)+1
    return K+degree*K*K/4


def _ridge_query(support, query, ref, wt_weight, degree, alpha):
    records=list(support)+list(query);n=len(support)
    K=_position_kernel(records,ref,wt_weight,degree)
    y=np.array([float(r['fitness']) for r in support])
    cf=solve(K[:n,:n]+alpha*np.eye(n),y,assume_a='pos')
    return K[n:,:n]@cf


def propose(case, assignment):
    """Fit solely from the current measured records and rank every query.

    Validation-selected configurations (not fitted coefficients):
      50% site/WT degree-two ridge; 25% additive, more strongly regularized
      site/WT ridge; 25% smooth-min latent mutation effects.
    Rank averaging avoids assuming that component outputs are calibrated.
    There is no file I/O, cached model, or ID-dependent predictive feature.
    """
    np.random.seed(2026)
    query=case.get('query',case.get('queries',[]))
    support=[r for r in case.get('support',case.get('supports',[]))
             if 'fitness' in r and np.isfinite(float(r['fitness']))]
    if not query: return {'scores':{},'selected_ids':[]}
    if not support:
        return {'scores':{r['id']:0.0 for r in query},'selected_ids':[]}
    ref=case['reference_sequence']
    p1=_ridge_query(support,query,ref,wt_weight=8.,degree=.3,alpha=.1)
    p2=_ridge_query(support,query,ref,wt_weight=2.,degree=0.,alpha=2.)
    p3=pooled_predict(list(support),list(query),ref,ws=4.,we=.1,wc=.25,
                      alpha=1.,t=.05,mix=1.)
    nr=len(query)
    blended=sum(w*(rankdata(p,method='average')-.5)/nr
                for w,p in [(0.5,p1),(.25,p2),(.25,p3)])
    ordinal=2*(rankdata(blended,method='average')-.5)/nr-1
    return {'scores':{r['id']:float(s) for r,s in zip(query,ordinal)},
            'selected_ids':[]}
