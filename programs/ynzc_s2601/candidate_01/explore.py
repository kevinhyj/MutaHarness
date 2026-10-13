import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold,StratifiedKFold
from proposal import features
c=json.load(open('case.json')); records=c['support']; ref=c['reference_sequence']; n=len(records)
y=np.array([r['fitness'] for r in records]); order=np.array([r['mutation_order'] for r in records]); feedback=np.arange(32,n)
f=features(records,ref)
# Extra transferable residue class features and location-dependent interactions.
mut=f['mut'].reshape(n,len(ref),20)
f['amino']=mut.sum(axis=1)
p=8; ph=f['phys'].reshape(n,len(ref),p)
f['localcross']=np.concatenate([ph[:,a,:]*ph[:,b,:] for a in range(len(ref)) for b in range(a+1,len(ref))],axis=1)
keys=list(f); kernels=np.array([f[k]@f[k].T for k in keys])
# Three repeated training regimes: randomized all-support folds; keep original
# single anchors and hold out independently sampled feedback; thinner supports.
splits=[]; regimes=[]
for seed in [11,29,61]:
 for tr,te in StratifiedKFold(4,shuffle=True,random_state=seed).split(np.arange(n),order): splits.append((tr,te));regimes.append('all')
 for tr0,te0 in StratifiedKFold(4,shuffle=True,random_state=seed).split(feedback,order[feedback]):
  splits.append((np.r_[np.arange(32),feedback[tr0]],feedback[te0]));regimes.append('feedback')
# Precenter each component independently using only training records.
cache=[]
for tr,te in splits:
 kt=kernels[:,tr][:,:,tr]; qt=kernels[:,te][:,:,tr]
 mean=kt.mean(axis=1); grand=mean.mean(axis=1)
 kc=kt-mean[:,:,None]-mean[:,None,:]+grand[:,None,None]
 qc=qt-qt.mean(axis=2,keepdims=True)-mean[:,None,:]+grand[:,None,None]
 cache.append((kc,qc,tr,te))

def evaluate(weights,alpha,target='raw',ret=False):
 preds=[np.full(n,np.nan) for _ in range(6)]
 for ix,(kc,qc,tr,te) in enumerate(cache):
  z=y[tr].copy()
  if target=='rank': z=rankdata(z)/(len(z)+1)
  if target=='asinh': z=np.arcsinh(z)
  zm=z.mean(); zs=z.std() or 1; z=(z-zm)/zs
  K=np.einsum('k,kij->ij',weights,kc); Q=np.einsum('k,kij->ij',weights,qc)
  a=solve(K+alpha*np.eye(len(tr)),z,assume_a='pos',check_finite=False)
  v=(Q@a)*zs+zm
  # Normalize per fold to preserve interpretable OOF scales.
  if target=='rank': v=v
  preds[ix//4][te]=v
 metrics=[]
 for i,v in enumerate(preds):
  mask=np.isfinite(v); metrics.append(spearmanr(y[mask],v[mask]).statistic)
 double=[]
 for v in preds:
  mask=np.isfinite(v)&(order==2);double.append(spearmanr(y[mask],v[mask]).statistic)
 objective=np.mean(metrics)*.75+np.mean(double)*.25
 if ret:return np.array(preds),metrics,double
 return objective,float(np.mean(metrics)),float(np.mean(double))

rng=np.random.default_rng(824);results=[]
# Broad log-scale search, with sparse groups to expose unnecessary hypotheses.
for t in range(450):
 weights=np.array([1.,10**rng.uniform(-1.5,1.5),10**rng.uniform(-1.5,1.5),10**rng.uniform(-2,1),10**rng.uniform(-3,0),10**rng.uniform(-3,0),10**rng.uniform(-4,-.2),10**rng.uniform(-3,0),10**rng.uniform(-4,-.2)])
 for j in [4,5,6,7,8]:
  if rng.random()<.35:weights[j]=0
 alpha=10**rng.uniform(-1.6,1.1)
 target='raw'
 score=evaluate(weights,alpha,target)
 results.append({'objective':score[0],'overall':score[1],'double':score[2],'weights':dict(zip(keys,weights.tolist())),'alpha':alpha,'target':target})
 if t%100==0: print(t, max(r['objective'] for r in results),flush=True)
results.sort(key=lambda r:r['objective'],reverse=True)
json.dump(results,open('cv_search.json','w'),indent=2)
for r in results[:12]: print(r,flush=True)
