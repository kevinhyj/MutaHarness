import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import ShuffleSplit
from model_features import blocks
c=json.load(open('case.json')); rows=c['support'];y=np.array([r['fitness'] for r in rows]);n=len(y)
b=blocks(rows+c['query'],c['reference_sequence'])
K={k:v[:n]@v[:n].T for k,v in b.items()}
# Center features in fit only, including intercept via centered kernel.
splits=[]
for rep in range(20):
 rng=np.random.default_rng(470+rep); val=rng.choice(np.arange(32,n),size=20,replace=False); tr=np.setdiff1d(np.arange(n),val);splits.append((tr,val))
for rep in range(10):
 rng=np.random.default_rng(980+rep);val=rng.choice(n,size=24,replace=False);tr=np.setdiff1d(np.arange(n),val);splits.append((tr,val))
def cv(K,alpha,trans='raw'):
 preds=[];metrics=[]
 for tr,va in splits:
  yt=y[tr].copy()
  if trans=='rank':yt=(rankdata(yt)-0.5)/len(yt)
  if trans=='asinh':yt=np.arcsinh(yt/.3)
  if trans=='exp':yt=np.exp(yt)
  if trans=='exp2':yt=np.exp(2*yt)
  if trans=='signed':yt=np.sign(yt)*np.sqrt(abs(yt))
  kt=K[np.ix_(tr,tr)];kv=K[np.ix_(va,tr)];mu=kt.mean(0);grand=mu.mean()
  kt=kt-mu[:,None]-mu[None,:]+grand
  kv=kv-kv.mean(1)[:,None]-mu[None,:]+grand
  p=kv@solve(kt+alpha*np.eye(len(tr)),yt-yt.mean(),assume_a='pos')
  preds.append(p);metrics.append(spearmanr(y[va],p).statistic)
 return np.mean(metrics[:20]),np.mean(metrics[20:]),preds,metrics
configs=[]
for alpha in [.03,.1,.3,1,3,10]:
 for pw in [0,.1,.3,1,3]:
  k=K['exact']+pw*K['pos']+K['count']*.1
  configs.append((f'base_a{alpha}_p{pw}',k,alpha,'raw'))
for name in ['subst','aa','chem','local','region','struct']:
 for weight in [.03,.1,.3,1,3]:
  for alpha in [.1,1]:
   k=K['exact']+.3*K['pos']+K['count']*.1+weight*K[name]
   configs.append((f'{name}_w{weight}_a{alpha}',k,alpha,'raw'))
results=[];preds={};metric={}
for name,k,a,t in configs:
 f,r,p,m=cv(k,a,t);results.append((.75*f+.25*r,f,r,name));preds[name]=p;metric[name]=m
results.sort(reverse=True)
for x in results[:30]:print(x,flush=True)
np.savez('explore_results.npz',**{name:np.concatenate(p) for name,p in preds.items()})
json.dump(results,open('explore_results.json','w'),indent=2)
# target link functions on best kernels
for score,f,r,name in results[:5]:
 _,k,a,_=next(z for z in configs if z[0]==name)
 for t in ['rank','asinh','exp','exp2','signed']:
  f,r,p,m=cv(k,a,t);print('transform',name,t,round(f,4),round(r,4),flush=True)
