import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,itertools,pickle,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
import proposal
c=json.load(open('case.json'));rs=c['support'];y=np.array([r['fitness'] for r in rs]);n=len(rs);ref=c['reference_sequence']
fs=proposal.features(rs,ref);base=[f@f.T for f in fs]+[np.ones((n,n))]
WT=np.array([[float(a==b) for b in proposal.AA] for a in ref]);Kwt=fs[0]@WT@(fs[0]@WT).T
G=np.zeros((n,proposal.PROPS.shape[1]*4));wchem=np.zeros((n,20*proposal.PROPS.shape[1]))
for i,r in enumerate(rs):
 for p,a,b in proposal.mutations(r,ref):
  j=proposal.AA.index(a);k=proposal.AA.index(b);d=proposal.PROPS[k]-proposal.PROPS[j]
  G[i]+=np.r_[d,np.abs(d),proposal.PROPS[k],proposal.PROPS[j]]/3
  wchem[i,j*proposal.PROPS.shape[1]:(j+1)*proposal.PROPS.shape[1]]+=proposal.PROPS[k]/3
Kg=G@G.T;Kwc=wchem@wchem.T
sp=np.load('cv_splits.npz');splits=[(sp[f'tr{i}'],sp[f'te{i}']) for i in range(15)]
res=[];start=time.time()
for we,wc,wg,typ in itertools.product((0,.1),(.0,.25),(0,.5,2),('g','wc')):
 K=4*base[0]+we*base[1]+wc*base[2]+base[4]+base[5]+8*Kwt+wg*(Kg if typ=='g' else Kwc)
 for alpha,deg,robust in itertools.product((.1,1),(0,.3),('raw',.15,.3,.6,'rank')):
  KK=K+deg*K*K/4
  ps=[];scores=[]
  for tr,te in splits:
   yt=y[tr].copy();k=KK[np.ix_(tr,tr)];kt=KK[np.ix_(te,tr)]
   if robust=='rank':yt=rankdata(yt)/len(yt)-.5
   w=np.ones(len(tr))
   for step in range(8 if isinstance(robust,float) else 1):
    coef=solve(k+alpha*np.diag(1/w),yt,assume_a='pos')
    if isinstance(robust,float): w=np.minimum(1,robust/np.maximum(abs(yt-k@coef),1e-9))
   p=kt@coef;ps.append(p);scores.append(spearmanr(p,y[te]).statistic)
  res.append((.75*np.mean(scores[:10])+.25*np.mean(scores[10:]),np.mean(scores[:10]),np.mean(scores[10:]),(we,wc,wg,typ,alpha,deg,robust),ps,scores))
res.sort(key=lambda t:-t[0])
for r in res[:30]:print(r[:4])
with open('refine_results.pkl','wb') as f:pickle.dump(res,f)
print('elapsed',time.time()-start)
