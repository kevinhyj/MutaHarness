import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,itertools,pickle,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
import proposal
c=json.load(open('case.json'));rs=c['support'];y=np.array([r['fitness'] for r in rs]);n=len(rs)
fs=proposal.features(rs,c['reference_sequence']);base=[f@f.T for f in fs]+[np.ones((n,n))]
# Sequence-only, explicitly assumed SH3 secondary-structure hypotheses.
ref=c['reference_sequence'];L=len(ref)
WT=np.array([[float(a==b) for b in proposal.AA] for a in ref]);Kwt=fs[0]@WT@(fs[0]@WT).T
regions=[(159,165),(166,175),(176,180),(181,190),(191,194),(195,201),(202,206),(207,209),(210,214)]
R=np.zeros((L,len(regions)))
for j,(a,b) in enumerate(regions): R[a-1:b,j]=1
Kr=fs[0]@R@(fs[0]@R).T
# transfer site+chem by WT chemistry, no positional annotations
wcfeat=np.zeros((n,20*proposal.PROPS.shape[1]))
for i,r in enumerate(rs):
 for p,a,b in proposal.mutations(r,ref):
  j=proposal.AA.index(a);wcfeat[i,j*proposal.PROPS.shape[1]:(j+1)*proposal.PROPS.shape[1]]+=proposal.PROPS[proposal.AA.index(b)]/3
Kwc=wcfeat@wcfeat.T
# Local sequence interpolation, same parity favored at distance two.
positions=np.arange(L);dist=abs(positions[:,None]-positions)
local=np.exp(-dist/3)*(1+(dist%2==0))/2
Kn=fs[0]@local@fs[0].T
spl=np.load('cv_splits.npz');splits=[(spl[f'tr{i}'],spl[f'te{i}']) for i in range(15)];res=[];start=time.time()
for ws,we,wc,alpha,deg in itertools.product((4,16,64),(.0,.1,1),(.0,.25,1),(0.1,.5,2),(0,.1,.3,1)):
 K=ws*base[0]+we*base[1]+wc*base[2]+base[4]+base[5]
 if deg: K+=deg*K*K/4
 scores=[];ps=[]
 for tr,te in splits:
  p=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),y[tr],assume_a='pos');ps.append(p);scores.append(spearmanr(p,y[te]).statistic)
 res.append((.75*np.mean(scores[:10])+.25*np.mean(scores[10:]),np.mean(scores[:10]),np.mean(scores[10:]),('base',ws,we,wc,alpha,deg),ps,scores))
res.sort(key=lambda t:-t[0]);basebest=res[0][3];print('basebest',basebest,flush=True)
_,ws,we,wc,alpha,deg=basebest
for typ,T in [('wt',Kwt),('region',Kr),('wtchem',Kwc),('neighbor',Kn)]:
 for weight,al,dg in itertools.product((.1,.5,2,8),(.1,.5,2),(0,.3)):
  K=ws*base[0]+we*base[1]+wc*base[2]+base[4]+base[5]+weight*T
  if dg: K+=dg*K*K/4
  ps=[];scores=[]
  for tr,te in splits:
   p=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+al*np.eye(len(tr)),y[tr],assume_a='pos');ps.append(p);scores.append(spearmanr(p,y[te]).statistic)
  res.append((.75*np.mean(scores[:10])+.25*np.mean(scores[10:]),np.mean(scores[:10]),np.mean(scores[10:]),(typ,ws,we,wc,al,dg,weight),ps,scores))
res.sort(key=lambda t:-t[0])
for r in res[:25]:print(r[:4])
with open('transfer_results.pkl','wb') as f:pickle.dump(res,f)
print('elapsed',time.time()-start)
