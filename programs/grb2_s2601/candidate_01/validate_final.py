import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata,spearmanr
from sklearn.model_selection import ShuffleSplit
from nonlinear import pooled_predict
import proposal
c=json.load(open('case.json'));rs=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in rs]);fs=proposal.features(rs,ref);base=[f@f.T for f in fs]+[np.ones((len(rs),len(rs)))];WT=np.array([[float(a==b) for b in proposal.AA] for a in ref]);Kw=fs[0]@WT@(fs[0]@WT).T
wchem=np.zeros((len(rs),20*proposal.PROPS.shape[1]))
for i,r in enumerate(rs):
 for p,a,b in proposal.mutations(r,ref):
  j=proposal.AA.index(a);wchem[i,j*proposal.PROPS.shape[1]:(j+1)*proposal.PROPS.shape[1]]+=proposal.PROPS[proposal.AA.index(b)]/3
Kwc=wchem@wchem.T
orig=np.array([i for i,r in enumerate(rs) if r['id'].startswith('S')]);feed=np.array([i for i,r in enumerate(rs) if not r['id'].startswith('S')])
splits=[]
for tr,te in ShuffleSplit(n_splits=25,test_size=.25,random_state=1203).split(feed):splits.append(('feedback',np.r_[orig,feed[tr]],feed[te]))
for tr,te in ShuffleSplit(n_splits=20,test_size=.25,random_state=1409).split(y):splits.append(('mixed',tr,te))
for tr,te in ShuffleSplit(n_splits=8,test_size=.50,random_state=1591).split(y):splits.append(('half',tr,te))
configs=[(4,0,0,8,.3,.1),(4,0,0,0,.1,.1),(4,0,0,2,0,2),(4,.1,.25,8,.3,1)]
allres=[];start=time.time()
for typ,tr,te in splits:
 preds=[]
 for j,(ws,we,wc,wt,deg,alpha) in enumerate(configs):
  K=ws*base[0]+we*base[1]+wc*base[2]+base[4]+base[5]+wt*Kw+(0.5*Kwc if j==3 else 0);K+=deg*K*K/4;k=K[np.ix_(tr,tr)];kt=K[np.ix_(te,tr)];w=np.ones(len(tr))
  for _ in range(8 if j==3 else 1):
   cf=solve(k+alpha*np.diag(1/w),y[tr],assume_a='pos')
   if j==3:w=np.minimum(1,.15/np.maximum(abs(y[tr]-k@cf),1e-9))
  preds.append(kt@cf)
 for alpha in (1,.2):preds.append(pooled_predict([rs[i] for i in tr],[rs[i] for i in te],ref,ws=4,wc=.25,alpha=alpha,t=.05,mix=1))
 ranks=[rankdata(p)/len(p) for p in preds]
 ens3=.5*ranks[0]+.25*ranks[4]+.25*ranks[5]
 ens6=(2*ranks[0]+ranks[1]+ranks[2]+ranks[3]+2*ranks[4]+ranks[5])/8
 conservative=.5*ranks[0]+.25*ranks[2]+.25*ranks[4]
 ss=[spearmanr(p,y[te]).statistic for p in preds+[ens3,ens6,conservative]]
 allres.append((typ,ss))
for typ in ('feedback','mixed','half'):
 arr=np.array([ss for t,ss in allres if t==typ]);print(typ,'n',len(arr),'means',np.round(arr.mean(0),4),'sd',np.round(arr.std(0),4))
np.savez('fresh_validation.npz',types=np.array([t for t,ss in allres]),scores=np.array([ss for t,ss in allres]))
print('elapsed',time.time()-start)
