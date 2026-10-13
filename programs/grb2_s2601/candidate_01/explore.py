import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,itertools
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import ShuffleSplit
import proposal
c=json.load(open('case.json')); rs=c['support']; y=np.array([r['fitness'] for r in rs]);n=len(y)
fs=proposal.features(rs,c['reference_sequence'])
base=[f@f.T for f in fs]+[np.ones((n,n))]
# Same-position Gaussian chemical similarity, with site indicator for changed residues.
mm=[proposal.mutations(r,c['reference_sequence']) for r in rs]
desc=proposal.PROPS
chemks=[]
for length in (0.5,1,2):
 ak=np.exp(-np.sum((desc[:,None,:]-desc[None,:,:])**2,axis=-1)/(2*length**2))
 k=np.zeros((n,n))
 for i,ms in enumerate(mm):
  for j,ns in enumerate(mm):
   k[i,j]=sum(ak[proposal.AA.index(b),proposal.AA.index(d)] for p,a,b in ms for q,cc,d in ns if p==q)
 chemks.append(k)
# Original single supports always retained in feedback validation; separately evaluate all-data splits.
orig=np.array([i for i,r in enumerate(rs) if r['id'].startswith('S')]); feed=np.array([i for i,r in enumerate(rs) if not r['id'].startswith('S')])
splits=[]
for tr,te in ShuffleSplit(n_splits=10,test_size=.25,random_state=42).split(feed):splits.append((np.r_[orig,feed[tr]],feed[te]))
for tr,te in ShuffleSplit(n_splits=5,test_size=.25,random_state=73).split(y):splits.append((tr,te))
np.savez('cv_splits.npz',**{f'tr{i}':tr for i,(tr,te) in enumerate(splits)},**{f'te{i}':te for i,(tr,te) in enumerate(splits)})
results=[]; start=time.time()
for typ in ('linear','rbfchem'):
 for ws,we,wc,wg,deg in itertools.product((.25,1,4),(0.1,1),(0.25,1,4),(.0,.3),(0,.3)):
  if typ=='linear': K=ws*base[0]+we*base[1]+wc*base[2]+wg*base[3]+base[4]+base[5]
  else: K=ws*base[0]+we*base[1]+wc*chemks[1]+wg*base[3]+base[4]+base[5]
  if deg: K=K+deg*K*K/4
  for alpha in (.1,.5,2,8):
   for trans in ('raw','logit'):
    ps=[];ss=[]
    for tr,te in splits:
     yt=y[tr]
     if trans=='logit':
      lo=min(yt.min()-.1,-1.6);hi=max(yt.max()+.1,.35)
      t=np.clip((yt-lo)/(hi-lo),.001,.999);yt=np.log(t/(1-t))
     p=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),yt,assume_a='pos')
     ps.append(p);ss.append(spearmanr(p,y[te]).statistic)
    avg=.75*np.mean(ss[:10])+.25*np.mean(ss[10:])
    results.append((avg,np.mean(ss[:10]),np.mean(ss[10:]),(typ,ws,we,wc,wg,deg,alpha,trans),ps,ss))
results.sort(key=lambda t:-t[0])
for r in results[:25]:print(r[:4])
print('best ensemble',[(k,np.mean([spearmanr(np.mean([rankdata(r[4][i]) for r in results[:k]],axis=0),y[te]).statistic for i,(tr,te) in enumerate(splits)][:10])) for k in (1,3,10,30)])
import pickle
with open('cv_results.pkl','wb') as f:pickle.dump(results,f)
print('elapsed',time.time()-start)
