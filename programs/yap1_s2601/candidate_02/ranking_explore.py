import json
import numpy as np
from scipy.special import expit
from scipy.linalg import cho_factor,cho_solve,solve
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from scientific_model import kernels,combine,predict_kernel,target_transform

def rank_fit(K,train,test,y,reg=1,power=0):
 A=K[np.ix_(train,train)]+.001*np.eye(len(train))
 inv=cho_solve(cho_factor(A,lower=True),np.eye(len(train)))
 order=np.argsort(y,kind='stable'); ii,jj=np.triu_indices(len(y),1); lo=order[ii];hi=order[jj]
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
if __name__=='__main__':
 c=json.load(open('case.json'));s=c['support'];n=len(s);y=np.array([r['fitness'] for r in s]);ks=kernels(s+c['query'],c['reference_sequence']);folds=[]
 for seed in [17,41,73]:
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,n)):folds.append((np.r_[np.arange(32),32+tr],32+te))
 results=[]
 for w in [{'exact':1,'pos':.3,'local':.2,'severity':1,'target':1},{'exact':1,'pos':1,'rbf':1,'severity':1,'target':1},{'exact':1,'pos':.3,'local':.2,'severity':3,'target':3}]:
  K=combine(ks,w)
  for reg in [.01,.03,.1,.3,1,3]:
   for power in [0,1]:
    vals=[spearmanr(rank_fit(K,tr,te,y[tr],reg,power),y[te]).statistic for tr,te in folds]
    r=(float(np.mean(vals)),w,reg,power);results.append(r);print(r,flush=True)
 results.sort(key=lambda x:-x[0]);json.dump(results,open('ranking_results.json','w'))
