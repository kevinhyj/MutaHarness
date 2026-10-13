import json,itertools
import numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold
from scientific_model import kernels,combine,predict_kernel,target_transform
c=json.load(open('case.json'));s=c['support'];n=len(s);y=np.array([r['fitness'] for r in s]);ks=kernels(s+c['query'],c['reference_sequence'])
orig=np.arange(32);fb=np.arange(32,n);folds=[]
for seed in [17,41,73]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(fb):folds.append((np.r_[orig,fb[tr]],fb[te]))
def ev(w,alpha,trans,epi=0,intercept=2):
 K=combine(ks,w,epi);vals=[]
 for tr,te in folds:
  yy=target_transform(y[tr],trans)
  p=predict_kernel(K,tr,te,yy,alpha,intercept)
  vals.append(spearmanr(p,y[te]).statistic)
 return float(np.mean(vals))
r=[]
for pv,sev,tar,alpha,trans in itertools.product([.1,.3,1],[.3,1,3],[.3,1,3],[.03,.1,.4,1],['log','quarter','sqrt','raw','rank','asinh']):
 w={'exact':1,'pos':pv,'local':.2,'severity':sev,'target':tar}
 score=ev(w,alpha,trans);r.append((score,w,alpha,trans,0))
r.sort(key=lambda x:-x[0]);print('REFINED',flush=True)
for x in r[:25]:print(x,flush=True)
# One-extra-kernel checks, with fixed stronger baseline.
r2=[]
for _,w,alpha,trans,_ in r[:4]:
 for name,val in itertools.product(['sub','chem','regional','regtarget','wtchem','wtblosum','count','rbf','blosum'],[.3,1,3]):
  ww=dict(w);ww[name]=val
  for epi in [0,.2]:
   r2.append((ev(ww,alpha,trans,epi),ww,alpha,trans,epi))
r+=r2;r.sort(key=lambda x:-x[0]);json.dump(r,open('refined_results.json','w'))
print('EXTENDED',flush=True)
for x in r[:25]:print(x,flush=True)
