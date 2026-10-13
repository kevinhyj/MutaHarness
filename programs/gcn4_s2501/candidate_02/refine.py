import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge,HuberRegressor,ElasticNet
import proposal as p
c=json.load(open('case.json'));ss=c['support'];y=np.array([s['fitness'] for s in ss]);orders=np.array([s['mutation_order'] for s in ss]);fs=p.features(ss,c['reference_sequence'])
def compact(records):
 out=[]
 for r in records:
  s=r['sequence'][100:144];a=np.array(list(s));g=np.array([[x in z for z in ['DE','KR','FWY','W','Y','F','LIMV','G','P','STNQ','D','E']] for x in s],float)
  f=list(g.sum(0));f.extend([sum(x!=z for x,z in zip(s,c['reference_sequence'][100:144])),sum(p.KD[x] for x in s)])
  for lo,hi in [(0,19),(19,25),(25,44)]:f.extend(g[lo:hi].sum(0))
  for k in [0,2,6]:
   f.extend([np.dot(g[:,k],np.arange(44))/max(1,g[:,k].sum())])
  out.append(f)
 return np.array(out)
fs['compact']=compact(ss)
fs['counts']=fs['comp'][:,:20]
# Low-dimensional nonlinear descriptors with controlled interaction between mutation classes.
fs['small']=fs['compact'][:,:14]
fs['poly']=np.concatenate([fs['small'],np.array([fs['small'][:,i]*fs['small'][:,j] for i in range(14) for j in range(i,14)]).T],axis=1)
fs['gated']=np.concatenate([fs['compact'],fs['compact']*(orders>20)[:,None]],axis=1)
singles=np.where(orders==1)[0];multi=np.where(orders>1)[0];low=multi[orders[multi]<10];high=multi[orders[multi]>20]
folds=[]
for seed in [41,113,227,501]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(multi):folds.append((np.r_[singles,multi[tr]],multi[te]))
results=[];preds=[]
def add(conf,q):
 rho=[spearmanr(y[multi],z[multi]).statistic for z in q]
 lo=np.mean([spearmanr(y[low],z[low]).statistic for z in q]);hi=np.mean([spearmanr(y[high],z[high]).statistic for z in q])
 results.append((np.mean(rho),np.std(rho),lo,hi,conf));preds.append(q)
for key in ['compact','counts','small','poly','gated','regional']:
 for alpha in [.03,.1,.3,1,3,10,30]:
  for kind in ['rank','raw']:
   for sw in [.05,.5,2]:
    q=np.zeros((4,len(ss)))
    for k,(tr,te) in enumerate(folds):
     X=fs[key][tr];Z=fs[key][te];mu=X.mean(0);sd=np.maximum(X.std(0),1.)
     X=(X-mu)/sd;Z=(Z-mu)/sd
     m=Ridge(alpha=alpha)
     m.fit(X,p.target_values(y[tr],kind),sample_weight=np.where(orders[tr]==1,sw,1.));q[k//4,te]=m.predict(Z)
    add(('ridge',key,alpha,kind,sw),q)
for key in ['compact','small','regional','gated']:
 for leaf in [2,4,8]:
  for kind in ['rank','raw']:
   for sw in [.05,.5,2]:
    q=np.zeros((4,len(ss)))
    for k,(tr,te) in enumerate(folds):
     m=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=.8,random_state=991,n_jobs=1)
     m.fit(fs[key][tr],p.target_values(y[tr],kind),sample_weight=np.where(orders[tr]==1,sw,1.));q[k//4,te]=m.predict(fs[key][te])
    add(('extra',key,leaf,kind,sw),q)
    print('done',results[-1],flush=True)
ix=np.argsort([r[0] for r in results])[::-1]
print('BEST',flush=True)
for j in ix[:35]:print(results[j],flush=True)
np.savez('refine_validation.npz',preds=np.array(preds),y=y,orders=orders)
json.dump(results,open('refine_results.json','w'))
