import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import proposal as p
c=json.load(open('case.json'));ss=c['support'];y=np.array([s['fitness'] for s in ss]);orders=np.array([s['mutation_order'] for s in ss]);fs=p.features(ss,c['reference_sequence'])
def more(records):
 out=[]
 for r in records:
  s=r['sequence'][100:144];f=[]
  for lo,hi in [(0,44),(0,19),(19,25),(25,44),(0,10),(10,19),(19,32),(32,44),(0,25),(15,30),(0,22),(22,44)]:f.extend([s[lo:hi].count(x) for x in p.AA])
  out.append(f)
 return np.array(out,float)
fs['aacomp']=more(ss)
fs['mix']=np.concatenate([fs[k] for k in ['comp','regional','interactions']],axis=1)
fs['rich']=np.concatenate([fs[k] for k in ['aacomp','regional','interactions']],axis=1)
singles=np.where(orders==1)[0];multi=np.where(orders>1)[0];low=multi[orders[multi]<10];high=multi[orders[multi]>20]
folds=[]
for seed in [41,113,227,501]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(multi):folds.append((np.r_[singles,multi[tr]],multi[te]))
results=[];preds=[]
def add(conf,q):
 rho=[spearmanr(y[multi],z[multi]).statistic for z in q]
 lo=np.mean([spearmanr(y[low],z[low]).statistic for z in q]);hi=np.mean([spearmanr(y[high],z[high]).statistic for z in q])
 results.append((np.mean(rho),np.std(rho),lo,hi,conf));preds.append(q)
for key in ['mix','rich','aacomp']:
 for model in ['extra','boost','ridge']:
  for par in ([3,6,10] if model!='ridge' else [1,10,100]):
   for kind in ['rank','raw']:
    for sw in [.1,1,3]:
     q=np.zeros((4,len(ss)))
     for k,(tr,te) in enumerate(folds):
      if model=='extra':m=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=par,max_features=.8,random_state=991,n_jobs=1)
      if model=='boost':m=GradientBoostingRegressor(n_estimators=150,max_depth=2,min_samples_leaf=par,learning_rate=.025,loss='huber',random_state=991)
      if model=='ridge':m=Ridge(alpha=par)
      X=fs[key][tr];Z=fs[key][te]
      if model=='ridge':mu=X.mean(0);sd=np.maximum(X.std(0),1.);X=(X-mu)/sd;Z=(Z-mu)/sd
      m.fit(X,p.target_values(y[tr],kind),sample_weight=np.where(orders[tr]==1,sw,1.));q[k//4,te]=m.predict(Z)
     add((model,key,par,kind,sw),q)
     print('done',results[-1],flush=True)
ix=np.argsort([r[0] for r in results])[::-1]
print('BEST',flush=True)
for j in ix[:25]:print(results[j],flush=True)
np.savez('nonlinear_validation.npz',preds=np.array(preds),y=y,orders=orders)
json.dump(results,open('nonlinear_results.json','w'))
