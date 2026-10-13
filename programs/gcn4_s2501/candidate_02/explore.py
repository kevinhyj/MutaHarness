import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
import proposal as p
c=json.load(open('case.json'));ss=c['support'];y=np.array([s['fitness'] for s in ss]);orders=np.array([s['mutation_order'] for s in ss]);fs=p.features(ss,c['reference_sequence'])
singles=np.where(orders==1)[0]; multi=np.where(orders>1)[0]
folds=[]
for seed in [41,113,227,501]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(multi):
  folds.append((np.r_[singles,multi[tr]],multi[te]))
configs=[]
weights=[(1,0,0,0,0),(1,1,0,0,0),(1,1,.2,0,0),(1,1,.2,.2,0),(1,1,.2,.2,.5),(1,1,0,0,.5),(0,1,0,0,0),(0,0,1,1,0),(0,0,1,0,0),(1,0,0,0,1),(1,.3,.1,0,.3)]
for w in weights:
 for alpha in [.03,.1,.3,1,3,10]:
  for kind in ['rank','raw']:
   configs.append((w,alpha,kind,.5,0))
for w in [(1,0,0,0,0),(1,1,0,0,0),(1,1,.2,0,.2),(1,0,0,0,1)]:
 for alpha in [.03,.1,.3,1]:
  for rbf in [.2,.7,2]:configs.append((w,alpha,'rank',.5,rbf))
results=[];preds=[]
for config in configs:
 w,alpha,kind,sw,rbf=config
 oof=np.zeros((4,len(ss)))
 for k,(tr,te) in enumerate(folds):
  oof[k//4,te]=p.kernel_predict(fs,tr,te,y[tr],alpha,w,kind,sw,orders,rbf)
 rho=[spearmanr(y[multi],q[multi]).statistic for q in oof]
 low=multi[orders[multi]<10];high=multi[orders[multi]>20]
 lo=np.mean([spearmanr(y[low],q[low]).statistic for q in oof]);hi=np.mean([spearmanr(y[high],q[high]).statistic for q in oof])
 results.append((np.mean(rho),np.std(rho),lo,hi,config));preds.append(oof)
ix=np.argsort([r[0] for r in results])[::-1]
for j in ix[:25]:print(results[j],flush=True)
np.savez('kernel_validation.npz',preds=np.array(preds),y=y,orders=orders)
json.dump(results,open('kernel_results.json','w'))
# Tree models on low-dimensional sequence summaries, with regularized leaves.
for keys in [('comp',),('comp','regional'),('comp','regional','interactions'),('comp','regional','position','interactions')]:
 X=np.concatenate([fs[k] for k in keys],axis=1)
 for model in ['extra','forest','boost']:
  for leaf in [3,6,10]:
   oof=np.zeros((4,len(ss)))
   for k,(tr,te) in enumerate(folds):
    if model=='extra':m=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=leaf,max_features=.8,random_state=991,n_jobs=1)
    if model=='forest':m=RandomForestRegressor(n_estimators=250,min_samples_leaf=leaf,max_features=.8,random_state=991,n_jobs=1)
    if model=='boost':m=GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=leaf,learning_rate=.03,loss='huber',random_state=991)
    sw=np.where(orders[tr]==1,.5,1.)
    m.fit(X[tr],p.target_values(y[tr],'rank'),sample_weight=sw);oof[k//4,te]=m.predict(X[te])
   rho=[spearmanr(y[multi],q[multi]).statistic for q in oof]
   lo=np.mean([spearmanr(y[low],q[low]).statistic for q in oof]);hi=np.mean([spearmanr(y[high],q[high]).statistic for q in oof])
   print('tree',round(np.mean(rho),4),round(lo,4),round(hi,4),keys,model,leaf,flush=True)
