import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge
import proposal as p
c=json.load(open('case.json'));ss=c['support'];y=np.array([s['fitness'] for s in ss]);orders=np.array([s['mutation_order'] for s in ss]);fs=p.features(ss,c['reference_sequence']);fs['mix']=np.concatenate([fs[k] for k in ['comp','regional','interactions']],axis=1)
si=np.where(orders==1)[0];mi=np.where(orders>1)[0]
configs=[('extra','regional',2,'rank',2),('extra','regional',2,'raw',.5),('extra','regional',4,'rank',2),('extra','regional',4,'raw',.5),('boost','mix',10,'rank',.5),('boost','mix',6,'rank',1),('boost','mix',10,'raw',1),('ridge','regional',1,'rank',.5),('ridge','regional',1,'raw',.5),('extra','mix',6,'rank',3)]
folds=[]
for seed in [887,2201,7453]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(mi):folds.append((np.r_[si,mi[tr]],mi[te]))
qs=[]
for conf in configs:
 model,key,par,kind,sw=conf;q=np.zeros((3,len(ss)))
 for k,(tr,te) in enumerate(folds):
  X=fs[key][tr];Z=fs[key]
  if model=='extra':m=ExtraTreesRegressor(n_estimators=320,min_samples_leaf=par,max_features=.8,random_state=991,n_jobs=1)
  if model=='boost':m=GradientBoostingRegressor(n_estimators=150,max_depth=2,min_samples_leaf=par,learning_rate=.025,loss='huber',random_state=991)
  if model=='ridge':
   m=Ridge(alpha=par);mu=X.mean(0);sd=np.maximum(X.std(0),1.);X=(X-mu)/sd;Z=(Z-mu)/sd
  m.fit(X,p.target_values(y[tr],kind),sample_weight=np.where(orders[tr]==1,sw,1.))
  z=m.predict(Z);q[k//4,te]=((rankdata(z)-.5)/len(z))[te]
 qs.append(q)
 print(conf,[round(spearmanr(y[mi],z[mi]).statistic,3) for z in q],flush=True)
qs=np.array(qs)
for ind in [[0,1],[0,1,2,3],[0,1,2,3,4],[0,1,2,3,4,7],[0,1,2,3,4,8],[0,1,4],[0,2,4],[0,1,2,3,4,9],[0,1,2,3,4,5,6]]:
 q=qs[ind].mean(0)
 print('ensemble',ind,'rho',[round(spearmanr(y[mi],z[mi]).statistic,3) for z in q], 'low',np.mean([spearmanr(y[mi[orders[mi]<10]],z[mi[orders[mi]<10]]).statistic for z in q]),'high',np.mean([spearmanr(y[mi[orders[mi]>20]],z[mi[orders[mi]>20]]).statistic for z in q]),flush=True)
np.savez('fresh_validation.npz',preds=qs,y=y,orders=orders)
