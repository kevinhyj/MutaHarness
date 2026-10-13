import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.stats import spearmanr,rankdata
from scipy.special import logit
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel
from proposal import encode,mutations
c=json.load(open('case.json')); s=c['support']; ref=c['reference_sequence']
F,v,G,N=encode(s,ref); d=len(v); X=F[:,:d]; Y=np.array([r['fitness'] for r in s]); mid=np.where(N>1)[0]; sid=np.where(N==1)[0]
# Feature maps: main substitutions, positions, block context, all-pair conjunctions.
positions=sorted(set(p for p,a in v)); P=np.column_stack([X[:,[j for j,(pp,a) in enumerate(v) if pp==p]].sum(axis=1) for p in positions])
M=(N>1).astype(float)[:,None]; pairs=[(i,j) for i in range(d) for j in range(i) if v[i][0]!=v[j][0] and np.sum(X[:,i]*X[:,j])>=2]
Z=np.column_stack([X[:,i]*X[:,j] for i,j in pairs])
features={'main':X,'context':F,'mutpos':np.column_stack([X,P,G*M,N,N*N/10]),'pair':np.column_stack([F,Z*.5]),'trees':np.column_stack([X,P,G,N])}
transforms={'raw':lambda y:y,'rank':lambda y:rankdata(y)/len(y),'log':lambda y:np.log(y+.1),'logit':lambda y:logit(np.clip(y/1.2,.02,.98))}
splits=[]
for seed in [17,43,81]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(mid): splits.append((np.r_[sid,mid[tr]],mid[te]))
configs=[]
for f in ['main','context','mutpos','pair']:
 for a in [.1,1,5,20]:
  for t in ['raw','rank','logit']:configs.append(('ridge',f,a,t))
for f in ['main','context']:
 for a in [.03,.3,3]:
  for gamma in [.05,.15,.4]:
   for t in ['raw','rank']:configs.append(('rbf',f,(a,gamma),t))
for a in [.1,1,5]:
 for degree in [2,3]:configs.append(('poly','context',(a,degree),'raw'))
for kind in ['extra','forest']:
 for leaf in [1,2,4]:
  for maxf in [.5,1.]:
   for t in ['raw','rank']:configs.append((kind,'trees',(leaf,maxf),t))
for depth in [1,2,3]:
 for leaf in [2,5]:
  for t in ['raw','rank']:configs.append(('boost','trees',(depth,leaf),t))
results=[]; started=time.time()
for k,(kind,f,h,t) in enumerate(configs):
 XX=features[f]; preds=np.full((3,len(s)),np.nan)
 if kind=='rbf': K=rbf_kernel(XX,gamma=h[1])
 if kind=='poly': K=(1+XX@XX.T/6)**h[1]
 for split,(tr,te) in enumerate(splits):
  y=transforms[t](Y[tr])
  if kind=='ridge': model=Ridge(alpha=h); p=model.fit(XX[tr],y).predict(XX[te])
  elif kind in ['rbf','poly']:
   mu=y.mean(); p=mu+K[np.ix_(te,tr)]@np.linalg.solve(K[np.ix_(tr,tr)]+np.eye(len(tr))*h[0],y-mu)
  else:
   if kind=='extra':model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=h[0],max_features=h[1],random_state=123,n_jobs=1)
   elif kind=='forest':model=RandomForestRegressor(n_estimators=120,min_samples_leaf=h[0],max_features=h[1],random_state=123,n_jobs=1)
   else:model=GradientBoostingRegressor(n_estimators=140,learning_rate=.04,max_depth=h[0],min_samples_leaf=h[1],random_state=123,loss='squared_error')
   p=model.fit(XX[tr],y).predict(XX[te])
  preds[split//4,te]=p
 corr=np.array([spearmanr(Y[mid],p[mid]).statistic for p in preds]); results.append((corr.mean(),corr.std(),(kind,f,h,t),preds))
 if k%20==0:print(k,round(time.time()-started,1),flush=True)
results.sort(key=lambda r:r[0],reverse=True)
for mean,sd,conf,p in results[:35]:print(round(mean,4),round(sd,4),conf)
np.savez('validation_predictions.npz',Y=Y,mid=mid,predictions=np.array([r[3] for r in results]))
json.dump([{'rho':r[0],'sd':r[1],'config':r[2]} for r in results],open('validation_results.json','w'),indent=2)
