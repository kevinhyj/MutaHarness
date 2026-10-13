import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.stats import spearmanr,rankdata,norm
from scipy.special import logit,expit
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);ref=c['reference_sequence']
x,v=proposal.features(s,ref)
np.savez('features_explore.npz',x=x,y=y)
feedback=np.array([i for i,r in enumerate(s) if r['id'].startswith('M')]);original=np.array([i for i in range(len(s)) if i not in feedback])
splits=[]
for rep in range(3):
 for tr,te in KFold(4,shuffle=True,random_state=150+rep).split(feedback):
  splits.append((np.concatenate([original,feedback[tr]]),feedback[te]))
# Models fixed before examining validation outcomes.
preds={};results=[]
targets=['raw','rank','latent','binary']
for kind in ['chem','add','pos','pair','all','poly']:
 if kind=='chem': f=x[:,:79]
 elif kind=='add': f=x[:,:79+len(v)]
 elif kind=='pos': f=np.concatenate([x[:,:79],x[:,79+len(v):79+len(v)+len(ref)]],1)
 elif kind=='pair': f=np.concatenate([x[:,:79],x[:,-400:]],1)
 else:f=x
 for target in targets:
  for alpha in [1,10,50,200]:
   name=f'{kind}_{target}_{alpha}';p=np.full((3,len(s)),np.nan)
   for j,(tr,te) in enumerate(splits):
    mean=f[tr].mean(0);std=np.maximum(f[tr].std(0),.5);xt=(f[tr]-mean)/std;zt=(f[te]-mean)/std
    yy=y[tr]
    if target=='rank':yy=norm.ppf((rankdata(yy)-.5)/len(yy))
    if target=='latent':yy=logit(np.clip((yy-1.3)/2.6,.015,.985))
    if target=='binary':yy=(yy>2.8).astype(float)
    if kind=='poly':
     k=xt@xt.T;kt=zt@xt.T
     k=k+.015*k*k;kt=kt+.015*kt*kt
     ym=yy.mean();pr=kt@np.linalg.solve(k+alpha*np.eye(len(tr)),yy-ym)+ym
    else:pr=Ridge(alpha=alpha).fit(xt,yy).predict(zt)
    p[j//4,te]=pr
   preds[name]=p
   rhos=[spearmanr(y[feedback],pp[feedback]).statistic for pp in p]
   results.append((np.mean(rhos),np.std(rhos),name))
for kind in ['ET','RF','HGB']:
 for leaf in [2,5,10]:
  for feat in ['chem','all']:
   f=x[:,:79] if feat=='chem' else x
   name=f'{kind}_{feat}_{leaf}';p=np.full((3,len(s)),np.nan)
   for j,(tr,te) in enumerate(splits):
    if kind=='ET': model=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=leaf,max_features=.8,random_state=14,n_jobs=1)
    elif kind=='RF':model=RandomForestRegressor(n_estimators=180,min_samples_leaf=leaf,max_features=.8,random_state=14,n_jobs=1)
    else:model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=leaf,l2_regularization=10,random_state=14)
    p[j//4,te]=model.fit(f[tr],y[tr]).predict(f[te])
   preds[name]=p
   results.append((np.mean([spearmanr(y[feedback],pp[feedback]).statistic for pp in p]),np.std([spearmanr(y[feedback],pp[feedback]).statistic for pp in p]),name))
print('count',spearmanr(y[feedback],-x[feedback,0]).statistic)
for a in sorted(results,reverse=True)[:35]:print(a,flush=True)
np.savez('cv_predictions.npz',**preds)
json.dump(sorted(results,reverse=True),open('cv_results.json','w'),indent=2)
