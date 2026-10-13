import json,time
import numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from explore_v2 import y,n,B,order,splits,evaluate
X=np.column_stack([B[k] for k in ['global','context','site','identity','group','aa']])
X=X[:,np.any(X!=0,axis=0)]
def score(preds):
 s=[]
 for p in preds: s.append([spearmanr(y,p).statistic,spearmanr(y[order==2],p[order==2]).statistic,spearmanr(y[order==1],p[order==1]).statistic])
 v=np.mean(s,0); return float(v@[.5,.35,.15]),v
allp=[]; cfg=[]
for target in ['raw','rank']:
 for typ in ['extra','gbr']:
  for depth,leaf in [(2,2),(3,2),(None,2),(None,1)] if typ=='extra' else [(1,3),(2,3),(3,3)]:
   preds=[]
   for rep in splits[:4]:
    p=np.zeros(n)
    for tr,te in rep:
     z=y[tr] if target=='raw' else (rankdata(y[tr])-.5)/len(tr)
     if typ=='extra': model=ExtraTreesRegressor(n_estimators=180,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=42,n_jobs=1)
     else: model=GradientBoostingRegressor(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,learning_rate=.035,loss='huber',random_state=42)
     model.fit(X[tr],z); p[te]=model.predict(X[te])
    preds.append(p)
   preds=np.asarray(preds); s,v=score(preds); print(target,typ,depth,leaf,s,v,flush=True)
   allp.append(preds); cfg.append([target,typ,depth,leaf,s,v.tolist()])
np.savez('tree_predictions.npz',predictions=np.asarray(allp)); json.dump(cfg,open('tree_configs.json','w'),indent=2)
