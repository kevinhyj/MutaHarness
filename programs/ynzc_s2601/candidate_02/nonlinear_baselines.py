import json
import numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
import explore as e

x=np.hstack([e.b[k] for k in ['global','site','local','context','identity','pattern']]);x=x[:,np.any(x,axis=0)]
models=[ExtraTreesRegressor(n_estimators=200,max_features=.8,min_samples_leaf=1,random_state=719,n_jobs=1),
        ExtraTreesRegressor(n_estimators=200,max_features=.8,min_samples_leaf=2,random_state=719,n_jobs=1),
        RandomForestRegressor(n_estimators=200,max_features=.8,min_samples_leaf=2,random_state=719,n_jobs=1),
        GradientBoostingRegressor(n_estimators=120,max_depth=2,learning_rate=.035,loss='huber',random_state=719)]
res=[]
for model in models:
    mm=[]
    for j in range(3):
        pred=np.zeros(e.n)
        for tr,te in e.folds[j*5:(j+1)*5]:model.fit(x[tr],e.y[tr]);pred[te]=model.predict(x[te])
        mm.append([spearmanr(e.y,pred).statistic,spearmanr(e.y[~e.sing],pred[~e.sing]).statistic,spearmanr(e.y[e.sing],pred[e.sing]).statistic])
    result={'model':str(model),'metrics':np.mean(mm,0).tolist()};res.append(result);print(result,flush=True)
json.dump(res,open('nonlinear_baselines.json','w'),indent=2)
