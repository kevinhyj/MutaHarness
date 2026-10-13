import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor,GradientBoostingRegressor
from proposal import features
c=json.load(open('case.json'));r=c['support'];n=len(r);ref=c['reference_sequence'];y=np.array([a['fitness'] for a in r]);order=np.array([a['mutation_order'] for a in r]);f=features(r,ref);ph=f['phys'].reshape(n,len(ref),8);mut=f['mut'].reshape(n,len(ref),20)
# Positional descriptors plus aggregate descriptors. All positions remain represented.
X=np.c_[f['pos'],f['pair'],f['phys'],f['general'],f['cross']]
# Compress invariant features, not outcome based selection.
X=X[:,np.ptp(X,axis=0)>0]
Xcat=np.c_[X,f['mut'][:,np.ptp(f['mut'],axis=0)>0]]
# Repeated folds identical to first search.
splits=[]
for seed in [11,29,61]:
 for tr,te in StratifiedKFold(4,shuffle=True,random_state=seed).split(np.arange(n),order):splits.append((tr,te))
 fb=np.arange(32,n)
 for tr,te in StratifiedKFold(4,shuffle=True,random_state=seed).split(fb,order[fb]):splits.append((np.r_[np.arange(32),fb[tr]],fb[te]))
res=[];predlist=[]
for cat in [False,True]:
 xx=Xcat if cat else X
 for model in ['extra','random','boost']:
  for leaf in [1,2,4]:
   for mf in [.5,1.]:
    params=dict(cat=cat,model=model,leaf=leaf,mf=mf)
    pred=np.full((6,n),np.nan)
    for ix,(tr,te) in enumerate(splits):
     if model=='extra':m=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=mf,random_state=52,n_jobs=1)
     elif model=='random':m=RandomForestRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=mf,random_state=52,n_jobs=1)
     else:m=GradientBoostingRegressor(n_estimators=120,learning_rate=.04,max_depth=2 if leaf<4 else 1,min_samples_leaf=leaf,max_features=mf,random_state=52,loss='huber')
     m.fit(xx[tr],y[tr]);pred[ix//4,te]=m.predict(xx[te])
    overall=[];double=[]
    for pp in pred:
     ok=np.isfinite(pp);overall.append(spearmanr(y[ok],pp[ok]).statistic);ok=ok&(order==2);double.append(spearmanr(y[ok],pp[ok]).statistic)
    v=.75*np.mean(overall)+.25*np.mean(double)
    res.append(dict(**params,objective=v,overall=np.mean(overall),double=np.mean(double)));predlist.append(pred)
    print(res[-1],flush=True)
json.dump(res,open('tree_search.json','w'),indent=2);np.save('tree_oof.npy',np.array(predlist))
