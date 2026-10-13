import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr,rankdata,norm
import proposal
c=json.load(open('case.json'));S=c['support'];y=np.array([r['fitness'] for r in S]);ref=c['reference_sequence'];ms=[m for r in S for m in proposal.mutations(r,ref)];pos=sorted(set(m[0] for m in ms));alleles=sorted(set(ms));B=proposal.design(S,ref,pos,alleles)
X=np.concatenate([B[0],B[2],B[3],B[4]],axis=1)
fs=[]
for seed in [11,29,47]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,96)):fs.append((np.r_[np.arange(32),32+tr],32+te))
for cls in [ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor]:
 for depth,leaf in [(2,3),(3,3),(None,2),(None,5)]:
  if cls is GradientBoostingRegressor and depth is None:continue
  sc=[]
  for tr,te in fs:
   model=cls(n_estimators=180,max_depth=depth,min_samples_leaf=leaf,random_state=42)
   model.fit(X[tr],norm.ppf((rankdata(y[tr])-.5)/len(tr)))
   sc.append(spearmanr(y[te],model.predict(X[te])).statistic)
  print(cls.__name__,depth,leaf,np.mean(sc),flush=True)
