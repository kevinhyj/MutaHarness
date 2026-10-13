import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge,ElasticNet
from proposal import _mutations,_features
c=json.load(open('case.json')); r=c['support'];ref=c['reference_sequence'];y=np.array([a['fitness'] for a in r]);muts=[_mutations(a,ref) for a in r]
# Sequence/design windows: not asserted structural units.
def block(p):
 if p<40:return 0
 if p<=65 or 76<=p<=85:return 1
 if p<96:return 2
 if p<=125:return 3
 if p<=144 or 157<=p<=163:return 4
 if p<=170:return 5
 if p<=180 or 195<=p<=200:return 6
 return 7
x,vocab=_features(r,ref)
positions=sorted(set(p for p,a in vocab));px=np.array([[float(any(p==pp for p,a in ms)) for pp in positions]for ms in muts])
bc=np.array([[sum(block(p)==b for p,a in ms) for b in range(8)]for ms in muts],float)
bm=(bc>=2).astype(float)
feat=np.column_stack([x,px*.5,bc*.25,bm])
mult=np.array([len(ms)>1 for ms in muts]);mi=np.where(mult)[0];si=np.where(~mult)[0]
print('n multi',len(mi),'block count',[(b,int(sum(bc[mi,b]>0)))for b in range(8)],flush=True)
configs=[]
for w in [1,3]:
 for deg in [1,2,3]:
  for alpha in [0.3,1,3,10]:
   for target in ['raw','rank']:
    configs.append(('k',w,deg,alpha,target))
for kind in ['et','rf','gb']:
 for depth in [2,4,None] if kind!='gb' else [1,2,3]:
  for leaf in [1,3]:configs.append((kind,depth,leaf))
allpred={str(cfg):np.zeros((3,len(mi))) for cfg in configs}
start=time.time()
for rep in range(3):
 for fold,(tr,te) in enumerate(KFold(4,shuffle=True,random_state=101+rep).split(mi)):
  train=np.r_[si,mi[tr]];test=mi[te];yt=y[train]
  rank=norm.ppf((rankdata(yt)-0.5)/len(yt))*.3
  for cfg in configs:
   key=str(cfg)
   if cfg[0]=='k':
    _,w,d,a,t=cfg
    # Polynomial mutation sharing, plus an explicit block-specific multi background.
    zz=x@x.T
    K=zz+2*(bm@bm.T)+1
    if d>=2:K=K+.25*zz**2
    if d>=3:K=K+.02*zz**3
    weights=np.where(mult[train],w,1)
    target=yt if t=='raw' else rank
    pred=target.mean()+K[np.ix_(test,train)]@solve(K[np.ix_(train,train)]+np.diag(a/weights),target-target.mean(),assume_a='pos')
   else:
    kind,depth,leaf=cfg
    if kind=='et':model=ExtraTreesRegressor(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,max_features=1.,random_state=21,n_jobs=1)
    if kind=='rf':model=RandomForestRegressor(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,max_features=.8,random_state=21,n_jobs=1)
    if kind=='gb':model=GradientBoostingRegressor(n_estimators=120,max_depth=depth,min_samples_leaf=leaf,learning_rate=.03,random_state=21,loss='huber')
    model.fit(feat[train],yt);pred=model.predict(feat[test])
   allpred[key][rep,te]=pred
 print('rep',rep,'elapsed',time.time()-start,flush=True)
res=[]
for k,p in allpred.items():
 sc=[spearmanr(y[mi],pp).statistic for pp in p]
 res.append((np.mean(sc),np.std(sc),k,sc))
for row in sorted(res,reverse=True): print(row)
np.savez('explore_predictions.npz',y=y[mi],**allpred)
