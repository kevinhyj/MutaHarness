import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.special import expit,logit
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from proposal import *

# A priori approximate secondary-structure segments and hydrophobic face.
# These are hypotheses, not structure coordinates.
STRANDS=[(11,22,0),(25,37,1),(40,50,0),(80,89,0),(94,104,0),(117,124,1),(129,142,0),(145,153,0),(165,172,1),(181,190,0),(198,207,1),(219,229,1)]
def mfeat(ms,ref):
    out=[]
    for p,a,b in ms:
        h=HYD[a];dh=HYD[b]-h;v=(VOL[b]-VOL[a])/100;q=CH[b]-CH[a]
        strand=next(((l,u,f) for l,u,f in STRANDS if l<=p<=u),None)
        core=float(strand is not None and p%2==strand[2]);helix=float(58<=p<=75);critical=float(p in (65,66,67));outer=float(strand is not None and not core)
        x=[1,abs(dh)/5,abs(v),abs(q),max(0,-dh)/5,max(0,dh)/5,max(0,v),max(0,-v),float(b=='P'),float(a=='P'),float(b=='G'),float(a=='G'),float(a in 'FWY' and b not in 'FWY'),float(a in 'ILVMFWY' and b in 'DEKRQN'),float(a in 'DEKR' and b in 'DEKR' and q!=0),float(a in 'ST' and b in 'ST'),float(a in 'ILVM' and b in 'ILVM'),float(a in 'FWY' and b in 'FWY'),critical,helix,float(p<=9 or p>=230)]
        x += [core,outer,float(strand is not None),core*abs(dh)/5,core*max(0,-dh)/5,core*abs(v),core*abs(q),core*(b=='P'),outer*(b=='P'),helix*(b=='P'),float(strand is None)*(b=='P'),core*(a=='G'),float(p in (65,66,67,96,148,203,205,222)),float(b in 'DEKR'),float(a in 'DEKR'),float(b in 'QN'),float(b in 'FWY')]
        # Wildtype and target residue pooling gives generic tolerances.
        x += [float(a==aa) for aa in AA]+[float(b==aa) for aa in AA]
        out.append(x)
    return np.array(out) if out else np.zeros((1,77))

def extra_features(records,ref):
    rows=[]
    for r in records:
        ms=mutations(r,ref); m=mfeat(ms,ref)
        rows.append(np.r_[m.sum(0),m.max(0),len(ms)**2/10,np.log1p(len(ms))])
    return np.array(rows)

def tokens(records,ref):
    mts=[mutations(r,ref) for r in records]
    cats=sorted(set((p,a,b) for ms in mts for p,a,b in ms));pos=sorted(set(p for ms in mts for p,a,b in ms));trs=sorted(set((a,b) for ms in mts for p,a,b in ms))
    def design(rows):
        mt=[mutations(r,ref) for r in rows]
        e=np.array([[float(t in m) for t in cats] for m in mt]);p=np.array([[float(any(t[0]==pp for t in m)) for pp in pos] for m in mt]);t=np.array([[sum((a,b)==tr for pp,a,b in m) for tr in trs] for m in mt])
        return e,p,t
    return design

c=json.load(open('case.json'));r=c['support'];y=np.array([v['fitness'] for v in r]);ref=c['reference_sequence'];base=features(r,ref);extra=extra_features(r,ref)
print('features',base.shape,extra.shape,flush=True)
# held-out feedback with original singles retained; also mixed subsets.
folds=[]
for seed in [41,1729,2026]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,len(y))):
  folds.append((np.r_[np.arange(32),tr+32],te+32))
for tr,te in KFold(4,shuffle=True,random_state=170).split(y):folds.append((tr,te))
np.savez('folds.npz',dummy=np.array([1]))
results={}
def run(name,fn):
 st=time.time();o=[];cors=[]
 for tr,te in folds:
  try:p=fn(tr,te)
  except Exception as e:print(name,e,flush=True);return
  o.append((te,p));cors.append(spearmanr(y[te],p).statistic)
 results[name]=o
 print(name,round(np.nanmean(cors[:12]),4),round(np.nanmean(cors[12:]),4),round(np.nanstd(cors[:12]),3),'time',round(time.time()-st,2),flush=True)

def tree(X,kind='et',leaf=3,depth=None,mf=.8):
 def fn(tr,te):
  if kind=='et': model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=leaf,max_features=mf,max_depth=depth,random_state=42,n_jobs=1)
  elif kind=='rf':model=RandomForestRegressor(n_estimators=250,min_samples_leaf=leaf,max_features=mf,max_depth=depth,random_state=42,n_jobs=1)
  elif kind=='gb':model=GradientBoostingRegressor(n_estimators=120,max_depth=depth or 2,min_samples_leaf=leaf,learning_rate=.035,loss='squared_error',random_state=42)
  model.fit(X[tr],y[tr]);return model.predict(X[te])
 return fn

for X,n in [(base,'base'),(extra,'chem'),(np.c_[base,extra],'both')]:
 for leaf in [1,3,5]:run(n+'_et'+str(leaf),tree(X,leaf=leaf))
 for dep in [1,2]:run(n+'_gb'+str(dep),tree(X,kind='gb',leaf=4,depth=dep))
 for alpha in [1,10,100]:
  def fn(tr,te,X=X,alpha=alpha):
   sc=StandardScaler().fit(X[tr]); model=Ridge(alpha=alpha).fit(sc.transform(X[tr]),y[tr]);return model.predict(sc.transform(X[te]))
  run(n+'_ridge'+str(alpha),fn)

# Sparse additive regularization shares effects by site and substitution chemistry.
for chemweight in [0,1,3]:
 for alpha in [1,10,100]:
  def fn(tr,te,cw=chemweight,alpha=alpha):
   ds=tokens([r[i] for i in tr],ref);e,p,t=ds(r);sc=StandardScaler().fit(extra[tr]);X=np.c_[e*2,p,t,sc.transform(extra)*cw]
   model=Ridge(alpha=alpha).fit(X[tr],y[tr]);return model.predict(X[te])
  run('sparse'+str(chemweight)+'_'+str(alpha),fn)

np.savez('validation_predictions.npz',**{n:np.concatenate([p for te,p in vals]) for n,vals in results.items()},target=np.concatenate([y[te] for tr,te in folds]))
