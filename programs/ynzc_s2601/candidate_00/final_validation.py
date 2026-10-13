import json
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata,spearmanr
from sklearn.model_selection import StratifiedKFold
from explore_v2 import B,grams,y,n,order

configs=[
 {'global':31.,'context':2.,'site':15.,'identity':.04,'pairprop':.74,'group':.7,'environment':1.},
 {'global':3.1,'context':.76,'site':1.,'identity':.007,'smooth1.5':.26,'epistasis1.5':.94,'environment':.3},
 {'global':19.,'context':1.5,'site':6.5,'identity':.88,'pair':.16,'group':1.4,'environment':1.},
 {'global':1.4,'context':1.1,'site':1.,'identity':.57,'pair':.021,'group':1.4,'environment':.1},
]
def predictions(w,folds):
 K=sum(w.get(k,0)*g for k,g in grams.items()); out=[]
 for rep in folds:
  p=np.full(n,np.nan)
  for tr,te in rep:
   kt=K[np.ix_(tr,tr)]; ks=K[np.ix_(te,tr)]; km=kt.mean(0); m=kt.mean()
   kt=kt-km[None,:]-km[:,None]+m; ks=ks-ks.mean(1)[:,None]-km[None,:]+m
   z=(rankdata(y[tr])-.5)/len(tr)
   p[te]=ks@solve(kt+np.eye(len(tr)),z-z.mean(),assume_a='pos')+z.mean()
  out.append(p)
 return np.array(out)
def metrics(P):
 scores=[]
 for p in P:
  valid=np.isfinite(p); scores.append([spearmanr(y[valid],p[valid]).statistic]+[spearmanr(y[valid&(order==o)],p[valid&(order==o)]).statistic for o in [2,1]])
 return np.mean(scores,0).tolist(),np.std(scores,0).tolist()

records={}
for mode in ['all','feedback']:
 idx=np.arange(n) if mode=='all' else np.arange(32,n)
 folds=[]
 for seed in [111,222,333,444,555,666,777,888]:
  rep=[]
  for tr,te in StratifiedKFold(4,shuffle=True,random_state=seed).split(idx,order[idx]):
   train=idx[tr] if mode=='all' else np.r_[np.arange(32),idx[tr]]
   rep.append((train,idx[te]))
  folds.append(rep)
 ps=np.array([predictions(w,folds) for w in configs])
 metrics_all=[metrics(p) for p in ps]
 metrics_all.append(metrics(ps.mean(0)))
 rankps=[]
 for p in ps:
  r=[]
  for row in p:
   valid=np.isfinite(row); z=row.copy(); z[valid]=rankdata(row[valid])/sum(valid); r.append(z)
  rankps.append(r)
 metrics_all.append(metrics(np.mean(rankps,0)))
 records[mode]=metrics_all
 print(mode,json.dumps(metrics_all),flush=True)
json.dump(records,open('validation_summary.json','w'),indent=2)
json.dump(configs,open('final_configs.json','w'),indent=2)
