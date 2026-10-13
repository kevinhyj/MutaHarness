import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json, numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, ARDRegression
from proposal import mutations
c=json.load(open('case.json')); ss=c['support']; sets=[mutations(r,c['reference_sequence']) for r in ss]
idx=np.array([i for i,s in enumerate(sets) if len(s)>1]); singles=np.array([i for i,s in enumerate(sets) if len(s)<=1])
alpha=sorted(set().union(*(sets[i] for i in idx)))
x=np.array([[float(m in s) for m in alpha] for s in sets]); y=np.array([r['fitness'] for r in ss])
print('multis',len(idx),'alleles',len(alpha),flush=True)
folds=[]
for seed in range(6):
 for tr,te in KFold(4,shuffle=True,random_state=seed+17).split(idx):folds.append((idx[tr],idx[te]))
results=[]; predictions={}
for transform in ['raw','sqrt','log','rank']:
 yt={'raw':y,'sqrt':np.sqrt(np.maximum(y,0)), 'log':np.log(np.maximum(y,.02)), 'rank':(rankdata(y)-.5)/len(y)}[transform]
 for pair in [0,.01,.03,.1,.3,1,3]:
  for reg in [.01,.03,.1,.3,1,3,10]:
   scores=[]; preds=np.zeros((6,len(idx)))
   for j,(tr,te) in enumerate(folds):
    xt=x-x[tr].mean(0)
    k=xt@xt.T; k=k+pair*k*k
    p=yt[tr].mean()+k[np.ix_(te,tr)]@np.linalg.solve(k[np.ix_(tr,tr)]+reg*np.eye(len(tr)),yt[tr]-yt[tr].mean())
    scores.append(spearmanr(y[te],p).statistic)
    preds[j//4,np.searchsorted(idx,te)]=p
   key=(transform,pair,reg); val=np.mean(scores); results.append((val,key,np.std(scores)));predictions[key]=preds
for r in sorted(results,reverse=True)[:30]:print(r,flush=True)
np.savez('exploration_predictions.npz',y=y[idx],**{str(k):v for k,v in predictions.items()})
json.dump([(a,list(b),d) for a,b,d in sorted(results,reverse=True)],open('exploration_results.json','w'),indent=2)
