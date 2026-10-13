import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from proposal import features,mutations,_P,_AI
threadpool_limits(1)
c=json.load(open('case.json')); records=c['support']; y=np.array([r['fitness'] for r in records]);n=len(y)
p,e,ch,g=features(records,c['reference_sequence'])
ms=[mutations(r,c['reference_sequence']) for r in records]
components={'pos':p@p.T,'exact':e@e.T,'chem':ch@ch.T,'global':g@g.T}
# Kernel on substitute chemistry, with position and wild-type sharing separately.
for bw in [1.,3.,10.]:
 kr=np.exp(-(((_P[:,None]-_P[None])**2).sum(2))/bw)
 local=np.zeros((n,n)); shared=np.zeros((n,n))
 for i in range(n):
  for j in range(i+1):
   for pi,ai in ms[i]:
    for pj,aj in ms[j]:
     v=kr[ai,aj]
     if pi==pj:local[i,j]+=v
     if c['reference_sequence'][pi]==c['reference_sequence'][pj]:shared[i,j]+=v
   local[j,i]=local[i,j];shared[j,i]=shared[i,j]
 components['rbf'+str(bw)]=local;components['wt'+str(bw)]=shared
# weak local position smoothing as a testable sequence-only assumption
for bw in [2,5]:
 sm=np.exp(-((np.arange(p.shape[1])[:,None]-np.arange(p.shape[1])[None,:])/bw)**2)
 components['smooth'+str(bw)]=p@sm@p.T
singles=np.array([i for i,r in enumerate(records) if r['mutation_order']==1]); doubles=np.array([i for i,r in enumerate(records) if r['mutation_order']==2])
folds=[]
for seed in [2601,731]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(doubles):folds.append((np.r_[singles,doubles[tr]],doubles[te]))
# assay transformations, all monotonic. Floor and ceiling are hypotheses, not WT measurements.
targets={'raw':y, 'rank':norm.ppf((rankdata(y)-.5)/n)}
for lo in [-1.35,-1.5,-1.8,-2.5]:
 for hi in [.2,.5,1.]:
  z=np.clip((y-lo)/(hi-lo),.025,.975)
  targets[f'logit_{lo}_{hi}']=np.log(z/(1-z))
configs=[]
for pw in [.2,1,3]:
 for ew in [.1,1,3]:
  for ck in ['none','chem','rbf1.0','rbf3.0','rbf10.0']:
   for cw in ([0] if ck=='none' else [.3,1]):
    for gw in [0,.03,.15]:
     configs.append((pw,ew,ck,cw,gw,0,0))
# retain manageable broadly different additions
for wt in [.1,.4,1.]:
 for sm in [0,.2,1]:configs.append((1,.5,'rbf3.0',1,.05,wt,sm))
results=[];preds=[]
start=time.time()
for config in configs:
 pw,ew,ck,cw,gw,wt,sm=config
 K=pw*components['pos']+ew*components['exact']+gw*components['global']+1
 if cw:K+=cw*components[ck]
 if wt:K+=wt*components['wt3.0']
 if sm:K+=sm*components['smooth2']
 for alpha in [.1,.4,1.5,5,15]:
  fold_pred={name:np.zeros((2,n)) for name in targets}
  for f,(tr,te) in enumerate(folds):
   # intercept eliminated using kernel centering on the training subset
   kt=K[np.ix_(tr,tr)];kx=K[np.ix_(te,tr)]
   row=kt.mean(1);grand=row.mean();kt=kt-row[:,None]-row[None,:]+grand
   kx=kx-kx.mean(1)[:,None]-row[None,:]+grand
   vals,vec=eigh(kt); inv=vec/(np.maximum(vals,0)+alpha)
   Y=np.stack(list(targets.values()),1)[tr]; mean=Y.mean(0)
   ph=kx@(inv@(vec.T@(Y-mean)))+mean
   for col,name in enumerate(targets):fold_pred[name][f//4,te]=ph[:,col]
  for name,pr in fold_pred.items():
   cors=[spearmanr(y[doubles],pr[z,doubles]).statistic for z in range(2)]
   results.append((np.mean(cors),config,alpha,name,cors))
   preds.append(pr)
order=np.argsort([r[0] for r in results])[::-1]
print('seconds',time.time()-start)
for idx in order[:35]:print(results[idx])
np.savez('explore_results.npz',preds=np.array(preds),order=order,doubles=doubles,y=y)
json.dump(results,open('explore_results.json','w'))
