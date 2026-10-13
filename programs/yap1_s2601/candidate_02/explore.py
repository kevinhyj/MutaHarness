import json,time
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from scientific_model import kernels,target_transform,combine,predict_kernel
c=json.load(open('case.json'));n=len(c['support']);s=c['support'];y=np.array([r['fitness'] for r in s]);ks=kernels(s+c['query'],c['reference_sequence'])
# Keep all original single mutants, independently hold out feedback.
orig=np.array([i for i,r in enumerate(s) if r['id'].startswith('S')]);fb=np.array([i for i in range(n) if i not in orig]);folds=[]
for seed in [17,41,73]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(fb):folds.append((np.r_[orig,fb[tr]],fb[te]))
# All-support splits check robustness to public training subsets.
other=[(tr,te) for tr,te in KFold(4,shuffle=True,random_state=113).split(s)]
def evaluate(weights,alpha,trans,epi=0,full=False):
 K=combine(ks,weights,epi); out=[]
 for tr,te in folds+(other if full else []):
  p=predict_kernel(K,tr,te,target_transform(y[tr],trans),alpha)
  out.append(spearmanr(y[te],p).statistic)
 return float(np.mean(out[:12])),float(np.mean(out[12:])) if full else 0
configs=[]
# Coarse structured search; exact mutation effects + transfer via chemistry.
for kind in ['local','rbf','blosum','brbf']:
 for pv in [0,.3,1,3]:
  for lv in [.2,1,3]:
   for glob in [0,.2,1]:
    weights={'exact':1,'pos':pv,kind:lv,'severity':glob}
    for alpha in [.1,.5,2]:
     for trans in ['log','quarter','sqrt']:
      a,b=evaluate(weights,alpha,trans);configs.append((a,weights,alpha,trans,0))
configs.sort(key=lambda x:-x[0]);print('COARSE TOP',flush=True)
for x in configs[:20]:print(x,flush=True)
json.dump(configs,open('coarse_results.json','w'))
# Richer alternatives from robust coarse families.
base=[]
for kind in ['local','rbf','blosum','brbf']:
 base+=sorted([r for r in configs if kind in r[1]],reverse=True,key=lambda x:x[0])[:2]
rich=[]
for _,w,alpha,trans,_ in base:
 for name in ['target','sub','chem','regional','regtarget','wtchem','wtblosum']:
  for val in [.2,1,3]:
   ww=dict(w);ww[name]=val
   for ep in [0,.3,1]:
    score,b=evaluate(ww,alpha,trans,ep);rich.append((score,ww,alpha,trans,ep))
rich.sort(key=lambda x:-x[0]);print('RICH TOP',flush=True)
for x in rich[:20]:print(x,flush=True)
allres=configs+rich;allres.sort(key=lambda x:-x[0]);json.dump(allres,open('model_results.json','w'))
print('ROBUSTNESS',flush=True)
for x in allres[:25]:print(evaluate(x[1],x[2],x[3],x[4],True),x[1:],flush=True)
