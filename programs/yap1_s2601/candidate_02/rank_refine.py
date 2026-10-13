import json,itertools
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from scientific_model import kernels,combine
from ranking_explore import rank_fit
c=json.load(open('case.json'));s=c['support'];n=len(s);y=np.array([r['fitness'] for r in s]);ks=kernels(s+c['query'],c['reference_sequence']);b={'exact':1,'pos':.3,'local':.2,'severity':1,'target':1}
folds=[]
for seed in [17,41,73]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,n)):folds.append((np.r_[np.arange(32),32+tr],32+te))
results=[]
for count,epi,reg,power in itertools.product([0,1,3],[0,.2],[.003,.01,.03,.1],[.5,1,2]):
 w=dict(b,count=count);K=combine(ks,w,epi)
 vals=[spearmanr(rank_fit(K,tr,te,y[tr],reg,power),y[te]).statistic for tr,te in folds]
 results.append((float(np.mean(vals)),w,reg,power,epi))
results.sort(key=lambda x:-x[0]);print('TOP',flush=True)
for r in results[:20]:print(r,flush=True)
# Compare a shortlist on distinct feedback and all-support split schedules.
for r in results[:6]:
 K=combine(ks,r[1],r[4]);out=[]
 for seed in [103,139,197,251]:
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,n)):
   tr=np.r_[np.arange(32),32+tr];te=32+te;out.append(spearmanr(rank_fit(K,tr,te,y[tr],r[2],r[3]),y[te]).statistic)
 vals=[]
 for seed in [113,181,277]:
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(s):vals.append(spearmanr(rank_fit(K,tr,te,y[tr],r[2],r[3]),y[te]).statistic)
 print('ASSESS',r, float(np.mean(out)),float(np.mean(vals)),flush=True)
json.dump(results,open('rank_refinement_results.json','w'))
# Use previously computed validation predictions to quantify simple rank-dominant blends.
d=json.load(open('assessment_results.json'))
for fraction in [.5,.75,1]:
 for tag in ['feedback','all']:
  vals=[]
  for f in d['folds']:
   if f['tag']!=tag:continue
   p=np.array(f['predictions']);mix=fraction*p[6]+(1-fraction)*(.25*p[0]+.25*p[2]+.25*p[3]+.25*p[4]);vals.append(spearmanr(mix,y[f['test']]).statistic)
  print('BLEND',fraction,tag,float(np.mean(vals)),flush=True)
