import json,time,numpy as np
from scipy.stats import spearmanr,rankdata
from model_dev import *
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);x,z,ms=featurize(s,c['reference_sequence']);multi=np.array([i for i,m in enumerate(ms) if len(m)>1]);single=np.setdiff1d(np.arange(len(y)),multi)
v=json.load(open('validation_ge.json'));splits=[(np.array(a),np.array(b)) for a,b in v['splits']]
for a in [.001,.01,.1,1]:
 for power in [0,.5,1]:
  co=[]
  for tr,te in splits:
   p=pair_pred(x[tr],x[te],z[tr],z[te],y[tr],a,power=power);co.append(spearmanr(p,y[te]).statistic)
  print('PAIR',a,power,np.mean(co),flush=True)
# Confirmation splits, both feedback-focused and all-support deployment.
results=[]
for setting in ['feedback','all','low_feedback']:
 splits2=[]
 for seed in [209,317,421]:
  inds=multi if setting!='all' else np.arange(len(y))
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(inds):
   tr=inds[tr];te=inds[te]
   if setting!='all':
    if setting=='low_feedback':tr=tr[:16]
    tr=np.r_[single,tr]
   splits2.append((tr,te))
 for structural in [True,False]:
  xx=x.copy()
  if not structural:
   # Delete beta/helix/proxy burial and all their interactions.
   remove=list(range(15,19))+list(range(20,32))
   cols=[3+i for i in remove]+[44+i for i in remove]+[87,88,89]
   xx[:,cols]=0
  scores={k:[] for k in ['linear','rbf1','rbf10','ge','ensemble','burden']}
  for tr,te in splits2:
   p1=kernel_pred(xx[tr],xx[te],z[tr],z[te],y[tr],.3,'raw',.3)
   p2=kernel_pred(xx[tr],xx[te],z[tr],z[te],y[tr],.3,'raw',.3,rbf=1)
   p3=kernel_pred(xx[tr],xx[te],z[tr],z[te],y[tr],.3,'raw',.3,rbf=10)
   p4=ge_pred(xx[tr],xx[te],z[tr],z[te],y[tr],.3,.3,4.)
   pe=sum(rankdata(p)/len(p) for p in [p1,p2,p3,p4])/4
   for k,p in zip(scores,[p1,p2,p3,p4,pe,-x[te,0]]):scores[k].append(float(spearmanr(p,y[te]).statistic))
  print('CONFIRM',setting,structural,{k:round(np.mean(v),4) for k,v in scores.items()},flush=True)
  results.append({'setting':setting,'structural':structural,'scores':scores})
json.dump(results,open('confirmation_results.json','w'),indent=2)
