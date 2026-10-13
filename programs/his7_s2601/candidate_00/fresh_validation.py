import json,numpy as np
from scipy.stats import rankdata,spearmanr
from sklearn.model_selection import KFold
from proposal import _components,mutations
c=json.load(open('case.json'));s=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in s]);mid=np.array([i for i,r in enumerate(s) if len(mutations(r,ref))>1]);sid=np.array([i for i,r in enumerate(s) if len(mutations(r,ref))<=1])
report={}
for folds,seeds in [(4,[109,157,211,263,307]),(2,[109,157,211,263,307])]:
 pp=[]
 for seed in seeds:
  pred=np.zeros((6,len(s)))
  for tr,te in KFold(folds,shuffle=True,random_state=seed).split(mid):
   train=np.r_[sid,mid[tr]];test=mid[te]
   pred[:,test]=_components([s[i] for i in train],[s[i] for i in test],ref,comparator=True)
  pp.append(np.array([rankdata(p[mid])/len(mid) for p in pred]))
 pp=np.array(pp)
 choices={'context':[1,0,0,0,0,0],'pair':[0,1,0,0,0,0],'ridge_ensemble':[.25,.25,.25,.25,0,0],'final':[.25,.2,.2,.2,.15,0],'chem_blend':[.225,.18,.18,.18,.135,.1],'tree':[0,0,0,0,1,0]}
 print('FOLDS',folds,flush=True);report[str(folds)]={}
 for name,weights in choices.items():
  blend=np.einsum("rmn,m->rn",pp,np.array(weights));rr=np.array([spearmanr(y[mid],p).statistic for p in blend]);report[str(folds)][name]={'mean':float(rr.mean()),'sd':float(rr.std()),'replicates':rr.tolist()};print(name,round(rr.mean(),4),round(rr.std(),4),flush=True)
json.dump(report,open('fresh_validation_results.json','w'),indent=2)
