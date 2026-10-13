import json
import numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold
from scientific_model import kernels,combine,predict_kernel,target_transform
from ranking_explore import rank_fit
c=json.load(open('case.json'));s=c['support'];n=len(s);q=len(c['query']);y=np.array([r['fitness'] for r in s]);ks=kernels(s+c['query'],c['reference_sequence'])
b={'exact':1,'pos':.3,'local':.2,'severity':1,'target':1}
ranks=json.load(open('ranking_results.json'))
print('BEST RANK',ranks[:3],flush=True)
models=[('base',b,.1,'sqrt',0),('add_count',dict(b,count=1),.1,'sqrt',0),('poly_count',dict(b,count=3),.03,'sqrt',.2),('poly_count_shrunk',dict(b,count=3),.1,'sqrt',.2),('blosum',{'exact':1,'pos':.3,'brbf':.2,'severity':1,'target':1},.1,'sqrt',0),('log',dict(b,count=1),.4,'log',0),('rank',ranks[0][1],ranks[0][2],'pair',ranks[0][3])]
folds=[]
for seed in [103,139,197,251]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,n)):folds.append(('feedback',np.r_[np.arange(32),32+tr],32+te))
for seed in [113,181,277]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(s):folds.append(('all',tr,te))
metrics={};allouts=[]
for tag,tr,te in folds:
 test=np.r_[te,np.arange(n,n+q)];pp=[]
 for name,w,alpha,trans,epi in models:
  K=combine(ks,w,epi if trans!='pair' else 0)
  if trans=='pair':p=rank_fit(K,tr,test,y[tr],alpha,epi)
  else:p=predict_kernel(K,tr,test,target_transform(y[tr],trans),alpha)
  pp.append((rankdata(p)-.5)/len(p));metrics.setdefault((tag,name),[]).append(spearmanr(p[:len(te)],y[te]).statistic)
 P=np.array(pp)
 for name,weights in [('sqrt_ensemble',[.25,.25,.25,.25,0,0,0]),('diverse',[.15,.25,.20,.15,.10,0,.15]),('additive',[.3,.4,0,0,.3,0,0]),('rank_blend',[.15,.2,.15,.15,.1,0,.25]),('log_blend',[.15,.25,.2,.15,.1,.15,0])]:
  p=np.array(weights)@P;metrics.setdefault((tag,name),[]).append(spearmanr(p[:len(te)],y[te]).statistic)
 allouts.append({'tag':tag,'train':tr.tolist(),'test':te.tolist(),'predictions':P[:,:len(te)].tolist()})
for key,v in metrics.items():print(key,round(float(np.mean(v)),4),'SD',round(float(np.std(v)),4),flush=True)
json.dump({'models':models,'metrics':{str(k):v for k,v in metrics.items()},'folds':allouts},open('assessment_results.json','w'))
