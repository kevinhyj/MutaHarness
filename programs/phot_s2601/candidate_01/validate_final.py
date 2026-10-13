import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[v]='1'
import json,time
import numpy as np
from sklearn.model_selection import KFold
from proposal import _mutations,_matrix,_ensemble,_corr,propose
c=json.load(open('case.json'));ref=c['reference_sequence'];r=[a for a in c['support'] if len(_mutations(a,ref))>1];v=sorted({m for a in r for m in _mutations(a,ref)});x=_matrix(r,ref,v);y=np.array([a['fitness'] for a in r]);pos=np.array([int(m[1:-1]) for m in v]);p=np.zeros((2,len(r)));f=np.zeros_like(p);cs=[];fs=[];t=time.time()
for rep,seed in enumerate((809,1021)):
 for tr,va in KFold(4,shuffle=True,random_state=seed).split(x):
  p[rep,va]=_ensemble(x[tr],y[tr],x[va],pos)
  f[rep,va]=_ensemble(x[tr],y[tr],x[va],pos,validate=False)
  cs.append(_corr(y[va],p[rep,va]));fs.append(_corr(y[va],f[rep,va]))
print('Nested adaptive mean fold Spearman',np.mean(cs),'folds',cs,'pooled',_corr(y,p.mean(0)))
blend=.75*f+.25*p
print('Final blended ensemble mean fold Spearman',np.mean([_corr(y[va],blend[rep,va]) for rep,seed in enumerate((809,1021)) for tr,va in KFold(4,shuffle=True,random_state=seed).split(x)]),'pooled',_corr(y,blend.mean(0)))
print('Fixed ensemble mean fold Spearman',np.mean(fs),'folds',fs,'pooled',_corr(y,f.mean(0)))
t1=time.time();out=propose(c,json.load(open('assignment.json')));print('Fit seconds',time.time()-t1,'total validation seconds',time.time()-t,'scores',len(out['scores']));json.dump({'nested_fold_spearman':cs,'fixed_fold_spearman':fs,'nested_pooled':_corr(y,p.mean(0)),'fixed_pooled':_corr(y,f.mean(0))},open('final_validation.json','w'),indent=2)
