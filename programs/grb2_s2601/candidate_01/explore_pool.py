import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import itertools,json,time,pickle
import numpy as np
from scipy.stats import spearmanr
from nonlinear import pooled_predict
c=json.load(open('case.json'));rs=c['support'];y=np.array([r['fitness'] for r in rs]);s=np.load('cv_splits.npz');splits=[(s[f'tr{i}'],s[f'te{i}']) for i in range(15)]
results=[];start=time.time()
for ws,wc,alpha,t,mix in itertools.product((4,16),(.0,.25,2),(0.2,1),(0.05,.25),(0.5,1)):
 ps=[];ss=[]
 for tr,te in splits:
  p=pooled_predict([rs[i] for i in tr],[rs[i] for i in te],c['reference_sequence'],ws=ws,wc=wc,alpha=alpha,t=t,mix=mix)
  ps.append(p);ss.append(spearmanr(p,y[te]).statistic)
 avg=.75*np.mean(ss[:10])+.25*np.mean(ss[10:]);cfg=(ws,wc,alpha,t,mix)
 results.append((avg,np.mean(ss[:10]),np.mean(ss[10:]),cfg,ps,ss))
 if len(results)%8==0: print('progress',len(results),'best',max(r[0] for r in results),flush=True)
results.sort(key=lambda t:-t[0])
for r in results[:20]:print(r[:4])
with open('pool_results.pkl','wb') as f:pickle.dump(results,f)
print('elapsed',time.time()-start)
