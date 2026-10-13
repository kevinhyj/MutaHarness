import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np,time
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import proposal
threadpool_limits(1)
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);o=np.array([r['mutation_order'] for r in s]);fb=np.where(o>1)[0];si=np.where(o==1)[0]
res={};start=time.time()
for splits,seeds in [(4,range(1001,1005)),(2,range(2001,2004))]:
 ps={k:[] for k in ['kernel','global_tree','local_tree','focused','focused_boost','ensemble','equal']}
 for seed in seeds:
  p={k:np.zeros(len(s)) for k in ps}
  for tr,te in StratifiedKFold(splits,shuffle=True,random_state=seed).split(fb,o[fb]>15):
   tr=np.r_[si,fb[tr]];te=fb[te];cs=dict(c,support=[s[i] for i in tr],query=[{k:v for k,v in s[i].items() if k!='fitness'} for i in te])
   parts=proposal.component_predictions(cs)
   for k,v in parts.items():p[k][te]=v
   p['ensemble'][te]=sum(proposal.ENSEMBLE_WEIGHTS[k]*v for k,v in parts.items())
   p['equal'][te]=sum(parts.values())/5
  for k,v in p.items():ps[k].append(v)
 for k,pp in ps.items():
  q={}
  for name,ix in [('feedback',fb),('low',fb[o[fb]<15]),('high',fb[o[fb]>15])]:
   vals=[spearmanr(y[ix],p[ix]).statistic for p in pp];q[name]={'mean':float(np.mean(vals)),'sd':float(np.std(vals))}
  res[str(splits)+'fold_'+k]=q
 print(splits,json.dumps({k:res[str(splits)+'fold_'+k] for k in ps},indent=2),flush=True)
json.dump(res,open('final_validation_results.json','w'),indent=2)
print('seconds',time.time()-start)
