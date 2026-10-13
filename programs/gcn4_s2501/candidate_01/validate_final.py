import json,time
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import proposal
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);o=np.array([r['mutation_order'] for r in s]);feed=np.flatnonzero(o>1)
start=time.time();metrics=[]
for seed in [109,233,419]:
 pred=np.zeros(len(s))
 for _,te in StratifiedKFold(4,shuffle=True,random_state=seed).split(feed,o[feed]>15):
  test=feed[te];train=np.setdiff1d(np.arange(len(s)),test)
  with threadpool_limits(limits=1):
   pred[test]=proposal._predict([s[i] for i in train],[s[i] for i in test],c['reference_sequence'])
 vals=[spearmanr(y[ix],pred[ix]).statistic for ix in [feed,feed[o[feed]<15],feed[o[feed]>15]]]
 metrics.append(vals);print('fresh splits',seed,vals,flush=True)
# Contract and reproducibility under full and reduced public training subsets.
checks=[]
for n in [32,48,64,96]:
 cc=dict(c);cc['support']=s[:n]
 p=proposal.propose(cc,{})
 assert set(p['scores'])=={r['id'] for r in c['query']}
 assert p['selected_ids']==[]
 assert all(np.isfinite(v) and -1<=v<=1 for v in p['scores'].values())
 checks.append(n)
p1=proposal.propose(c,{});p2=proposal.propose(c,{})
assert p1==p2
r={'fresh_resampling_metrics':metrics,'mean_metrics':np.mean(metrics,axis=0).tolist(),'verified_support_sizes':checks,'seconds':time.time()-start}
json.dump(r,open('final_validation.json','w'),indent=2)
print(r,flush=True)
