import json,time
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold,ShuffleSplit
from proposal import propose,predict_records,tokens
import proposal_baseline
c=json.load(open('case.json')); s=c['support']; multi=np.array([len(tokens(r,c['reference_sequence']))>1 for r in s]); inds=np.flatnonzero(multi); singles=np.flatnonzero(~multi)
results=[];t=time.time()
for seed in [307,331,353,379]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(inds):
  support=[s[i] for i in np.r_[singles,inds[tr]]]; query=[s[i] for i in inds[te]]; y=np.array([r['fitness'] for r in query]);p=predict_records(support,query,c['reference_sequence'])
  base=proposal_baseline.predict_records(support,query,c['reference_sequence'])
  results.append({'size':len(tr),'rho':float(spearmanr(y,p).statistic),'baseline_rho':float(spearmanr(y,base).statistic)})
for tr,te in ShuffleSplit(8,train_size=32,random_state=401).split(inds):
 support=[s[i] for i in np.r_[singles,inds[tr]]];query=[s[i] for i in inds[te]];y=np.array([r['fitness'] for r in query]);p=predict_records(support,query,c['reference_sequence']);base=proposal_baseline.predict_records(support,query,c['reference_sequence'])
 results.append({'size':len(tr),'rho':float(spearmanr(y,p).statistic),'baseline_rho':float(spearmanr(y,base).statistic)})
for size in [48,32]:
 rr=[r for r in results if r['size']==size];print(size,'rho',np.mean([r['rho'] for r in rr]),'sd',np.std([r['rho'] for r in rr]),'baseline',np.mean([r['baseline_rho'] for r in rr]))
start=time.time();a=propose(c,{});print('final runtime',time.time()-start,'n',len(a['scores']),'range',min(a['scores'].values()),max(a['scores'].values()))
b=propose(c,{});assert a==b
rng=np.random.default_rng(4);d=dict(c);d['support']=list(rng.permutation(c['support']));assert propose(d,{})==a
assert a['selected_ids']==[] and set(a['scores'])=={r['id'] for r in c['query']}
assert all(np.isfinite(v) and -1<=v<=1 for v in a['scores'].values())
print('contract and determinism checks passed; total time',time.time()-t)
json.dump(results,open('final_validation.json','w'),indent=2)
