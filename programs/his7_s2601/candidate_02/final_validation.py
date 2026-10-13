"""Checks the actual proposal on support-only holdouts; no hidden outcomes."""
import json,time
import numpy as np
from scipy.stats import spearmanr
from proposal import propose
c=json.load(open('case.json'));s=c['support'];singles=[r for r in s if r['mutation_order']==1];multi=[r for r in s if r['mutation_order']>1]
results=[];t=time.time()
for train_count,repetitions in [(48,12),(32,8)]:
 vals=[]
 for seed in range(repetitions):
  rng=np.random.default_rng(9201+seed);idx=rng.permutation(len(multi));train=singles+[multi[i] for i in idx[:train_count]];test=[multi[i] for i in idx[train_count:]]
  cc={**c,'support':train,'query':[{k:v for k,v in r.items()if k!='fitness'}for r in test]}
  out=propose(cc,{'batch_size':0});pred=[out['scores'][r['id']]for r in test];truth=[r['fitness']for r in test]
  vals.append(float(spearmanr(truth,pred).statistic))
 results.append({'train_feedback':train_count,'train_singles':32,'repetitions':repetitions,'mean_spearman':float(np.mean(vals)),'sd_across_splits':float(np.std(vals)),'split_spearman':vals})
# Constant outcomes and support-free cases are finite and handle every ID.
for supports in [[],s[:1],[dict(r,fitness=1.)for r in s[:5]]]:
 out=propose({**c,'support':supports},{})
 assert len(out['scores'])==len(c['query'])
 assert all(np.isfinite(v)and -1<=v<=1 for v in out['scores'].values())
 assert out['selected_ids']==[]
print(json.dumps({'validation':results,'elapsed_seconds':time.time()-t},indent=2))
