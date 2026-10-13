import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np,time
from scipy.stats import spearmanr
from proposal import propose,CONFIGS
exec(open('explore2.py').read().split('\nconfigs=[]')[0])
models={}
for a,w in CONFIGS:
 models[a,w]=evaluate(K['exact']+K['pos']+w*K['region']+.1*K['count'],a,'rank')[0]
met=[]
for j,(tr,va) in enumerate(splits):
 p=np.mean([rankdata(v[j]) for v in models.values()],axis=0)
 met.append(spearmanr(y[va],p).statistic)
print('ensemble final search/confirmation/mixed',np.mean(met[:20]),np.mean(met[20:50]),np.mean(met[50:]))
for name,k,a,t in [('exact_raw',K['exact']+.1*K['count'],.3,'raw'),('position_rank',K['exact']+K['pos']+.1*K['count'],.1,'rank')]:
 p,m=evaluate(k,a,t);print('ablation',name,np.mean(m[:20]),np.mean(m[20:50]),np.mean(m[50:]))
a=json.load(open('assignment.json'));t=time.monotonic();p=propose(c,a);p2=propose(c,a)
assert p==p2
assert set(p['scores'])=={r['id'] for r in c['query']}
assert p['selected_ids']==[]
assert all(np.isfinite(v) and -1<=v<=1 for v in p['scores'].values())
# Verify no state survives a smaller fit and record reordering preserves scores.
small=dict(c,support=c['support'][:32]);propose(small,a);assert p==propose(c,a)
reverse=dict(c,support=c['support'][::-1],query=c['query'][::-1]);assert p==propose(reverse,a)
print('contracts, determinism, subset refit, reorder all passed; elapsed',time.monotonic()-t)
