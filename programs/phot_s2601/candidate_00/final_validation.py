import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import ShuffleSplit,KFold
from proposal import propose
c=json.load(open('case.json'));ss=c['support'];mi=[i for i,r in enumerate(ss) if r['mutation_order']>1];si=[i for i,r in enumerate(ss) if r['mutation_order']<=1]
results={}
for size in [48,32,24]:
 vals=[]
 for tr,te in ShuffleSplit(20,train_size=size,random_state=1917).split(mi):
  train=[ss[mi[i]] for i in tr]+[ss[i] for i in si];test=[ss[mi[i]] for i in te]
  cc=dict(c,support=train,query=[{k:v for k,v in r.items() if k!='fitness'} for r in test]);p=propose(cc,{})['scores']
  vals.append(spearmanr([r['fitness'] for r in test],[p[r['id']] for r in test]).statistic)
 results[str(size)]={'mean':float(np.mean(vals)),'sd':float(np.std(vals))};print(size,results[str(size)],flush=True)
start=time.time();p=propose(c,{})
assert set(p['scores'])=={r['id'] for r in c['query']}
assert all(np.isfinite(v) and -1<=v<=1 for v in p['scores'].values())
assert p['selected_ids']==[]
assert p==propose(c,{})
print('Full fit',time.time()-start,'seconds; deterministic contract passed')
json.dump(results,open('final_validation_results.json','w'),indent=2)
