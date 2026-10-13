import json,time
import numpy as np
from explore_v2 import evaluate,splits
base=json.load(open('cv_results_v2.json'))[:20]
rng=np.random.default_rng(84943); results=[]
for r in base:
 for wenv in [0,.003,.01,.03,.1,.3,1,3]:
  w=r['weights'].copy(); w['environment']=wenv
  s,v,_=evaluate(w,r['target']); results.append(dict(score=s,parts=v.tolist(),weights=w,target=r['target']))
results.sort(key=lambda r:-r['score'])
json.dump(results,open('cv_results_v3.json','w'),indent=2)
for r in results[:12]: print(r)
