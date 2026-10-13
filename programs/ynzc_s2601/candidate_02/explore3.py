import json,time
import numpy as np
from scipy.stats import spearmanr,rankdata
import explore as e

rng=np.random.default_rng(9318);base=json.load(open('validation_results2.json'))[:12]
results=base.copy();t=time.time()
for i in range(440):
    old=base[i%len(base)];w=old['weights'].copy();a=old['alpha'];trans=old['transform']
    if i%4==0:trans='normal'
    if i%9==0:trans='raw'
    if i%11==0:trans='floor'
    if i%2:
        w['pair_local']=10**rng.uniform(-4,-.5)
        w['pair_interaction']=10**rng.uniform(-5,-2)
    if i%3:
        key='smooth_short' if i%3==1 else 'smooth_long';w[key]=w['smooth'];w['smooth']=0
    score,avg,pred=e.evaluate(w,a,trans)
    results.append({'score':float(score),'metrics':avg.tolist(),'weights':w,'alpha':a,'transform':trans})
results.sort(key=lambda r:r['score'],reverse=True)
json.dump(results,open('validation_results3.json','w'),indent=2)
print('Elapsed',time.time()-t)
for r in results[:15]:print(r)
print('Best per transform')
for trans in ['raw','rank','normal','floor']:
    print(trans,next(r for r in results if r['transform']==trans))
