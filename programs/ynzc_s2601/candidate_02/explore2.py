import json,time
import numpy as np
from scipy.stats import spearmanr,rankdata
import explore as e

rng=np.random.default_rng(2917);base=json.load(open('validation_results.json'))[:20]
results=[];t=time.time()
for i in range(420):
    old=base[i%len(base)];w=old['weights'].copy();a=old['alpha'];trans=old['transform']
    if i>=20:
        w['smooth']=10**rng.uniform(-1.4,1.0)
        w['context']=10**rng.uniform(-3,-.5) if i%3 else 0
        w['curved']=10**rng.uniform(-4,-1.5) if i%4 else 0
        if i%5==0:w['identity']=0
        if i%6==0:w['local']=0
        if i%7==0:w['interaction']=0
        if i%8==0:w['global']=0
        a*=10**rng.uniform(-.5,.5)
    score,avg,pred=e.evaluate(w,a,trans)
    results.append({'score':float(score),'metrics':avg.tolist(),'weights':w,'alpha':a,'transform':trans})
results.sort(key=lambda r:r['score'],reverse=True)
json.dump(results,open('validation_results2.json','w'),indent=2)
print('Elapsed',time.time()-t)
for r in results[:12]:print(r)
for top in [1,3,5,10,20]:
    pp=np.mean([np.array([rankdata(v) for v in e.evaluate(r['weights'],r['alpha'],r['transform'])[2]]) for r in results[:top]],0)
    print('Ensemble',top,np.mean([[spearmanr(e.y,v).statistic,spearmanr(e.y[~e.sing],v[~e.sing]).statistic,spearmanr(e.y[e.sing],v[e.sing]).statistic] for v in pp],0))
