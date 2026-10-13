import json
import numpy as np
import explore as e
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr,rankdata

rng=np.random.default_rng(8292);base=json.load(open('model_audit.json'))['configs'][:5];results=[]
for i in range(250):
    w,a,t=base[i%5];w=w.copy()
    w['severity']=10**rng.uniform(-1,1)
    w['class']=10**rng.uniform(-2,1) if i%3 else 0
    w['site']*=10**rng.uniform(-1.7,.1)
    score,avg,pred=e.evaluate(w,a,t)
    results.append({'score':float(score),'metrics':avg.tolist(),'weights':w,'alpha':a,'transform':t})
results.sort(key=lambda r:r['score'],reverse=True)
json.dump(results,open('validation_results4.json','w'),indent=2)
print('Best',results[:3])
# Do not optimize on these confirmation splits.
for r in results[:5]:
    k=sum(r['weights'].get(key,0)*v for key,v in e.parts.items());mm=[]
    for seed in [101,137,191,229,271,307]:
        pred=np.zeros(e.n)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(e.y,e.order):
            pred[te]=e.p.predict_kernel(k,e.y,tr,te,r['alpha'],r['transform'])
        mm.append([spearmanr(e.y,pred).statistic,spearmanr(e.y[~e.sing],pred[~e.sing]).statistic,spearmanr(e.y[e.sing],pred[e.sing]).statistic])
    print('Confirmation',np.mean(mm,0))
