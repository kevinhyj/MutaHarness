"""Local experiments; never imported by the submitted predictor."""
import json,time
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
import proposal as p

c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);n=len(s)
b=p.features(s,c['reference_sequence']);order=np.array([r['mutation_order'] for r in s]);sing=order==1
folds=[]
for seed in [17,29,43]:
    folds+=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(y,order))
parts={key:value@value.T for key,value in b.items()}

def evaluate(weights,alpha,transform='raw',details=False):
    k=sum(weights.get(key,0)*value for key,value in parts.items())
    preds=[];stats=[]
    for j in range(3):
        pred=np.zeros(n)
        for tr,te in folds[j*5:(j+1)*5]:pred[te]=p.predict_kernel(k,y,tr,te,alpha,transform)
        preds.append(pred);stats.append([spearmanr(y,pred).statistic,spearmanr(y[~sing],pred[~sing]).statistic,spearmanr(y[sing],pred[sing]).statistic])
    avg=np.mean(stats,0);score=avg@[.65,.25,.1]
    return score,avg,np.array(preds)

if __name__=='__main__':
    rng=np.random.default_rng(891);results=[];t=time.time()
    configs=[(p.DEFAULT,1.,'raw')]
    for i in range(240):
        w={'global':10**rng.uniform(-2.3,-.3),'site':10**rng.uniform(-.5,.7),'local':10**rng.uniform(-2,-.3),
           'identity':10**rng.uniform(-1,1),'pattern':10**rng.uniform(-1.5,.7),'interaction':10**rng.uniform(-4,-1.5)}
        if i%5==0:w['interaction']=0
        if i%7==0:w['pattern']=0
        configs.append((w,10**rng.uniform(-1.2,.7),'rank' if i%4==0 else 'raw'))
    for i,(w,a,trans) in enumerate(configs):
        score,avg,pred=evaluate(w,a,trans)
        results.append({'score':float(score),'metrics':avg.tolist(),'weights':w,'alpha':a,'transform':trans})
    results.sort(key=lambda x:x['score'],reverse=True)
    json.dump(results,open('validation_results.json','w'),indent=2)
    print('Elapsed',time.time()-t)
    for r in results[:15]:print(r)
    print('Baseline',evaluate(p.DEFAULT,1.)[:2])
