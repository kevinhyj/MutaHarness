import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import StratifiedKFold
import proposal

c=json.load(open('case.json')); rows=c['support']; y=np.array([r['fitness'] for r in rows]); n=len(y)
B=proposal.features(rows,c['reference_sequence']); keys=list(B)
grams={k:v@v.T for k,v in B.items()}
order=np.array([r['mutation_order'] for r in rows]); label=np.array([r['mutation_order'] for r in rows])
splits=[]
for seed in [42,159,2026,71]:
    folds=list(StratifiedKFold(4,shuffle=True,random_state=seed).split(y,label))
    splits.append(folds)

def evaluate(w,target='raw',folds=splits):
    K=sum(w.get(k,0)*v for k,v in grams.items())
    z=y if target=='raw' else (rankdata(y)-.5)/n
    scores=[]; preds=[]
    for rep in folds:
        p=np.zeros(n)
        for tr,te in rep:
            kt=K[np.ix_(tr,tr)]; ks=K[np.ix_(te,tr)]
            km=kt.mean(0); m=kt.mean(); kt=kt-km[None,:]-km[:,None]+m
            ks=ks-ks.mean(1)[:,None]-km[None,:]+m
            zz=z[tr]
            if target=='rank': zz=(rankdata(y[tr])-.5)/len(tr)
            p[te]=ks@solve(kt+np.eye(len(tr)),zz-zz.mean(),assume_a='pos')+zz.mean()
        s=[spearmanr(y,p).statistic,spearmanr(y[order==2],p[order==2]).statistic,spearmanr(y[order==1],p[order==1]).statistic]
        scores.append(s); preds.append(p)
    v=np.mean(scores,axis=0)
    return float(v@[.5,.35,.15]),v,preds

rng=np.random.default_rng(20261013); results=[]; t=time.time()
for i in range(700):
    w={k:10**rng.uniform(-3,1.5) for k in keys}
    if rng.random()<.4: w['pairprop']=0
    if rng.random()<.3: w['pair']=0
    if rng.random()<.3: w['aa']=0
    target='rank' if i%4==0 else 'raw'
    s,v,_=evaluate(w,target)
    results.append({'score':s,'parts':v.tolist(),'weights':w,'target':target})
results.sort(key=lambda r:-r['score'])
json.dump(results,open('cv_results_v1.json','w'),indent=2)
for r in results[:15]: print(r)
print('time',time.time()-t)
