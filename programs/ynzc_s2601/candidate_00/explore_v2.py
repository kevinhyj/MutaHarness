import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import StratifiedKFold
import proposal
c=json.load(open('case.json')); rows=c['support']; y=np.array([r['fitness'] for r in rows]); n=len(y)
B=proposal.features(rows,c['reference_sequence']); grams={k:v@v.T for k,v in B.items()}
order=np.array([r['mutation_order'] for r in rows]); label=order
splits=[]
for seed in [301,402,503,604,705,806,907,1008]:
    splits.append(list(StratifiedKFold(4,shuffle=True,random_state=seed).split(y,label)))

def evaluate(w,target='raw',folds=splits,returnpred=False):
    K=sum(w.get(k,0)*v for k,v in grams.items())
    scores=[]; preds=[]
    for rep in folds:
        p=np.zeros(n)
        for tr,te in rep:
            kt=K[np.ix_(tr,tr)]; ks=K[np.ix_(te,tr)]
            km=kt.mean(0); m=kt.mean(); kt=kt-km[None,:]-km[:,None]+m
            ks=ks-ks.mean(1)[:,None]-km[None,:]+m
            zz=y[tr]
            if target!='raw':
                zz=(rankdata(zz)-.5)/len(tr)
                if target=='normal': zz=norm.ppf(zz)
            p[te]=ks@solve(kt+np.eye(len(tr)),zz-zz.mean(),assume_a='pos')+zz.mean()
        s=[spearmanr(y,p).statistic,spearmanr(y[order==2],p[order==2]).statistic,spearmanr(y[order==1],p[order==1]).statistic]
        scores.append(s); preds.append(p)
    v=np.mean(scores,axis=0)
    return float(v@[.5,.35,.15]),v,np.array(preds)

if __name__=='__main__':
    rng=np.random.default_rng(3737); results=[]; t=time.time()
    old=json.load(open('cv_results_v1.json'))[:40]
    for r in old:
        s,v,_=evaluate(r['weights'],r['target']); r.update(score=s,parts=v.tolist()); results.append(r)
    print('OLD independent:')
    for r in sorted(results,key=lambda r:-r['score'])[:5]: print(r,flush=True)
    for i in range(1300):
        w={k:10**rng.uniform(-3,1.5) for k in ['global','context','site','identity','group','aa']}
        w['pair']=10**rng.uniform(-4,-.5) if rng.random()<.5 else 0
        w['pairprop']=10**rng.uniform(-4,-.5) if rng.random()<.3 else 0
        l=rng.choice([.7,1.5,3.]); w['smooth'+str(l)]=10**rng.uniform(-1,1.5)
        if rng.random()<.5: w['epistasis'+str(l)]=10**rng.uniform(-1,1.5)
        if rng.random()<.4: w['site']=0
        if rng.random()<.3: w['aa']=0
        target=rng.choice(['rank','rank','raw','normal'])
        s,v,_=evaluate(w,target)
        results.append({'score':s,'parts':v.tolist(),'weights':w,'target':str(target)})
    results.sort(key=lambda r:-r['score'])
    json.dump(results,open('cv_results_v2.json','w'),indent=2)
    print('NEW:')
    for r in results[:10]: print(r)
    print('time',time.time()-t)
