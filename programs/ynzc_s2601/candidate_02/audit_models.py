import json
import numpy as np
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import StratifiedKFold
import explore as e

# Rounded, diverse configurations rather than one highly tuned optimum.
configs=[
({'global':.012,'site':3.2,'local':.08,'pattern':.16,'interaction':.007,'smooth':4.,'context':.2},.32,'rank'),
({'global':.012,'site':3.2,'local':.08,'pattern':.16,'interaction':.007,'smooth':4.,'context':.01},.13,'rank'),
({'site':3.2,'local':.08,'pattern':.16,'interaction':.007,'smooth':.32,'context':.01},.13,'rank'),
({'global':.043,'site':2.,'local':.1,'pattern':3.5,'interaction':.001,'smooth':2.,'pair_interaction':.004},.09,'rank'),
({'global':.043,'site':2.,'local':.1,'pattern':3.5,'interaction':.001,'smooth_short':2.,'context':.04,'curved':.001,'pair_interaction':.008,'pair_local':.005},.22,'rank'),
({'global':.012,'site':3.2,'local':.08,'pattern':.16,'interaction':.007,'smooth':4.,'context':.2},.32,'raw'),
({'global':.012,'site':3.2,'local':.08,'pattern':.16,'interaction':.007,'smooth':4.,'context':.2},.32,'normal')]
ks=[sum(w.get(key,0)*v for key,v in e.parts.items()) for w,a,t in configs]
mixes={'rank5':[.2,.2,.2,.2,.2,0,0], 'rank3':[1/3,1/3,1/3,0,0,0,0],
       'mixed':[.18,.18,.18,.18,.18,.05,.05]}

def metrics(y,pred,mask=None):
    if mask is None:mask=e.sing
    return [float(spearmanr(y,pred).statistic),float(spearmanr(y[~mask],pred[~mask]).statistic),float(spearmanr(y[mask],pred[mask]).statistic)]

records=[]
for seed in [101,137,191,229,271,307]:
    preds=np.zeros((len(configs),e.n))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(e.y,e.order):
        for j,(w,a,t) in enumerate(configs):preds[j,te]=e.p.predict_kernel(ks[j],e.y,tr,te,a,t)
    pp=np.array([rankdata(v) for v in preds])
    records.append({'seed':seed,'individual':[metrics(e.y,v) for v in preds],
                    'ensemble':{key:metrics(e.y,np.array(w)@pp) for key,w in mixes.items()}})
print('Additional splits, individual',np.mean([r['individual'] for r in records],0))
for key in mixes:print('Additional splits, ensemble',key,np.mean([r['ensemble'][key] for r in records],0))

feedback=np.arange(32,e.n);original=np.arange(32);prospective=[]
for seed in [101,137,191,229,271,307,349,397,419,461]:
    rng=np.random.default_rng(seed);perm=rng.permutation(feedback);tr=np.r_[original,perm[:32]];te=perm[32:]
    ranked=[]
    for j,(w,a,t) in enumerate(configs):
        ranked.append(rankdata(e.p.predict_kernel(ks[j],e.y,tr,te,a,t)))
    prospective.append({key:metrics(e.y[te],np.array(w)@np.array(ranked),e.sing[te]) for key,w in mixes.items()})
print('Original32 plus feedback32 predicts remaining feedback32')
for key in mixes:print(key,np.mean([r[key] for r in prospective],0))

ranked=[]
for j,(w,a,t) in enumerate(configs):ranked.append(rankdata(e.p.predict_kernel(ks[j],e.y,original,feedback,a,t)))
print('Original32 alone predicts feedback64',metrics(e.y[feedback],np.array(mixes['rank5'])@np.array(ranked),e.sing[feedback]))
json.dump({'configs':configs,'additional_splits':records,'prospective':prospective},open('model_audit.json','w'),indent=2)
