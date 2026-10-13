import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json
import numpy as np
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
from explore import kernels,predict,region
from proposal import features,mutations,properties

c=json.load(open('case.json'));rec=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in rec]);n=len(y)
fb=np.array([i for i,r in enumerate(rec) if r['id'].startswith('M')]);ks=kernels(rec,ref)
ss=[]
for seed in [101,211,307,409]:
    for _,te in KFold(4,shuffle=True,random_state=seed).split(fb):
        test=fb[te];ss.append((np.setdiff1d(np.arange(n),test),test,'feedback'))
for seed in [113,223,317]:
    for tr,te in KFold(4,shuffle=True,random_state=seed).split(y):ss.append((tr,te,'all'))
search=json.load(open('kernel_search.json'))

def kval(cfg):
    K=np.ones((n,n))*cfg.get('bias',1)
    for name,w in cfg.items():
        if name in ks:K+=w*ks[name]
    if cfg.get('poly',0):K+=cfg['poly']*(ks['exact']+.3*ks['site']+.1*ks['localchem'])**2
    return K

def metric(ps):
    d={'feedback':[],'all':[]}
    for (_,te,k),p in zip(ss,ps):d[k].append(spearmanr(y[te],p).statistic)
    f=np.mean(d['feedback']);a=np.mean(d['all']);return [.7*f+.3*a,f,a]

allpred={}
for top in [1,3,5,10,20,40]:
    for transform in ['raw','rank','normal','asinh']:
        ps=[]
        for tr,te,_ in ss:
            yy=y.copy()
            if transform=='rank':yy[tr]=(rankdata(y[tr])-.5)/len(tr)
            if transform=='normal':yy[tr]=norm.ppf((rankdata(y[tr])-.5)/len(tr))
            if transform=='asinh':yy[tr]=np.arcsinh(y[tr]/.4)
            p=[]
            for _,cfg in search[:top]:p.append(predict(kval(cfg),yy,tr,te,cfg['noise']))
            ps.append(np.mean(p,axis=0))
        key=f'kernel_{top}_{transform}';allpred[key]=ps;print(key,metric(ps),flush=True)

# Nonlinear sequence-derived features, plus exact identities and smooth location.
E,S,C,_=features(rec,ref)
X=[]
for record in rec:
    mm=mutations(record,ref);cells=[]
    for a,p,b in mm:
        w=properties(a);m=properties(b);d=m-w
        cells.append(np.r_[p/188.,w,m,d,abs(d),float(p>166),float(p>=185)])
    arr=np.array(cells);z=np.r_[len(mm),arr.sum(0),arr.min(0),arr.max(0)]
    loc=[]
    for center in range(0,189,8):
        weights=np.exp(-((np.array([m[1] for m in mm])-center)/12)**2)
        loc.extend((arr[:,1:]*weights[:,None]).sum(0));loc.append(weights.sum())
    X.append(np.r_[z,loc])
X=np.column_stack([np.array(X),E,S])
for typ in ['et','rf','gb']:
 for leaf in [1,3,6]:
  ps=[]
  for tr,te,_ in ss:
    if typ=='et':model=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=leaf,max_features=.7,random_state=13,n_jobs=1)
    if typ=='rf':model=RandomForestRegressor(n_estimators=250,min_samples_leaf=leaf,max_features=.7,random_state=13,n_jobs=1)
    if typ=='gb':model=GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=leaf,learning_rate=.035,loss='huber',random_state=13)
    model.fit(X[tr],y[tr]);ps.append(model.predict(X[te]))
  key=f'{typ}_{leaf}';allpred[key]=ps;print(key,metric(ps),flush=True)
  for weight in [.2,.5]:
    comb=[(1-weight)*p+weight*q for p,q in zip(allpred['kernel_10_raw'],ps)]
    print('blend',key,weight,metric(comb),flush=True)
np.savez('validation_predictions.npz',**{k:np.concatenate(v) for k,v in allpred.items()})
json.dump({k:metric(v) for k,v in allpred.items()},open('validation_metrics.json','w'),indent=2)
