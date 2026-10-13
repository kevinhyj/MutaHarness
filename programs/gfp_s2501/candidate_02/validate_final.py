import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.stats import rankdata,spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
import proposal,extra_features
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);ref=c['reference_sequence']

def predict(train,test):
    x,v=proposal.features(train,ref);z,_=proposal.features(test,ref,v)
    xe,ve=extra_features.enhanced(train,ref);ze,_=extra_features.enhanced(test,ref,ve)
    base=79+len(v)+len(ref)+400;xe=np.c_[xe[:,:base+22],xe[:,-200:]];ze=np.c_[ze[:,:base+22],ze[:,-200:]]
    yy=np.array([r['fitness'] for r in train]);out=[]
    for cls,leaf,xx,zz in [(ExtraTreesRegressor,10,x,z),(RandomForestRegressor,2,x[:,:79],z[:,:79]),(ExtraTreesRegressor,2,x,z),(ExtraTreesRegressor,2,xe,ze)]:
        out.append(cls(n_estimators=180,min_samples_leaf=leaf,max_features=.8,random_state=14,n_jobs=1).fit(xx,yy).predict(zz))
    mean=x.mean(0);scale=np.maximum(x.std(0),.5)
    out.append(Ridge(alpha=200).fit((x-mean)/scale,(yy>2.8).astype(float)).predict((z-mean)/scale))
    pr=np.mean([(rankdata(v)-.5)/len(v) for v in out],axis=0)
    singles={};ys=[]
    for r in train:
        ms=proposal.mutations(r,ref)
        if len(ms)==1:singles.setdefault(ms[0][0],[]).append(r['fitness']);ys.append(r['fitness'])
    anchor_base=np.quantile(ys,.75) if ys else np.quantile(yy,.8)
    anchor=np.array([sum(np.mean(singles[p])-anchor_base for p,a,b in proposal.mutations(r,ref) if p in singles) for r in test])
    return pr,anchor
for typ in ['mixed','feedback']:
    ids=np.arange(len(s)) if typ=='mixed' else np.array([i for i,r in enumerate(s) if r['id'].startswith('M')]);orig=np.array([i for i in range(len(s)) if i not in ids],dtype=int);rhos={w:[] for w in [0,.1,.2,.3,.5]}
    for rep in range(2):
      yy=[];pps={w:[] for w in rhos}
      for a,b in KFold(4,shuffle=True,random_state=912+rep).split(ids):
        tr=np.r_[orig,ids[a]];te=ids[b];p,anchor=predict([s[i] for i in tr],[s[i] for i in te]);yy.extend(y[te])
        for w in rhos:pps[w].extend(p+w*anchor)
      for w in rhos:rhos[w].append(spearmanr(yy,pps[w]).statistic)
    print(typ,{w:(np.mean(v),v) for w,v in rhos.items()},flush=True)
