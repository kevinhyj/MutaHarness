from model_helpers import *
# load ge_fit without executing the exploration loop.
s=open('explore_latent.py').read();ns={};exec(s[s.index('def ge_fit'):s.index('# Raw counts')],globals())
c=json.load(open('case.json'));r=c['support'];ref=c['reference_sequence'];y=np.array([v['fitness'] for v in r]);base=features(r,ref);extra=extra_features(r,ref)
folds=[]
for seed in [701,702]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,len(y))):folds.append((np.r_[np.arange(32),tr+32],te+32))
for tr,te in KFold(4,shuffle=True,random_state=703).split(y):folds.append((tr,te))
res=[]
for tr,te in folds:
 ps=[]
 for X in [base,extra]:
  model=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=1,max_features=.8,random_state=1729,n_jobs=1);model.fit(X[tr],y[tr]);ps.append(model.predict(X[te]))
 ps.append(ge_fit(extra[tr],y[tr],extra[te],.3));ensemble=np.mean([rankdata(p)/len(p) for p in ps],0)
 res.append([spearmanr(y[te],p).statistic for p in ps+[ensemble]])
print('base,chem,logistic,ensemble',np.mean(res[:8],axis=0),np.mean(res[8:],axis=0),np.std(res[:8],axis=0),flush=True)
