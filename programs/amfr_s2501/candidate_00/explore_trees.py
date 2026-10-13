exec(open('explore.py').read().split('configs=[]')[0])
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
x=proposal.encode(s,c['reference_sequence']);wt=np.array([proposal.INDEX[a] for a in c['reference_sequence']]);z=x-x.sum(2)[:,:,None]*np.eye(20)[wt][None,:,:];d=z@proposal.PROP
mut=x@proposal.PROP;present=x.sum(2)
base=np.c_[d.reshape(n,-1),present,d.sum(1),np.abs(d).sum(1),present.sum(1)]
cat=np.c_[base,x.reshape(n,-1)]
configs=[]
for typ in ['ET','RF','GB']:
 for leaf in [1,2,4,8]:
  for feature in ['base','cat']:
   for transform in ['raw','rank']:
    configs.append((typ,leaf,feature,transform))
res=[]
for cfg in configs:
 typ,leaf,feat,transform=cfg;f=base if feat=='base' else cat;pred=np.zeros((1,n))
 for j,(tr,va) in enumerate(folds[:4]):
  if typ=='ET':model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=.8,random_state=981,n_jobs=1)
  elif typ=='RF':model=RandomForestRegressor(n_estimators=160,min_samples_leaf=leaf,max_features=.8,random_state=981,n_jobs=1)
  else:model=GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=leaf,learning_rate=.04,random_state=981,loss='huber')
  model.fit(f[tr],trans[transform](y[tr]));pred[0,va]=model.predict(f[va])
 a=spearmanr(y[32:],pred[0,32:]).statistic;b=spearmanr(y[order==2],pred[0,order==2]).statistic
 res.append((.75*b+.25*a,a,b,cfg))
res.sort(reverse=True)
print(*res[:20],sep='\n')
json.dump(res,open('tree_results.json','w'),indent=2)
