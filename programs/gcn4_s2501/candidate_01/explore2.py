from explore import *
start=time.time(); results=[]; preds={}
regions={'global':[(0,44)],'two':[(0,22),(22,44)],'three':[(0,14),(14,29),(29,44)],'domain':[(0,19),(19,24),(24,44)],'fine':[(0,10),(10,19),(19,24),(24,34),(34,44)]}
smallgroups=['DE','KR','FWY','FWYLIVM','P','G','STNQ']
for name,reg in regions.items():
 features[name]=np.stack([P[g][:,lo:hi].sum(1) for lo,hi in reg for g in smallgroups],1)
features['hydro']=np.stack([[sum(HYD[a] for a in s[lo:hi]) for s in S] for lo,hi in [(0,44),(0,14),(14,29),(29,44)]],1)
configs=[['global'],['three'],['domain'],['fine'],['global','three'],['global','domain'],['global','fine'],['global','three','hydro'],['global','three','windows']]
for combo in configs:
 X=np.concatenate([features[a] for a in combo],1)
 for interact in [False,True]:
  xx=np.concatenate([X,X*(order>15)[:,None],features['order']],1) if interact else X
  for kind in ['rank','robust','raw']:
   for alpha in [1,3,10,30,100]:
    for sw in [.1,.3,1.]:
     name=('+'.join(combo),kind,alpha,sw,'gated' if interact else 'linear',1.)
     p=ridge(xx,alpha,kind,sw); v=val(p); results.append((v,name)); preds[str(name)]=p
 for kind in ['rank','robust']:
  for alpha in [.03,.1,.3,1,3]:
   for gamma in [.1,.3,1,3]:
    name=('+'.join(combo),kind,alpha,.3,'rbf',gamma)
    p=ridge(X,alpha,kind,.3,'rbf',gamma); v=val(p); results.append((v,name)); preds[str(name)]=p
for combo in [['global','three'],['comp','bins3'],['global','domain','windows']]:
 X=np.concatenate([features[a] for a in combo]+[features['order']],1)
 for model in ['extra','forest','boost']:
  for kind in ['rank','robust']:
   for leaf in [2,5,10]:
    for sw in [.3,1.]:
     pred=np.zeros((3,N))
     for j,(tr,te) in enumerate(folds):
      if model=='extra': m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=leaf,max_features=0.8,random_state=123,n_jobs=1)
      elif model=='forest': m=RandomForestRegressor(n_estimators=120,min_samples_leaf=leaf,max_features=.8,random_state=123,n_jobs=1)
      else: m=GradientBoostingRegressor(n_estimators=80,max_depth=2,min_samples_leaf=leaf,learning_rate=.035,loss='huber',random_state=123)
      m.fit(X[tr],target(Y[tr],kind),sample_weight=np.where(order[tr]==1,sw,1.))
      pred[j//4,te]=m.predict(X[te])
     name=('+'.join(combo),kind,leaf,sw,model,1.)
     v=val(pred); results.append((v,name)); preds[str(name)]=pred
 results.sort(reverse=True)
 print('TREE DONE',combo,time.time()-start,flush=True)
results.sort(reverse=True)
for v,name in results[:45]: print(tuple(round(z,4) for z in v),name,flush=True)
np.savez('validation_predictions2.npz',**preds)
json.dump([{'metrics':list(v),'config':list(name)} for v,name in results],open('validation_results2.json','w'),indent=2)
print('seconds',time.time()-start,flush=True)
