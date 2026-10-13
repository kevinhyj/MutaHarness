from experiment import *
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics.pairwise import euclidean_distances
c=json.load(open('case.json'));rs=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in rs]);m=np.array([len(mutations(r,ref)) for r in rs]);single=np.flatnonzero(m==1);multi=np.flatnonzero(m>1);X=design(rs,ref,'mut',0,0);L=X@X.T;D=euclidean_distances(X,squared=True);Ks=[L,L+.2*L*L/8,np.exp(-.03*D)];alphas=[.5,1,.005]
results={}
for fraction in [.5,.75]:
 vals=[]; ensembles=[]; nofeedback=[]
 for ti,vi in ShuffleSplit(n_splits=16,train_size=fraction,random_state=20261013).split(multi):
  tr=np.r_[single,multi[ti]];te=multi[vi];yt=trans(y[tr],'rank');base=np.median(yt[m[tr]==1]);ps=[]
  for K,a in zip(Ks,alphas):ps.append(K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+a*np.eye(len(tr)),yt-base,assume_a='pos')+base)
  model=ElasticNet(alpha=.004,l1_ratio=.2,max_iter=5000,tol=1e-7);model.fit(X[tr],yt);ps.append(model.predict(X[te]));ps=np.array(ps)
  vals.append([metric(y[te],p) for p in ps]);ensembles.append([metric(y[te],np.mean(ps,axis=0)),metric(y[te],np.mean([rankdata(p) for p in ps],axis=0)),metric(y[te],.5*ps[0]+.5*ps[3])])
  b=np.median(y[single]);nf=L[np.ix_(te,single)]@solve(L[np.ix_(single,single)]+.5*np.eye(len(single)),y[single]-b,assume_a='pos')+b;nofeedback.append(metric(y[te],nf))
 print('FRACTION',fraction,'models ridge/poly/rbf/en',np.round(np.mean(vals,axis=0),4),'ensembles raw/rank/two',np.round(np.mean(ensembles,axis=0),4),'singles only',np.mean(nofeedback),flush=True)
 results[str(fraction)]={'models':vals,'ensembles':ensembles,'singles_only':nofeedback}
json.dump(results,open('stability_results.json','w'),indent=2)
# Original validation ensembles and within-region diagnostics.
z1=np.load('validation_predictions.npz');z2=np.load('validation2_predictions.npz')
keys=[str((('mut',0,0),'rank',.5)),str(('poly',('mut',0,0),'rank',.2,1)),str(('rbf',('mut',0,0),'rank',.03,.005)),str(('en',('mut',0,0),'rank',.2,.004))]
ps=np.array([z1[keys[0]]]+[z2[k] for k in keys[1:]])[:,:,multi];pp=ps.mean(axis=0);rr=np.mean([[rankdata(p) for p in model] for model in ps],axis=0)
print('FOURFOLD RAW',[metric(y[multi],p) for p in pp],flush=True);print('FOURFOLD RANK',[metric(y[multi],p) for p in rr],flush=True)
g=np.array([group(mutations(r,ref)) for r in rs])[multi]
for family in sorted(set(g)):
 mask=g==family
 if sum(mask)>=3:print('REGION',family,sum(mask),np.mean([metric(y[multi][mask],p[mask]) for p in pp]),flush=True)
