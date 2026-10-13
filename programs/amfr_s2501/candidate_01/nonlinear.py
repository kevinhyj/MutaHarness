exec(open('refine.py').read().split('\nresults=[];start=time.time()')[0])
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
BORDER='ARNDCQEGHILKMFPSTWYV'
BM=np.array([
[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]])
ix=[BORDER.index(a) for a in proposal.AA];bm=BM[np.ix_(ix,ix)];eig,v=np.linalg.eigh(bm);sim=(v*np.maximum(eig,0))@v.T;sim/=np.mean(np.diag(sim))
KB=np.zeros((n,n));GB=np.zeros((n,20))
for j,w in enumerate(ref):
 b=proposal.AA.index(w);center=sim-sim[:,b,None]-sim[None,b,:]+sim[b,b]
 KB+=one[:,j]@center@one[:,j].T
 GB+=one[:,j]@(bm-bm[b])
kernels['B']=KB;kernels['GB']=GB@GB.T/(20*np.mean(bm**2))
classes=['ILVM','FYW','NQST','DE','KRH','A','C','G','P']
group=np.array([[float(a in g) for g in classes] for a in proposal.AA]);group=np.c_[group,proposal.descriptors()/3]
KG=np.zeros((n,n));GG=np.zeros((n,group.shape[1]))
for j,w in enumerate(ref):
 f=one[:,j]@(group-group[proposal.AA.index(w)]);KG+=f@f.T;GG+=f
kernels['CL']=KG;kernels['GCL']=GG@GG.T
results=[]
for aa in ['B','CL']:
 for s in [.2,1.,3.]:
  for g in [.3,1.,3.]:
   for rg in [.3,1.,3.,10.]:
    for pairs in [0.,.3,1.]:
     cfg={aa:1.,'S':s,'G':g,'pairS':pairs}
     for t in ['raw','rank']:
      m,p=evaluate(cfg,rg,t);results.append((m,cfg,rg,t))
# Mix previous chemical kernels with categorical substitution kernel.
for aa in ['P','R3.0']:
 for bb in ['O','B','CL']:
  for weight in [.1,.5,2.]:
   for rg in [.5,2.,5.]:
    for g in [.3,1.,3.]:
     cfg={aa:1.,bb:weight,'S':1.,'G':g,'pairS':.3}
     m,p=evaluate(cfg,rg,'rank');results.append((m,cfg,rg,'rank'))
for r in sorted(results,key=lambda a:a[0][0],reverse=True)[:20]:print(r)
json.dump([{'metric':list(m),'config':cfg,'reg':rg,'target':t} for m,cfg,rg,t in sorted(results,key=lambda a:a[0][0],reverse=True)],open('cv_blosum.json','w'),indent=2)
np.savez('exploration_kernels.npz',**kernels)
# Small-data tree benchmarks, using sitewise chemical and onehot features.
X=np.c_[one.reshape(n,-1),prop.reshape(n,-1)/3,site,glob/3,site.sum(1),cross,absprop]
X=X[:,X.std(0)>0]
rr=[]
ss=splits[:16]
for typ in ['ET','GB']:
 for leaf in [1,2,4]:
  for features in [.5,1.]:
   for rank in [True,False]:
    ps=[];ms=[];yy=[]
    for tr,va in ss:
     z=y[tr] if not rank else norm.ppf((rankdata(y[tr])-.5)/len(tr))
     if typ=='ET':model=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=leaf,max_features=features,random_state=42,n_jobs=1)
     else:model=GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=leaf,max_features=features,learning_rate=.04,random_state=42)
     model.fit(X[tr],z);p=model.predict(X[va]);ps.extend(p);yy.extend(y[va]);ms.append(spearmanr(p,y[va]).statistic)
    rr.append((np.mean(ms),typ,leaf,features,rank,spearmanr(ps,yy).statistic))
print('TREES',sorted(rr,reverse=True))
json.dump(rr,open('cv_trees.json','w'))
