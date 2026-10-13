from exploration_utils import *
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
# A small, structured kernel family: local main effects plus smooth whole-sequence interactions.
configs=[]
for wone,wphys,wglobal,gamma in itertools.product([0,.1,.3],[.2,.5,1],[0,.1,.3],[.3,1,3]):
 k=kernels['one']*wone+kernels['phys']*wphys+kernels['global']*wglobal
 dist=np.maximum(np.diag(k)[:,None]+np.diag(k)[None,:]-2*k,0)
 key=f'new{wone},{wphys},{wglobal},{gamma}'
 kernels[key]=np.exp(-dist*gamma)
 for amp in [.3,1,3]:
  for alpha in [.1,.3,1]:
   kernels[key+'a'+str(amp)]=amp*kernels[key]
   configs.append(dict(aa=key+'a'+str(amp),pos=.5,phys=.25,transfer=0,global_=.25,inter=0,alpha=alpha,target='rank'))
res=[]
for cfg in configs:
 v=evaluate(cfg);res.append((*v[:3],cfg))
res.sort(reverse=True,key=lambda x:x[0]);json.dump(res,open('exploration_third.json','w'),indent=2)
for r in res[:20]:print(r,flush=True)
X=np.c_[P,Z,G,counts]; X=X[:,X.std(0)>1e-8]
res=[]
for leaf,depth,mtry in itertools.product([1,2,4],[None,4,6],[.5,1.]):
 out=[]
 for kind,tr,te in splits:
  model=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=leaf,max_depth=depth,max_features=mtry,random_state=734,n_jobs=1).fit(X[tr],target('rank',tr))
  pred=model.predict(X[te]);out.append((kind,spearmanr(pred,y[te]).statistic))
 a=np.mean([v for k,v in out if k=='all']);f=np.mean([v for k,v in out if k=='feedback'])
 res.append((.35*a+.65*f,a,f,leaf,depth,mtry))
res.sort(reverse=True,key=lambda x:x[0]);json.dump(res,open('exploration_trees.json','w'),indent=2)
print('Trees',flush=True)
for r in res[:10]:print(r,flush=True)
