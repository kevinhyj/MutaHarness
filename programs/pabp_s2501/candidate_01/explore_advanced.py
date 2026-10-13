exec(open('explore.py').read().split('rng=np.random')[0])
import advanced
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
base=json.load(open('cv_results.json'));start=time.time()
ek,desc=advanced.extra_kernels(s,ref);print('extra built',time.time()-start,flush=True)
bk=[]
for r in base[:12]:bk.append(np.einsum('k,kij->ij',r[6],kernels))
results=[]
rng=np.random.default_rng(3129)
for z in range(550):
 baseindex=int(rng.integers(12));r=base[baseindex];k=bk[baseindex].copy();extraweight=np.zeros(len(ek))
 if z<13: extraweight[z]=1
 else:
  j=int(rng.integers(4));extraweight[3*j:3*j+3]=10**rng.uniform([-1,-2,-3],[1.5,.5,-.3])
  extraweight[-1]=10**rng.uniform(-4,-1)
 k+=np.einsum('k,kij->ij',extraweight,ek)
 pairweight=0 if rng.random()<.5 else 10**rng.uniform(-2.5,0)
 if pairweight:k+=pairweight*(k*k)/(np.mean(np.diag(k)))
 alpha=r[5]*10**rng.uniform(-.3,.8)
 tf=r[4];target={'raw':y,'log05':np.log(y+.05),'log15':np.log(y+.15),'log01':np.log(y+.01),'rank':(rankdata(y)-.5)/n}[tf]
 sc,sd,allsc,out=cv(k,target,alpha);merit=.8*sc+.2*allsc-.1*sd
 results.append((merit,sc,sd,allsc,tf,float(alpha),r[6],extraweight.tolist(),pairweight,out.tolist()))
results.sort(reverse=True,key=lambda v:v[0]);json.dump(results,open('advanced_cv_results.json','w'))
print('KERNEL TOP',flush=True)
for r in results[:10]:print(r[:9],flush=True)
# Nonparametric regressors on low-dimensional, symmetric descriptions.
x=advanced.tree_features(s,ref);treeres=[]
for modelkind in ['extra','rf','hgb']:
 for tf in ['raw','log05','rank']:
  target={'raw':y,'log05':np.log(y+.05),'rank':(rankdata(y)-.5)/n}[tf]
  for leaf in [1,2,4,8]:
   out=np.zeros((5,n))
   for f,(tr,va) in enumerate(splits):
    if modelkind=='extra':model=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=leaf,max_features=.7,random_state=73,n_jobs=1)
    elif modelkind=='rf':model=RandomForestRegressor(n_estimators=150,min_samples_leaf=leaf,max_features=.7,random_state=73,n_jobs=1)
    else:model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=leaf,l2_regularization=3,random_state=73)
    model.fit(x[tr],target[tr]);out[f//4 if f<16 else 4,va]=model.predict(x[va])
   scores=[spearmanr(out[j,feedback],y[feedback]).statistic for j in range(4)]
   sc,sd=np.mean(scores),np.std(scores);allsc=spearmanr(out[4],y).statistic;merit=.8*sc+.2*allsc-.1*sd
   print('tree',modelkind,tf,leaf,round(merit,4),round(sc,4),round(allsc,4),flush=True)
   treeres.append([merit,sc,sd,allsc,modelkind,tf,leaf,out.tolist()])
treeres.sort(reverse=True,key=lambda v:v[0]);json.dump(treeres,open('tree_cv_results.json','w'))
