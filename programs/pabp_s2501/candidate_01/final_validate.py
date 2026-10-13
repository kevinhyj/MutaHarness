exec(open('explore.py').read().split('rng=np.random')[0])
import advanced
from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import minimize
from scipy.special import expit
adv=json.load(open('advanced_cv_results.json'));refined=json.load(open('refine_cv_results.json'));ek,_=advanced.extra_kernels(s,ref)
# Diverse, compact configurations: one best centered member per top kernel family.
configs=[]
for idx in [0,1,3,4,6,7]:
 row=next(r for r in refined if r[4]==idx and r[5] and r[6]==0)
 a=adv[idx];configs.append([a[5],a[6],a[7],a[8],row[7],True])
json.dump(configs,open('chosen_configs.json','w'))
ks=[]
for a,w,ew,p,tf,center in configs:
 k=np.einsum('k,kij->ij',w,kernels)+np.einsum('k,kij->ij',ew,ek)
 if p:k+=p*k*k/np.mean(np.diag(k))
 ks.append(k)
# Fresh splits unused in the configuration search.
new=[]
for seed in [1031,2051]:
 for tr,va in KFold(4,shuffle=True,random_state=seed).split(feedback):new.append((np.r_[orig,feedback[tr]],feedback[va]))
for tr,va in KFold(5,shuffle=True,random_state=407).split(y):new.append((tr,va))
x=advanced.tree_features(s,ref);o=np.zeros((len(configs)+2,3,n))
for f,(tr,va) in enumerate(new):
 for j,(k,config) in enumerate(zip(ks,configs)):
  alpha,_,_,_,tf,center=config
  # All response transforms and feature centering use training records only.
  target=np.log(y[tr]+float(tf[3:])/100)
  kt=k[np.ix_(tr,tr)].copy();kv=k[np.ix_(va,tr)].copy();m=kt.mean(axis=0);tot=kt.mean();kt=kt-m[:,None]-m[None,:]+tot;kv=kv-m[None,:]-kv.mean(axis=1)[:,None]+tot
  pred=kv@solve(kt+alpha*np.eye(len(tr)),target-target.mean(),assume_a='pos')+target.mean()
  o[j,f//4 if f<8 else 2,va]=pred
 for j,tf in enumerate(['raw','rank']):
  target=y[tr] if tf=='raw' else (rankdata(y[tr])-.5)/len(tr)
  model=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=1,max_features=.7,random_state=73,n_jobs=1);model.fit(x[tr],target)
  o[len(configs)+j,f//4 if f<8 else 2,va]=model.predict(x[va])

def metrics(v):
 return [float(spearmanr(v[i,feedback if i<2 else np.arange(n)],y[feedback if i<2 else np.arange(n)]).statistic) for i in range(3)]
# Rank-average only within each OOF population, since scales differ by transform.
ranked=np.zeros_like(o)
for j in range(len(o)):
 for i in range(3):
  ix=feedback if i<2 else np.arange(n);ranked[j,i,ix]=(rankdata(o[j,i,ix])-.5)/len(ix)
for j in range(len(o)):print('member',j,metrics(o[j]),flush=True)
ens=ranked[:len(configs)].mean(axis=0)
for tw in [0,.05,.1,.15,.2,.25,.35]:
 v=(1-tw)*ens+tw*ranked[-2:].mean(axis=0)
 print('ensemble tree weight',tw,metrics(v),flush=True)
json.dump({'configs':configs,'predictions':o.tolist()},open('fresh_validation.json','w'))
