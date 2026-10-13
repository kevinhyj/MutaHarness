import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time
from scipy.stats import norm,rankdata,spearmanr
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge,HuberRegressor
from threadpoolctl import threadpool_limits
import proposal
threadpool_limits(1)
c=json.load(open('case.json'));s=c['support'];n=len(s);y=np.array([r['fitness'] for r in s]);order=np.array([r['mutation_order'] for r in s]);fb=np.where(order>1)[0];singles=np.where(order==1)[0];f=proposal.features(s,c['reference_sequence'])
X={'global':np.c_[f['comp'][:,:20],f['basic']], 'regional':np.c_[f['comp'][:,:80],f['basic']], 'comp':np.c_[f['comp'],f['basic']], 'phys':np.c_[f['phys'],f['basic']], 'phys_inter':np.c_[f['phys'],f['inter'],f['basic']], 'rphys':np.c_[f['phys'][:,:76],f['basic']], 'simple':np.c_[f['phys'][:,:7],f['phys'][:,19:26],f['phys'][:,38:45],f['phys'][:,57:64],f['inter'][:,15:19],f['basic']]}
# Alternative biologically concentrated descriptors, with no structure assumed.
rows=[]
for r in s:
 q=r['sequence'][100:144];a=np.array([v in 'DE' for v in q],float);h=np.array([v in 'WFYLIMV' for v in q],float);ar=np.array([v in 'FWY' for v in q],float);b=np.array([v in 'KR' for v in q],float)
 rows.append([a[:19].sum(),a[24:].sum(),a[14:30].sum(),h[:19].sum(),h[24:].sum(),ar[:19].sum(),ar[24:].sum(),b[:19].sum(),b[24:].sum(),q.count('P'),q.count('G'),sum(q[i] in 'FWYLIMV' for i in [19,22,23]),sum(q[i]==v for i,v in zip([19,22,23],'WLF')),float(q[19]=='W' and q[22]=='L' and q[23]=='F')])
X['focused']=np.c_[rows,f['basic']]
folds=[]
for seed in range(6):
 for tr,te in StratifiedKFold(4,shuffle=True,random_state=105+seed).split(fb,order[fb]>15):folds.append((seed,np.r_[singles,fb[tr]],fb[te]))
def transform(z,t):
 if t=='rank':return norm.ppf((rankdata(z)-.5)/len(z))
 if t=='clip':return np.clip(z,np.quantile(z,.04),np.quantile(z,.96))
 return z

def quality(p):
 return [round(np.mean([spearmanr(y[ix],z[ix]).statistic for z in p]),4) for ix in [fb,fb[order[fb]<15],fb[order[fb]>15]]]+[round(np.std([spearmanr(y[fb],z[fb]).statistic for z in p]),4)]
res=[];pred={};start=time.time()
def keep(key,p):
 q=quality(p);res.append((q,key));pred[str(key)]=p
# Sample weighting and regime-specific models: all decisions validated on feedback.
for name in X:
 x=X[name]
 for split in ['pooled','separate']:
  for sw in [0,0.25,1]:
   for alpha in [1,10,100]:
    for target in ['rank','clip']:
     p=np.zeros((6,n))
     for seed,tr,te in folds:
      ty=transform(y[tr],target)
      for regime in ([False,True] if split=='separate' else [None]):
       trmask=np.ones(len(tr),bool) if regime is None else (order[tr]>15)==regime;temask=np.ones(len(te),bool) if regime is None else (order[te]>15)==regime
       ix=tr[trmask];jx=te[temask]
       weights=np.where(order[ix]==1,sw,1.);mu=np.average(x[ix],axis=0,weights=weights);sd=np.maximum(np.sqrt(np.average((x[ix]-mu)**2,axis=0,weights=weights)),1.)
       a=(x[ix]-mu)/sd;b=(x[jx]-mu)/sd;t=ty[trmask];m=np.average(t,weights=weights)
       K=a@a.T;w=np.sqrt(weights);coef=cho_solve(cho_factor(K*w[:,None]*w[None,:]+alpha*np.eye(len(ix))),(t-m)*w);p[seed,jx]=b@a.T@(coef*w)+m
     keep((name,'ridge',split,sw,alpha,target),p)
 print(name,'ridge best',sorted([r for r in res if r[1][0]==name],reverse=True)[:2],flush=True)
for name in ['global','regional','phys_inter','simple','focused']:
 x=X[name]
 for model in ['extra','forest','gbr','hist']:
  for param in [2,5,10]:
   for target in ['rank','clip']:
    p=np.zeros((6,n))
    for seed,tr,te in folds:
     if model=='extra':m=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=param,max_features=.8,random_state=12,n_jobs=1)
     if model=='forest':m=RandomForestRegressor(n_estimators=100,min_samples_leaf=param,max_features=.8,random_state=12,n_jobs=1)
     if model=='gbr':m=GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=param,learning_rate=.03,loss='huber',random_state=12)
     if model=='hist':m=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=4,min_samples_leaf=param,l2_regularization=10,learning_rate=.03,random_state=12)
     m.fit(x[tr],transform(y[tr],target));p[seed,te]=m.predict(x[te])
    keep((name,model,param,target),p)
  print(name,model,'best',sorted([r for r in res if r[1][0]==name and r[1][1]==model],reverse=True)[:1],'seconds',round(time.time()-start),flush=True)
ranked=sorted(res,reverse=True);print('TOP',ranked[:40],flush=True)
np.savez('exploration2_predictions.npz',**{'p'+str(i):pred[str(k)] for i,(_,k) in enumerate(ranked)})
json.dump([{'quality':q,'config':k} for q,k in ranked],open('validation2_results.json','w'),indent=2)
