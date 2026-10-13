import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,numpy as np
from scipy.stats import spearmanr,rankdata,norm
from scipy.linalg import cho_factor,cho_solve
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import proposal
threadpool_limits(1)
c=json.load(open('case.json'));s=c['support']; n=len(s);y=np.array([r['fitness'] for r in s]);order=np.array([r['mutation_order'] for r in s]);feedback=np.where(order>1)[0]; singles=np.where(order==1)[0]
f=proposal.features(s,c['reference_sequence'])
# Prefixes allow direct tests of whole-domain vs region-specific representation.
f['global']=f['comp'][:,:20];f['regional']=f['comp'][:,:80];f['gphys']=f['phys'][:,:19];f['rphys']=f['phys'][:,:76]
blocks={
'global':['global','basic'], 'regional':['regional','basic'], 'comp':['comp','basic'],
'gphys':['gphys','basic'], 'rphys':['rphys','basic'], 'phys':['phys','basic'],
'comp_inter':['comp','inter','basic'], 'global_inter':['global','inter','basic'],
'phys_inter':['phys','inter','basic'], 'regional_inter':['regional','inter','basic'],
'pos':['pos','basic'], 'onehot':['onehot','basic'],
'comp_pos':['comp','pos','basic'], 'comp_onehot':['comp','onehot','basic'],
'global_pos':['global','pos','basic'], 'rphys_pos':['rphys','pos','basic']}
X={k:np.hstack([f[b] for b in v]) for k,v in blocks.items()}
folds=[]
for seed in range(6):
 for tr,te in StratifiedKFold(4,shuffle=True,random_state=105+seed).split(feedback,order[feedback]>15):folds.append((seed,np.r_[singles,feedback[tr]],feedback[te]))
def transform(y,name):
 if name=='raw':return y
 if name=='clip':return np.clip(y,np.quantile(y,.04),np.quantile(y,.96))
 if name=='log':return np.log1p(y)
 if name=='rank':return norm.ppf((rankdata(y)-.5)/len(y))

def quality(p):
 vals=[];lo=[];hi=[]
 for z in p:
  vals.append(spearmanr(y[feedback],z[feedback]).statistic)
  for mask,dest in [(order[feedback]<15,lo),(order[feedback]>15,hi)]:dest.append(spearmanr(y[feedback][mask],z[feedback][mask]).statistic)
 return [round(float(np.mean(v)),4) for v in [vals,lo,hi]]+[round(float(np.std(vals)),4)]
results=[];pred={};start=time.time()
for name,x in X.items():
 for kind in ['linear','rbf','poly']:
  settings=[1,3,10,30,100,300] if kind=='linear' else ([.3,1,3] if kind=='rbf' else [.3,1])
  alphas=[.03,.1,.3,1,3,10] if kind!='linear' else settings
  gammas=[None] if kind=='linear' else settings
  for gamma in gammas:
   for alpha in alphas:
    for target in ['raw','clip','log','rank']:
     key=(name,kind,gamma,alpha,target);p=np.zeros((6,n))
     for seed,tr,te in folds:
      floor=.3 if name in ['pos','onehot'] else 1.
      sd=np.maximum(x[tr].std(0),floor);mu=x[tr].mean(0);a=(x[tr]-mu)/sd;b=(x[te]-mu)/sd
      K=a@a.T;Q=b@a.T
      if kind=='rbf':
       scale=max(np.trace(K)/len(tr),1e-9);d=np.maximum(np.diag(K)[:,None]+np.diag(K)[None,:]-2*K,0);dt=np.maximum((b*b).sum(1)[:,None]+np.diag(K)[None,:]-2*Q,0);K=np.exp(-gamma*d/scale);Q=np.exp(-gamma*dt/scale)
      elif kind=='poly':
       scale=max(np.trace(K)/len(tr),1e-9);K=K/scale+gamma*(K/scale)**2;Q=Q/scale+gamma*(Q/scale)**2
      t=transform(y[tr],target);m=t.mean();coef=cho_solve(cho_factor(K+alpha*np.eye(len(tr))),t-m);p[seed,te]=Q@coef+m
     q=quality(p);results.append((q,key));pred[key]=p
 print(name,'best',sorted([r for r in results if r[1][0]==name],reverse=True)[:2], 'seconds',round(time.time()-start),flush=True)
ranked=sorted(results,reverse=True);print('TOP',ranked[:35],flush=True)
# Keep validation predictions to assess ensembles without fitting to query outcomes.
np.savez('exploration_predictions.npz',**{'p'+str(i):pred[k] for i,(_,k) in enumerate(ranked)})
json.dump([{'quality':q,'config':k} for q,k in ranked],open('validation_results.json','w'),indent=2)
