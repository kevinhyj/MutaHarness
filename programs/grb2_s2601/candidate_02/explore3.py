import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from proposal import features
threadpool_limits(1)
c=json.load(open('case.json'));r=c['support'];y=np.array([z['fitness'] for z in r]);n=len(r);p,e,ch,g=features(r,c['reference_sequence'])
single=np.array([i for i,z in enumerate(r) if z['mutation_order']==1]);double=np.array([i for i,z in enumerate(r) if z['mutation_order']==2]);folds=[]
for seed in [2601,731]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(double):folds.append((np.r_[single,double[tr]],double[te]))
start=time.time();results=[];preds=[]
for pw in [.3,1,3]:
 for gw in [.05,.15,.5]:
  K=pw*(p@p.T)+.1*(e@e.T)+gw*(g@g.T)
  for lo,hi in [(-1.3,.3),(-1.5,.3),(-1.8,.5)]:
   for alpha in [.01,.04,.15,.5,1.5]:
    pr=np.zeros((2,n));nit=[]
    for f,(tr,te) in enumerate(folds):
     kt=K[np.ix_(tr,tr)];kx=K[np.ix_(te,tr)];row=kt.mean(1);grand=row.mean();kt=kt-row[:,None]-row[None,:]+grand;kx=kx-kx.mean(1)[:,None]-row[None,:]+grand
     vals,vec=eigh(kt);keep=vals>1e-7;vals=vals[keep];vec=vec[:,keep];X=vec*np.sqrt(vals);Xt=kx@(vec/np.sqrt(vals))
     yy=y[tr];zz=np.clip((yy-lo)/(hi-lo),.025,.975);target=np.log(zz/(1-zz));b=target.mean();w=(X.T@(target-b))/(vals+alpha*10)
     def loss(theta):
      z=X@theta[:-1]+theta[-1];sig=expit(z);pred=lo+(hi-lo)*sig;res=pred-yy;der=res*(hi-lo)*sig*(1-sig)
      return .5*res@res+.5*alpha*(theta[:-1]@theta[:-1]),np.r_[X.T@der+alpha*theta[:-1],der.sum()]
     opt=minimize(loss,np.r_[w,b],jac=True,method='L-BFGS-B',options={'maxiter':160,'ftol':1e-9})
     pr[f//4,te]=Xt@opt.x[:-1]+opt.x[-1];nit.append(opt.nit)
    cors=[spearmanr(y[double],pr[z,double]).statistic for z in range(2)]
    results.append((float(np.mean(cors)),[pw,gw,lo,hi,alpha],cors,np.mean(nit)));preds.append(pr)
order=np.argsort([x[0] for x in results])[::-1]
print('seconds',time.time()-start)
for i in order[:20]:print(results[i])
json.dump(results,open('explore3_results.json','w'));np.savez('explore3_results.npz',preds=np.array(preds),order=order,doubles=double,y=y)
