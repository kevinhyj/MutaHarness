exec(open('explore.py').read().split('configs=[]')[0])
from scipy.optimize import minimize
from scipy.special import expit
z=x@x.T
kernels={'linear':z+2*bm@bm.T,'pair':z+2*bm@bm.T+.1*(z*z-z)/2}
configs=[(kernel,a,weight) for kernel in kernels for a in [.0003,.001,.003,.01]for weight in ['rank','equal','raw']]
preds={str(cfg):np.zeros((4,len(mi)))for cfg in configs}
start=time.time()
for rep in range(4):
 for tr,te in KFold(4,shuffle=True,random_state=711+rep).split(mi):
  train=np.r_[si,mi[tr]];test=mi[te];yt=y[train];ii,jj=np.where(yt[:,None]>yt[None,:]);u=rankdata(yt)/len(yt)
  for name,k in kernels.items():
   K=k[np.ix_(train,train)];evals,evecs=np.linalg.eigh(K);ev=np.maximum(evals,1e-8)
   F=evecs*np.sqrt(ev);Q=k[np.ix_(test,train)]@(evecs/np.sqrt(ev))
   for a in [.0003,.001,.003,.01]:
    for weight in ['rank','equal','raw']:
     weights={'rank':u[ii]-u[jj],'equal':np.ones(len(ii)),'raw':yt[ii]-yt[jj]}[weight];weights=weights/weights.sum()
     def fun(w):
      f=F@w;d=f[ii]-f[jj];s=-expit(-d)*weights
      gradf=np.bincount(ii,s,minlength=len(yt))-np.bincount(jj,s,minlength=len(yt))
      return np.sum(weights*np.logaddexp(0,-d))+a*(w@w)/2,F.T@gradf+a*w
     opt=minimize(fun,np.zeros(len(train)),jac=True,method='L-BFGS-B',options={'maxiter':120,'ftol':1e-8})
     preds[str((name,a,weight))][rep,te]=Q@opt.x
 print('rep',rep,'seconds',time.time()-start,flush=True)
res=[]
for k,p in preds.items():
 sc=[spearmanr(y[mi],pp).statistic for pp in p];res.append((np.mean(sc),np.std(sc),k,sc))
for row in sorted(res,reverse=True):print(row)
np.savez('explore_rank_predictions.npz',y=y[mi],**preds)
