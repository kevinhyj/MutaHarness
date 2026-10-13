exec(open('explore2.py').read().split('\nconfigs=[]')[0])
from scipy.optimize import minimize
from scipy.special import expit
# Pairwise ranking kernel logistic regression, evaluated on the same search/confirmation sets.
base=K['exact']+K['pos']+.03*K['region']+.1*K['count']
for alpha in [.001,.01,.1,1,10]:
 for smoothing in [0,.1]:
  metrics=[]
  for tr,va in splits:
   kt=base[np.ix_(tr,tr)];kv=base[np.ix_(va,tr)];mu=kt.mean(0);g=mu.mean()
   kt=kt-mu[:,None]-mu[None,:]+g;kv=kv-kv.mean(1)[:,None]-mu[None,:]+g
   vals,vecs=np.linalg.eigh(kt);keep=vals>1e-8;vals=vals[keep];vecs=vecs[:,keep];Z=vecs*np.sqrt(vals)
   T=.5+(.5-smoothing)*np.sign(y[tr,None]-y[None,tr])
   def loss(w):
    f=Z@w;z=f[:,None]-f[None,:]
    value=.5*np.sum(np.logaddexp(0,z)-T*z)/len(tr)+alpha/2*(w@w)
    grad=Z.T@np.sum(expit(z)-T,axis=1)/len(tr)+alpha*w
    return value,grad
   fit=minimize(loss,np.zeros(len(vals)),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-9})
   p=kv@((vecs/np.sqrt(vals))@fit.x);metrics.append(spearmanr(y[va],p).statistic)
  print(alpha,smoothing,np.mean(metrics[:20]),np.mean(metrics[20:50]),np.mean(metrics[50:]),flush=True)
