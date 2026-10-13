import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def fit_rank(K,kt,y,alpha=.1,gap=.1,power=0):
 ev,Q=np.linalg.eigh(K);H=Q*np.sqrt(np.maximum(ev,1e-9))[None,:]
 i,j=np.where(y[:,None]-y[None,:]>gap)
 if not len(i):return np.zeros(len(kt))
 weight=(y[i]-y[j])**power;weight*=len(y)/weight.sum()
 def obj(w):
  p=H@w;d=p[i]-p[j];r=-weight*expit(-d)
  gp=np.bincount(i,weights=r,minlength=len(y))-np.bincount(j,weights=r,minlength=len(y))
  return np.sum(weight*np.logaddexp(0,-d))+.5*alpha*(w@w),H.T@gp+alpha*w
 w=minimize(obj,np.zeros(len(y)),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-9}).x
 return kt@(Q@(w/np.sqrt(np.maximum(ev,1e-9))))
