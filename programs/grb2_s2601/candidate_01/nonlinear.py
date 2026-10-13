import numpy as np
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
import proposal

def mutation_features(tokens,ref,ws=8.,we=.1,wc=.25):
 L=len(ref);X=np.zeros((len(tokens),L*(21+proposal.PROPS.shape[1])))
 for i,(p,a,b) in enumerate(tokens):
  j=proposal.AA.index(b); X[i,p]=np.sqrt(ws)
  X[i,L+p*20+j]=np.sqrt(we)
  X[i,L*21+p*proposal.PROPS.shape[1]:L*21+(p+1)*proposal.PROPS.shape[1]]=np.sqrt(wc/proposal.PROPS.shape[1])*proposal.PROPS[j]
 return X

def pooled_predict(rs, qs, ref, ws=8.,we=.1,wc=.25,alpha=.5,t=.1,mix=1):
 muts=[proposal.mutations(r,ref) for r in rs+qs]
 tokens=sorted(set(m for ms in muts[:len(rs)] for m in ms)); ids={m:i for i,m in enumerate(tokens)}
 X=mutation_features(tokens,ref,ws,we,wc); K=X@X.T
 C=cholesky(K+1e-8*np.eye(len(K)),lower=True)
 mmax=max([len(ms) for ms in muts]+[1]); idx=np.zeros((len(rs),mmax),int);mask=np.zeros_like(idx,dtype=bool)
 for i,ms in enumerate(muts[:len(rs)]):
  for j,m in enumerate(ms):idx[i,j]=ids[m];mask[i,j]=True
 y=np.array([r['fitness'] for r in rs]); counts=mask.sum(1)
 # Ridge additive start, same regularizer in the whitened mutation space.
 A=np.zeros((len(rs),len(tokens)))
 for i in range(len(rs)):
  for j in idx[i,mask[i]]: A[i,j]+=1
 H=np.c_[A@C,np.ones(len(rs)),counts-1]
 reg=np.r_[np.full(len(tokens),alpha),.01,.3]
 initial=solve(H.T@H+np.diag(reg),H.T@y,assume_a='pos')
 def objective(v):
  z=C@v[:-2]; zv=z[idx]; logits=np.where(mask,-zv/t,-1e6)
  probs=softmax(logits,axis=1)
  sm=-t*(logsumexp(logits,axis=1)-np.log(np.maximum(counts,1)))
  pred=(1-mix)*np.sum(zv*mask,axis=1)+mix*sm+v[-2]+v[-1]*(counts-1)
  residual=pred-y
  gradz=np.zeros(len(tokens)); g=(1-mix)*mask+mix*probs
  np.add.at(gradz,idx.ravel(),(residual[:,None]*g).ravel())
  grad=np.r_[C.T@gradz,residual.sum(),residual@(counts-1)]+reg*v
  return .5*(residual@residual+np.sum(reg*v*v)),grad
 fit=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':250,'ftol':1e-9,'gtol':1e-6})
 # Cross-kernel extension of latent mutation values. No labels from queries.
 qt=sorted(set(m for ms in muts[len(rs):] for m in ms)); qids={m:i for i,m in enumerate(qt)}
 Q=mutation_features(qt,ref,ws,we,wc)
 z=(Q@X.T)@solve(C.T,fit.x[:-2],assume_a='gen')
 out=[]
 for ms in muts[len(rs):]:
  zz=np.array([z[qids[m]] for m in ms]); n=len(zz)
  sm=-t*(logsumexp(-zz/t)-np.log(n)) if n else 0
  out.append((1-mix)*zz.sum()+mix*sm+fit.x[-2]+fit.x[-1]*(n-1))
 return np.array(out)
