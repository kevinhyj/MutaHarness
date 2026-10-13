exec(open('explore2.py').read().split('results=[]')[0])
from scipy.optimize import minimize
from scipy.special import expit
configs=[]
for f in ['context','small','shift','chem','pair']:
 for a in [.1,1,5]:
  for w in [0,.15,1]:configs.append(('wrank',f,(a,w)))
for freq in [1,2,3,4]:
 for scale in [.25,.5,1]:
  for a in [.3,1,3]:configs.append(('strictpair','context',(freq,scale,a)))
for f in ['small','context','chem']:
 for a in [.1,1,5]:configs.append(('sigmoid',f,a))
results=[];started=time.time()
for k,(kind,f,h) in enumerate(configs):
 XX=features[f];preds=np.full((3,len(s)),np.nan)
 for split,(tr,te) in enumerate(splits):
  y=rankdata(Y[tr])/len(tr)
  if kind=='wrank':
   w=np.where(N[tr]==1,h[1],1.)
   model=Ridge(alpha=h[0]);p=model.fit(XX[tr],y,sample_weight=w).predict(XX[te])
  elif kind=='strictpair':
   pp=[(i,j) for i in range(d) for j in range(i) if v[i][0]!=v[j][0] and np.sum(X[tr,i]*X[tr,j])>=h[0]]
   ZZ=np.column_stack([X[:,i]*X[:,j]*h[1] for i,j in pp]);xx=np.column_stack([F,ZZ]);p=Ridge(alpha=h[2]).fit(xx[tr],y).predict(xx[te])
  else:
   xt=XX[tr];target=Y[tr]; weights=np.where(N[tr]==1,.15,1.)
   def fg(b):
    z=expit(xt@b[:-1]+b[-1]);res=1.2*z-target
    loss=np.sum(weights*res*res)+h*.04*np.sum(b[:-1]*b[:-1])+.0025*(b[-1]-1.6)**2
    grad=2*weights*res*1.2*z*(1-z)
    return loss,np.r_[xt.T@grad+2*h*.04*b[:-1],grad.sum()+.005*(b[-1]-1.6)]
   init=np.r_[np.zeros(XX.shape[1]),1.6];res=minimize(fg,init,jac=True,method='L-BFGS-B',options={'maxiter':250,'ftol':1e-9});p=XX[te]@res.x[:-1]+res.x[-1]
  preds[split//4,te]=p
 corr=np.array([spearmanr(Y[mid],p[mid]).statistic for p in preds]);results.append((corr.mean(),corr.std(),(kind,f,h),preds))
 if k%20==0:print(k,round(time.time()-started,1),flush=True)
results.sort(key=lambda r:r[0],reverse=True)
for mean,sd,conf,p in results[:35]:print(round(mean,4),round(sd,4),conf)
np.savez('validation_predictions3.npz',Y=Y,mid=mid,predictions=np.array([r[3] for r in results]))
json.dump([{'rho':r[0],'sd':r[1],'config':r[2]} for r in results],open('validation_results3.json','w'),indent=2)
