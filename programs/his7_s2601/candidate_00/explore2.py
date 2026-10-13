exec(open('explore.py').read().split('configs=[]')[0])
from sklearn.linear_model import Lasso,ElasticNet
from scipy.optimize import least_squares
# Physicochemical properties: hydropathy, approximate residue volume, charge,
# aromaticity, proline and glycine flags. Constants do not encode assay outcomes.
hydro=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
volume=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88,108,111,138,190,60,153,167,169,167,163,114,113,144,173,89,116,140,228,194]))
def prop(a):return np.array([hydro[a]/4,volume[a]/150,{'D':-1,'E':-1,'K':1,'R':1,'H':.2}.get(a,0),float(a in 'FWY'),float(a=='P'),float(a=='G'),float(a in 'STNQ')])
B=np.zeros((len(s),len(positions)*7))
for i,r in enumerate(s):
 for p,a,b in mutations(r,ref):
  j=positions.index(p);B[i,j*7:(j+1)*7]=prop(b)-prop(a)
features['chem']=np.column_stack([F,B*.6,B*M*.6])
features['small']=np.column_stack([X,G*M])
features['shift']=np.column_stack([X*M,G*M])
results=[]; started=time.time()
configs=[]
for f in ['main','small','context','shift','chem','pair']:
 for a in [.05,.3,1,3,10]:
  for w in [0,.1,1]:
   for t in ['raw','logit']:configs.append(('priorridge',f,(a,w),t))
for f in ['small','context','chem']:
 for a in [.005,.02,.06]:
  for l1 in [.5,1.]:configs.append(('elastic',f,(a,l1),'raw'))
for f in ['small','context','chem']:
 for a in [.3,1,3]:
  for w in [0,.1]:configs.append(('sigmoid',f,(a,w),'raw'))
for k,(kind,f,h,t) in enumerate(configs):
 XX=features[f]; preds=np.full((3,len(s)),np.nan)
 for split,(tr,te) in enumerate(splits):
  y=transforms[t](Y[tr]); weights=np.where(N[tr]==1,h[1] if kind!='elastic' else .2,1.)
  base=transforms[t](np.array([np.median(Y[sid])]))[0]
  # rank/logit base transform here uses fixed assay-like saturation, no WT label.
  if kind=='priorridge':
   model=Ridge(alpha=h[0],fit_intercept=False);p=base+model.fit(XX[tr],y-base,sample_weight=weights).predict(XX[te])
  elif kind=='elastic':
   model=ElasticNet(alpha=h[0],l1_ratio=h[1],max_iter=5000,fit_intercept=False);p=base+model.fit(XX[tr],y-base,sample_weight=weights).predict(XX[te])
  else:
   from scipy.special import expit
   xt=XX[tr]; root=np.sqrt(weights)
   def fun(beta):
    eta=xt@beta[:-1]+beta[-1]
    return np.r_[root*(1.2*expit(eta)-y),np.sqrt(h[0])*.2*beta[:-1],.05*(beta[-1]-1.6)]
   def jac(beta):
    z=expit(xt@beta[:-1]+beta[-1]);v=1.2*z*(1-z)*root
    return np.vstack([np.column_stack([xt*v[:,None],v]),np.column_stack([np.eye(xt.shape[1])*np.sqrt(h[0])*.2,np.zeros(xt.shape[1])]),np.r_[np.zeros(xt.shape[1]),.05]])
   init=np.r_[np.zeros(XX.shape[1]),1.6];res=least_squares(fun,init,jac=jac,max_nfev=70);p=XX[te]@res.x[:-1]+res.x[-1]
  preds[split//4,te]=p
 corr=np.array([spearmanr(Y[mid],p[mid]).statistic for p in preds]);results.append((corr.mean(),corr.std(),(kind,f,h,t),preds))
 if k%30==0:print(k,round(time.time()-started,1),flush=True)
results.sort(key=lambda r:r[0],reverse=True)
for mean,sd,conf,p in results[:40]:print(round(mean,4),round(sd,4),conf)
np.savez('validation_predictions2.npz',Y=Y,mid=mid,predictions=np.array([r[3] for r in results]))
json.dump([{'rho':r[0],'sd':r[1],'config':r[2]} for r in results],open('validation_results2.json','w'),indent=2)
