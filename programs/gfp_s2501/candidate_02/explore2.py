import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json
import numpy as np
from scipy.stats import spearmanr,rankdata
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
import proposal,extra_features
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);ref=c['reference_sequence'];x,v=extra_features.enhanced(s,ref)
base=79+len(v)+len(ref)+400
feedback=np.array([i for i,r in enumerate(s) if r['id'].startswith('M')]);original=np.array([i for i in range(len(s)) if i not in feedback]);splits=[]
for rep in range(3):
 for tr,te in KFold(4,shuffle=True,random_state=150+rep).split(feedback):splits.append((np.r_[original,feedback[tr]],feedback[te]))
def nonlinear(xt,yt,zt,alpha,loss):
    yy=np.clip((yt-1.3)/2.6,0,1)
    xt=np.c_[np.ones(len(xt)),xt];zt=np.c_[np.ones(len(zt)),zt]
    def fun(w):
        eta=xt@w;p=expit(eta)
        if loss=='logistic':
            cost=np.sum(np.logaddexp(0,eta)-yy*eta);grad=xt.T@(p-yy)
        else:
            cost=np.sum((p-yy)**2)*2;grad=xt.T@(4*(p-yy)*p*(1-p))
        cost+=alpha*np.sum(w[1:]**2)/2;grad[1:]+=alpha*w[1:]
        return cost,grad
    w=np.zeros(xt.shape[1]);w[0]=logit(np.clip(yy.mean(),.1,.9))
    res=minimize(fun,w,jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-8})
    return zt@res.x
preds={};results=[]
for feat in ['burial','enhanced','site','wt']:
 if feat=='burial':f=np.c_[x[:,:base],x[:,base:base+22]]
 elif feat=='site':f=np.c_[x[:,:base+22],x[:,base+22:base+22+len(ref)*6]]
 elif feat=='wt':f=np.c_[x[:,:base+22],x[:,-200:]]
 else:f=x
 for target in ['raw','binary','latent','logistic','mse']:
  for alpha in ([.1,1,10] if target in ['logistic','mse'] else [10,50,200]):
   name=f'{feat}_{target}_{alpha}';p=np.full((3,len(s)),np.nan)
   for j,(tr,te) in enumerate(splits):
    mean=f[tr].mean(0);std=np.maximum(f[tr].std(0),.5);xt=(f[tr]-mean)/std;zt=(f[te]-mean)/std;yy=y[tr]
    if target in ['logistic','mse']:pr=nonlinear(xt,yy,zt,alpha,target)
    else:
     if target=='binary':yy=(yy>2.8).astype(float)
     if target=='latent':yy=logit(np.clip((yy-1.3)/2.6,.015,.985))
     pr=Ridge(alpha=alpha).fit(xt,yy).predict(zt)
    p[j//4,te]=pr
   preds[name]=p;rh=[spearmanr(y[feedback],pp[feedback]).statistic for pp in p];results.append((np.mean(rh),np.std(rh),name))
 for kind in ['ET','RF']:
  for leaf in [2,5,10]:
   name=f'{feat}_{kind}_{leaf}';p=np.full((3,len(s)),np.nan)
   for j,(tr,te) in enumerate(splits):
    cls=ExtraTreesRegressor if kind=='ET' else RandomForestRegressor
    p[j//4,te]=cls(n_estimators=200,min_samples_leaf=leaf,max_features=.8,random_state=14,n_jobs=1).fit(f[tr],y[tr]).predict(f[te])
   preds[name]=p;rh=[spearmanr(y[feedback],pp[feedback]).statistic for pp in p];results.append((np.mean(rh),np.std(rh),name))
 print(feat,sorted(results,reverse=True)[:6],flush=True)
np.savez('cv_predictions2.npz',**preds);json.dump(sorted(results,reverse=True),open('cv_results2.json','w'),indent=2)
for a in sorted(results,reverse=True)[:25]:print(a)
