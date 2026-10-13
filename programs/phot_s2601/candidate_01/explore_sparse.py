exec(open('explore.py').read().split('for scale in [')[0])
from sklearn.linear_model import Lasso,ElasticNet,ARDRegression,Ridge
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore',category=ConvergenceWarning)
p=PolynomialFeatures(2,interaction_only=True,include_bias=False).fit_transform(x); main=x.shape[1]
for centered in [False,True]:
 for s in ['raw','rank','sqrt']:
  for a in [.0003,.001,.003,.01,.03]:
   for ratio in [1.,.3]:
    def pred(tr,va,a=a,s=s,ratio=ratio,centered=centered):
     xc=x-x[tr].mean(0) if centered else x
     z=PolynomialFeatures(2,interaction_only=True,include_bias=False).fit_transform(xc);z[:,:main]*=10
     return ElasticNet(alpha=a,l1_ratio=ratio,max_iter=20000,tol=1e-6).fit(z[tr],transform(y[tr],s)).predict(z[va])
    evaluate(f'elastic/{centered}/{s}/{a}/{ratio}',pred)
for centered in [False,True]:
 for pw in [.05,.1,.2,.4]:
  for a in [.01,.03,.1,.3,1]:
   def pred(tr,va,a=a,pw=pw,centered=centered):
    xc=x-x[tr].mean(0) if centered else x
    z=PolynomialFeatures(2,interaction_only=True,include_bias=False).fit_transform(xc);z[:,main:]*=pw
    return ridge_pred(z,tr,va,a,'raw')
   evaluate(f'ridge2/{centered}/{pw}/{a}',pred)
# Handful of position-specific interactions, motivated by measured main effects.
positions=np.array([int(m[1:-1]) for m in vocab]); ii,jj=np.triu_indices(main,1)
for topk in [1,2,3,4,6]:
 for mode in ['both','either']:
  for pw in [.3,1,3]:
   for a in [.03,.3,3]:
    def pred(tr,va,k=topk,mode=mode,pw=pw,a=a):
     xm=x[tr].mean(0); xt=x[tr]-xm;yt=y[tr]-y[tr].mean()
     co=xt.T@solve(xt@xt.T+.1*np.eye(len(tr)),yt,assume_a='pos')
     poss=sorted(set(positions)); imp=np.array([np.linalg.norm(co[positions==p]) for p in poss]); chosen=np.array(poss)[np.argsort(imp)[-k:]]
     act=np.isin(positions,chosen); mask=(act[ii]&act[jj]) if mode=='both' else (act[ii]|act[jj]); mask &= positions[ii]!=positions[jj]
     xc=x-xm;z=np.column_stack([x,pw*xc[:,ii[mask]]*xc[:,jj[mask]]])
     return ridge_pred(z,tr,va,a,'raw')
    evaluate(f'hier/{topk}/{mode}/{pw}/{a}',pred)
for name,score in sorted(results.items(),key=lambda z:z[1][0],reverse=True)[:45]:print(name,*np.round(score,5),flush=True)
np.savez('sparse_predictions.npz',y=y,**preds)
json.dump(results,open('sparse_results.json','w'),indent=2)
# Baseline fit main coefficients and residuals.
b=Ridge(alpha=.03).fit(x,y);print('Main:',sorted(zip(vocab,np.round(b.coef_,4)),key=lambda z:z[1])); print('R2',b.score(x,y))
for i in np.argsort(abs(y-b.predict(x)))[-10:]: print(r[i]['mutation'],y[i],b.predict(x)[i],y[i]-b.predict(x)[i])
