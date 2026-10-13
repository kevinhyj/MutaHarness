import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='1'
import json,time
import numpy as np
from scipy.stats import spearmanr,rankdata
from scipy.linalg import solve
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor,GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from proposal import _matrix,_mutations
c=json.load(open('case.json')); ref=c['reference_sequence']
r=[r for r in c['support'] if len(_mutations(r,ref))>1]
vocab=sorted({m for a in r for m in _mutations(a,ref)})
x=_matrix(r,ref,vocab); y=np.array([a['fitness'] for a in r]); n=len(y)
splits=list(RepeatedKFold(n_splits=4,n_repeats=4,random_state=2026).split(x))
results={}; preds={}
def evaluate(name,predict):
 pp=np.zeros((4,n)); ss=[]
 for k,(tr,va) in enumerate(splits):
  p=predict(tr,va); pp[k//4,va]=p; ss.append(spearmanr(y[va],p).statistic)
 preds[name]=pp; results[name]=(np.mean(ss),spearmanr(y,pp.mean(axis=0)).statistic,np.mean((pp-y)**2)**.5)
def transform(y,scale):
 if scale=='raw': return y
 if scale=='log': return np.log(y+.01)
 if scale=='sqrt': return np.sqrt(y)
 if scale=='rank': return (rankdata(y)-.5)/len(y)
 if scale=='logit': return np.log((y+.03)/(1.7-y))
def ridge_pred(z,tr,va,alpha,scale):
 yt=transform(y[tr],scale); xm=z[tr].mean(0); ym=yt.mean(); zt=z[tr]-xm
 coef=zt.T@solve(zt@zt.T+alpha*np.eye(len(tr)),yt-ym,assume_a='pos')
 return ym+(z[va]-xm)@coef
p=PolynomialFeatures(2,interaction_only=True,include_bias=False).fit_transform(x); main=x.shape[1]
for scale in ['raw','log','sqrt','rank','logit']:
 for a in [.03,.1,.3,1,3,10]: evaluate(f'add/{scale}/{a}',lambda tr,va,a=a,s=scale:ridge_pred(x,tr,va,a,s))
 for pw in [.1,.3,1,3]:
  z=p.copy(); z[:,main:]*=pw
  for a in [.1,1,10]:evaluate(f'pair/{scale}/{pw}/{a}',lambda tr,va,a=a,s=scale,z=z:ridge_pred(z,tr,va,a,s))
# Sequence identity Gaussian kernels and main+pair kernels.
D=((x[:,None,:]-x[None,:,:])**2).sum(2)
for scale in ['raw','log','sqrt','rank']:
 for gamma in [.02,.05,.1,.2,.4]:
  K=np.exp(-gamma*D)
  for a in [.003,.03,.3]:
   def pred(tr,va,K=K,a=a,s=scale):
    yt=transform(y[tr],s); ym=yt.mean()
    return ym+K[np.ix_(va,tr)]@solve(K[np.ix_(tr,tr)]+a*np.eye(len(tr)),yt-ym,assume_a='pos')
   evaluate(f'rbf/{scale}/{gamma}/{a}',pred)
for leaf in [2,4,8]:
 for mf in [.6,1.]:
  for scale in ['raw','log']:
   def pred(tr,va,leaf=leaf,mf=mf,s=scale):
    return ExtraTreesRegressor(n_estimators=200,min_samples_leaf=leaf,max_features=mf,random_state=314,n_jobs=1).fit(x[tr],transform(y[tr],s)).predict(x[va])
   evaluate(f'extra/{scale}/{leaf}/{mf}',pred)
for depth in [1,2,3]:
 for trees in [80,200]:
  def pred(tr,va,depth=depth,trees=trees):
   return GradientBoostingRegressor(n_estimators=trees,max_depth=depth,min_samples_leaf=5,learning_rate=.03,random_state=314).fit(x[tr],y[tr]).predict(x[va])
  evaluate(f'boost/raw/{depth}/{trees}',pred)
for name,score in sorted(results.items(),key=lambda z:z[1][0],reverse=True)[:35]:print(name,*np.round(score,5),flush=True)
np.savez('explore_predictions.npz',y=y,**preds)
json.dump(results,open('explore_results.json','w'),indent=2)
print('vocab',vocab)
