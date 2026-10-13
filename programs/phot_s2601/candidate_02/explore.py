import json,time,collections
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr
from sklearn.model_selection import KFold,ShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet,ARDRegression
from proposal import tokens
c=json.load(open('case.json')); s=[r for r in c['support'] if len(tokens(r))>1]; n=len(s)
v=sorted(set(t for r in s for t in tokens(r))); x=np.array([[t in tokens(r) for t in v] for r in s],float); y=np.array([r['fitness'] for r in s]); q=[r for r in c['query'] if len(tokens(r))>1]
z=np.array([[t in tokens(r) for t in v] for r in q],float)
pairs=[(i,j) for i in range(len(v)) for j in range(i) if v[i][1:-1]!=v[j][1:-1]]
xp=np.array([x[:,i]*x[:,j] for i,j in pairs]).T
splits=[]
for seed in [17,31,53,89]:splits.extend(list(KFold(4,shuffle=True,random_state=seed).split(x)))
small=list(ShuffleSplit(8,train_size=32,random_state=117).split(x))

def transform(y,power):
 return np.log(y) if power==0 else (y**power-1)/power

def kernelpred(tr,te,k,power,alpha):
 yt=transform(y[tr],power); mean=k[tr][:,tr].mean(0); grand=mean.mean()
 kt=k[tr][:,tr]-mean[None,:]-mean[:,None]+grand
 kz=k[te][:,tr]; kz=kz-mean[None,:]-kz.mean(1)[:,None]+grand
 return kz@solve(kt+alpha*np.eye(len(tr)),yt-yt.mean(),assume_a='pos')+yt.mean()

def evaluate(name,predict):
 vals=[]; sm=[]; oof=np.zeros((4,n))
 for i,(tr,te) in enumerate(splits):
  p=predict(tr,te); vals.append(spearmanr(y[te],p).statistic);oof[i//4,te]=p
 for tr,te in small:sm.append(spearmanr(y[te],predict(tr,te)).statistic)
 result={'name':name,'cv':float(np.mean(vals)),'small':float(np.mean(sm)),'std':float(np.std(vals))}
 results.append(result); predictions[name]=oof
 return result
results=[]; predictions={};t=time.time()
# Genotype effects plus all pairwise effects; unequal regularization via kernels.
for center in [False,True]:
 xx=x-x.mean(0) if center else x
 pp=np.array([xx[:,i]*xx[:,j] for i,j in pairs]).T
 ka=xx@xx.T;kp=pp@pp.T
 for ratio in [0,.01,.03,.1,.3,1,3]:
  k=ka+ratio*kp
  for power in [0,.5,1,2]:
   for alpha in [.03,.1,.3,1,3,10]:
    name=f'pair_{center}_{ratio}_{power}_{alpha}'
    evaluate(name,lambda tr,te,k=k,power=power,alpha=alpha:kernelpred(tr,te,k,power,alpha))
# Very compact pairwise models: all pairs involving one given locus.
for pos in sorted(set(t[1:-1] for t in v),key=int):
 pp=xp[:,[v[i][1:-1]==pos or v[j][1:-1]==pos for i,j in pairs]]
 for ratio in [.1,1,10]:
  k=x@x.T+ratio*(pp@pp.T)
  for power in [0,.5,1]:
   for alpha in [.1,1,3]:
    evaluate(f'focus_{pos}_{ratio}_{power}_{alpha}',lambda tr,te,k=k,power=power,alpha=alpha:kernelpred(tr,te,k,power,alpha))
# Distance-based kernel, additive+nonparametric residual.
dist=((x[:,None,:]-x[None,:,:])**2).sum(2)
for length in [2,4,8,16]:
 for linear in [0,.1,1]:
  k=np.exp(-dist/length)+linear*x@x.T
  for power in [0,.5,1]:
   for alpha in [.003,.03,.3]:
    evaluate(f'rbf_{length}_{linear}_{power}_{alpha}',lambda tr,te,k=k,power=power,alpha=alpha:kernelpred(tr,te,k,power,alpha))
np.savez('exploration_oof.npz',**predictions)
json.dump(results,open('exploration_results.json','w'),indent=2)
print('Time',time.time()-t,'configurations',len(results))
for r in sorted(results,key=lambda r:-(.7*r['cv']+.3*r['small']))[:30]:print(r)
