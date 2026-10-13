import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
import numpy as np
from scipy.stats import rankdata,spearmanr
from scipy.linalg import solve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from proposal import AA,GROUPS

C=json.load(open('case.json')); R=C['support']; Y=np.array([r['fitness'] for r in R]); N=len(R)
S=[r['sequence'][100:144] for r in R]; ref=C['reference_sequence'][100:144]
order=np.array([sum(a!=b for a,b in zip(s,ref)) for s in S]); feed=np.flatnonzero(order>1)
# Fixed biochemical property tables, no external sequence data.
HYD=dict(zip(AA,[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
HEL=dict(zip(AA,[1.42,.7,1.01,1.51,1.13,.57,1.,1.08,1.16,1.21,1.45,.67,.57,1.11,.98,.77,.83,1.06,1.08,.69]))
sets=list(AA)+list(GROUPS)+['FW','W','Y','F','ILV','LM','ST','NQ','RK','ED','C','A','P','G']
props=np.array([[float(a in g) for g in sets]+[HYD[a],HEL[a],float(a in 'KR')-float(a in 'DE')] for a in AA])
T=np.array([[props[AA.index(a)] for a in s] for s in S])
OH=np.array([[[float(a==b) for b in AA] for a in s] for s in S])
features={}
features['comp']=T.sum(1)
features['compaa']=OH.sum(1)
for bins in [2,3,4,6]:
 edges=np.linspace(0,44,bins+1,dtype=int)
 features['bins'+str(bins)]=np.concatenate([T[:,lo:hi].sum(1) for lo,hi in zip(edges[:-1],edges[1:])],1)
features['motif']=np.concatenate([T[:,lo:hi].sum(1) for lo,hi in [(0,19),(6,14),(19,24),(24,44),(0,10),(10,19),(24,34),(34,44)]],1)
features['pos']=OH.reshape(N,-1)
features['posprop']=T[:,:,20:30].reshape(N,-1)
features['order']=np.stack([order,(order>15).astype(float)],1)
# Permutation-sensitive environment: acids and charges neighboring aromatics/hydrophobics.
P={g:np.array([[int(a in g) for a in s] for s in S]) for g in ['DE','KR','FWY','FWYLIVM','P','G','STNQ']}
pat=[]
for g in ['DE','KR','FWY','FWYLIVM','P','G']:
 for h in ['DE','KR','FWY','FWYLIVM','P','STNQ']:
  for lo,hi in [(1,4),(4,8),(8,15)]:
   mat=sum((P[g][:,:44-d]*P[h][:,d:]).sum(1)+(P[h][:,:44-d]*P[g][:,d:]).sum(1) for d in range(lo,hi))
   pat.append(mat/(hi-lo))
features['pairs']=np.array(pat).T
features['windows']=np.concatenate([np.stack([P[g][:,i:i+w].sum(1) for i in range(45-w)],1).max(1)[:,None] for g in P for w in [5,9,15]],1)
folds=[]
for seed in [19,43,71]:
 kf=StratifiedKFold(4,shuffle=True,random_state=seed)
 for tr,te in kf.split(feed,order[feed]>15):
  test=feed[te]; train=np.setdiff1d(np.arange(N),test)
  folds.append((train,test))

def target(y,kind):
 if kind=='rank': return (rankdata(y)-.5)/len(y)
 if kind=='raw': return y
 if kind=='robust': return np.clip(y,.75,1.95)
 if kind=='log': return np.log(np.maximum(y,.4))

def val(pred):
 # correlation per repetition and regime, avoids cross-fold offset artifacts
 vals=[]
 for j in range(3):
  vals.append(spearmanr(Y[feed],pred[j,feed]).statistic)
 low=feed[order[feed]<15]; high=feed[order[feed]>15]
 return np.mean(vals),np.std(vals),np.mean([spearmanr(Y[low],p[low]).statistic for p in pred]),np.mean([spearmanr(Y[high],p[high]).statistic for p in pred])

def ridge(X,alpha,kind='robust',sw=1.,kernel='linear',gamma=1.):
 pred=np.zeros((3,N))
 for j,(tr,te) in enumerate(folds):
  x=X[tr]; q=X[te]; mu=x.mean(0); sd=np.maximum(x.std(0),.5); x=(x-mu)/sd; q=(q-mu)/sd
  if kernel=='linear': k=x@x.T; kt=q@x.T
  elif kernel=='poly':
   k=(1+x@x.T/x.shape[1])**2; kt=(1+q@x.T/x.shape[1])**2
  elif kernel=='rbf':
   norm=(x*x).sum(1); qnorm=(q*q).sum(1)
   k=np.exp(-gamma*np.maximum(0,norm[:,None]+norm[None,:]-2*x@x.T)/x.shape[1]); kt=np.exp(-gamma*np.maximum(0,qnorm[:,None]+norm[None,:]-2*q@x.T)/x.shape[1])
  y=target(Y[tr],kind); weights=np.where(order[tr]==1,sw,1.)
  ym=np.average(y,weights=weights); y=y-ym
  a=solve(k+np.diag(alpha/weights),y,assume_a='pos')
  pred[j//4,te]=kt@a+ym
 return pred

if __name__=='__main__':
 start=time.time(); results=[]; preds={}
 combos=[['comp'],['compaa'],['bins2'],['bins3'],['bins4'],['comp','motif'],['comp','bins3'],['comp','bins3','order'],['comp','posprop'],['comp','pos'],['comp','bins3','pos'],['comp','pairs'],['comp','bins3','pairs'],['comp','windows'],['comp','motif','windows']]
 for combo in combos:
  X=np.concatenate([features[a] for a in combo],1)
  for kind in ['robust','rank','raw']:
   for alpha in [3,10,30,100,300]:
    for sw in [.3,1.]:
     name=('+'.join(combo),kind,alpha,sw,'linear',1.)
     p=ridge(X,alpha,kind,sw); v=val(p); results.append((v,name)); preds[str(name)]=p
 for combo in [['comp'],['comp','bins3'],['comp','motif'],['comp','bins3','pairs']]:
  X=np.concatenate([features[a] for a in combo],1)
  for kind in ['robust','rank']:
   for alpha in [.01,.1,1,10]:
    for kernel,gamma in [('poly',1),('rbf',.1),('rbf',.5),('rbf',2)]:
     name=('+'.join(combo),kind,alpha,.3,kernel,gamma)
     p=ridge(X,alpha,kind,.3,kernel,gamma); v=val(p); results.append((v,name)); preds[str(name)]=p
 results.sort(reverse=True)
 for v,name in results[:35]: print(tuple(round(z,4) for z in v),name,flush=True)
 np.savez('validation_predictions.npz',**preds)
 json.dump([{'metrics':list(v),'config':list(name)} for v,name in results],open('validation_results.json','w'),indent=2)
 print('seconds',time.time()-start,flush=True)
