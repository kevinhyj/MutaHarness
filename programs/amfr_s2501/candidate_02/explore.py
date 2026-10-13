import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,itertools
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));records=c['support'];ref=c['reference_sequence'];n=len(records)
y=np.array([r['fitness'] for r in records]); I,P,Z,G=proposal.features(records,ref)
props=proposal.properties();d=props.shape[1];L=len(ref)
mut=np.full((n,L),-1,int)
for i,r in enumerate(records):
 for j,(a,b) in enumerate(zip(ref,r['sequence'])):
  if a!=b:mut[i,j]=proposal.AA.index(b)

# Cross-site transfer via chemical identity and sequence-inferred environments.
env=[]
h=np.array(proposal.HYDRO)
for j,a in enumerate(ref):
 win=np.arange(max(0,j-5),min(L,j+6))
 vals=np.array([h[proposal.AA.index(ref[k])] for k in win])
 # Helical i +/- 3/4 neighbors as a possible side-chain packing environment.
 same=[k for k in [j-7,j-4,j-3,j+3,j+4,j+7] if 0<=k<L]
 face=np.mean([h[proposal.AA.index(ref[k])] for k in same])
 env.append(np.r_[1,props[proposal.AA.index(a)],np.mean(vals)/3,face/3,float(5<=j<=16 or 22<=j<=29 or 31<=j<=43),float(j in [17,18,19,20])])
env=np.array(env)
# Transfer features are scaled equally, avoiding support-dependent standardization.
T=np.einsum('nld,le->nde',P.reshape(n,L,d),env).reshape(n,-1)
counts=Z.sum(1); base=np.ones((n,n));
kernels={'one':I@I.T,'pos':Z@Z.T,'phys':P@P.T/d,'global':G@G.T/d,'transfer':T@T.T/T.shape[1]}
for width in [.5,1,2,4]:
 A=np.exp(-((props[:,None]-props[None,:])**2).mean(2)/(2*width**2))
 K=np.zeros((n,n));Kd=np.zeros_like(K)
 for j,a in enumerate(ref):
  ids=np.where(mut[:,j]>=0)[0]; aa=mut[ids,j];wt=proposal.AA.index(a)
  K[np.ix_(ids,ids)]+=A[np.ix_(aa,aa)]
  Kd[np.ix_(ids,ids)]+=A[np.ix_(aa,aa)]-A[aa,wt][:,None]-A[wt,aa][None,:]+A[wt,wt]
 kernels['rbf'+str(width)]=K;kernels['diff'+str(width)]=Kd
splits=[]
for seed in [13,41,97]:
 for tr,te in KFold(5,shuffle=True,random_state=seed).split(y):splits.append(('all',tr,te))
 feedback=np.arange(32,n)
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feedback):splits.append(('feedback',np.r_[np.arange(32),feedback[tr]],feedback[te]))

configs=[]
for aa in ['one','rbf0.5','rbf1','rbf2','diff0.5','diff1','diff2']:
 for pos,phys,transfer in [(0,0,0),(.5,0,0),(.5,.25,0),(.5,.25,1),(.5,.25,4),(1,0,1)]:
  for alpha in [.03,.1,.3,1,3]:
   configs.append(dict(aa=aa,pos=pos,phys=phys,transfer=transfer,global_=0,inter=0,alpha=alpha,target='raw'))

def matrix(cfg):
 k=kernels[cfg['aa']]+cfg['pos']*kernels['pos']+cfg['phys']*kernels['phys']+cfg['transfer']*kernels['transfer']+cfg['global_']*kernels['global']
 return base+k+cfg['inter']*(k*k)/3

def target(name,tr):
 z=y[tr]
 if name=='rank':return (rankdata(z)-.5)/len(z)*3
 if name=='softfloor':return np.log(np.expm1(np.maximum(z+3.6,.05)))
 if name=='floor4':return np.log(np.expm1(np.maximum(z+4.0,.05)))
 return z

def evaluate(cfg,details=False):
 K=matrix(cfg);out=[];preds=[]
 for kind,tr,te in splits:
  pred=K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+cfg['alpha']*np.eye(len(tr)),target(cfg['target'],tr),assume_a='pos')
  out.append((kind,spearmanr(pred,y[te]).statistic));preds.append((te,pred))
 all_=np.mean([v for k,v in out if k=='all']);feed=np.mean([v for k,v in out if k=='feedback'])
 return (.35*all_+.65*feed,all_,feed,preds if details else None)
results=[];t=time.time()
for cfg in configs:
 score,all_,feed,_=evaluate(cfg);results.append((score,all_,feed,cfg))
results.sort(key=lambda x:x[0],reverse=True)
print('time',time.time()-t)
for row in results[:25]:print(row)
json.dump(results,open('exploration_results.json','w'),indent=2)
# Extend only promising distinct families; test global features/interactions and monotone target links.
best=[]
for row in results:
 if row[3]['aa'] not in [x['aa'] for x in best]:best.append(row[3])
 if len(best)==7:break
extra=[]
for cfg in best:
 for glob,inter,tgt in itertools.product([0,.25,1],[0,.1,.3,1],['raw','rank','softfloor','floor4']):
  for alpha in [.1,.3,1]:
   q=dict(cfg);q.update(global_=glob,inter=inter,target=tgt,alpha=alpha)
   score,all_,feed,_=evaluate(q);extra.append((score,all_,feed,q))
extra.sort(key=lambda x:x[0],reverse=True)
print('extended time',time.time()-t)
for row in extra[:30]:print(row)
json.dump(extra,open('exploration_extra.json','w'),indent=2)
