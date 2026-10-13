import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from proposal import features,mutations,_P,_AI
threadpool_limits(1)
c=json.load(open('case.json'));r=c['support'];y=np.array([z['fitness'] for z in r]);n=len(r)
p,e,ch,g=features(r,c['reference_sequence']);ms=[mutations(z,c['reference_sequence']) for z in r]
core={161,163,175,177,182,183,185,193,194,205,210};pocket={160,165,209}
def grp(p):return 0 if p+1 in core else 1 if p+1 in pocket else 2 if c['reference_sequence'][p]=='G' else 3
atoms=sorted(set(m for muts in ms for m in muts));idx={x:i for i,x in enumerate(atoms)};A=np.zeros((n,len(atoms)))
for i,muts in enumerate(ms):
 for m in muts:A[i,idx[m]]=1
ps=np.array([m[0] for m in atoms]); aa=np.array([m[1] for m in atoms]);wt=np.array([_AI[c['reference_sequence'][m[0]]] for m in atoms]);gr=np.array([grp(m[0]) for m in atoms]);delta=_P[aa]-_P[wt]
D=((delta[:,None]-delta[None])**2).sum(2)/7
T=((_P[aa,None]-_P[None,aa])**2).sum(2)/7
W=((_P[wt,None]-_P[None,wt])**2).sum(2)/7
components={}
for bw in [.3,1,3]:
 components['wt'+str(bw)]=A@(np.exp(-(D+W)/bw))@A.T
 components['group'+str(bw)]=A@((gr[:,None]==gr[None])*np.exp(-D/bw))@A.T
 components['groupT'+str(bw)]=A@((gr[:,None]==gr[None])*np.exp(-T/bw))@A.T
components['groupPos']=A@(gr[:,None]==gr[None]).astype(float)@A.T
extra=np.zeros((n,60))
for i,muts in enumerate(ms):
 for pi,ai in muts:
  wi=_AI[c['reference_sequence'][pi]];extra[i,ai]+=1;extra[i,20+wi]+=1
  extra[i,40+ai]+=int(pi+1 in core)
components['aa']=extra@extra.T
single=np.array([i for i,z in enumerate(r) if z['mutation_order']==1]);double=np.array([i for i,z in enumerate(r) if z['mutation_order']==2]);folds=[]
for seed in [2601,731]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(double):folds.append((np.r_[single,double[tr]],double[te]))
configs=[]
for pw in [.3,1,3]:
 for gw in [.05,.15,.5]:
  for comp in ['none']+list(components):
   for ew in ([0] if comp=='none' else [.1,.5,2]):configs.append((pw,.1,gw,comp,ew))
results=[];preds=[];start=time.time()
for cfg in configs:
 pw,ew,gw,comp,weight=cfg;K=pw*(p@p.T)+ew*(e@e.T)+gw*(g@g.T)
 if weight:K+=weight*components[comp]
 for alpha in [.1,.4,1.5,5]:
  pr=np.zeros((5,2,n))
  for f,(tr,te) in enumerate(folds):
   kt=K[np.ix_(tr,tr)];kx=K[np.ix_(te,tr)];row=kt.mean(1);grand=row.mean();kt=kt-row[:,None]-row[None,:]+grand;kx=kx-kx.mean(1)[:,None]-row[None,:]+grand
   vals,vec=eigh(kt); inv=vec/(np.maximum(vals,0)+alpha)
   yy=y[tr];Y=[yy,norm.ppf((rankdata(yy)-.5)/len(yy))]
   for lo,hi in [(-1.35,.5),(-1.5,.5),(-1.35,.2)]:
    zz=np.clip((yy-lo)/(hi-lo),.025,.975);Y.append(np.log(zz/(1-zz)))
   Y=np.stack(Y,1);pred=kx@(inv@(vec.T@(Y-Y.mean(0))))+Y.mean(0)
   pr[:,f//4,te]=pred.T
  for ti in range(5):
   cors=[spearmanr(y[double],pr[ti,z,double]).statistic for z in range(2)]
   results.append((float(np.mean(cors)),cfg,alpha,ti,cors));preds.append(pr[ti])
order=np.argsort([x[0] for x in results])[::-1]
print('seconds',time.time()-start)
for i in order[:30]:print(results[i])
json.dump(results,open('explore2_results.json','w'));np.savez('explore2_results.npz',preds=np.array(preds),order=order,doubles=double,y=y)
