import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time,itertools
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold,ShuffleSplit
import proposal
c=json.load(open('case.json')); s=c['support']; n=len(s)
y=np.array([r['fitness'] for r in s]); orders=np.array([r['mutation_order'] for r in s]); feedback=np.arange(32,n)
P,E,D,G=proposal._features(s,c['reference_sequence'])
mut=[proposal._mutations(r,c['reference_sequence']) for r in s]
components={'pos':P@P.T,'exact':E@E.T,'prop':D@D.T,'global':G@G.T}
# Fixed amino-acid geometry; no data-trained amino-acid embedding.
for length in [.5,1.,2.]:
 ap=proposal.PROPS
 aak=np.exp(-((ap[:,None]-ap[None,:])**2).mean(2)/(2*length**2))
 for anchor in [False,True]:
  K=np.zeros((n,n))
  for i,mi in enumerate(mut):
   for j,mj in enumerate(mut):
    for p,w,a in mi:
     for q,v,b in mj:
      if p==q:
       ai=proposal.AA.index(a);bi=proposal.AA.index(b);wi=proposal.AA.index(w)
       K[i,j]+=aak[ai,bi]-(aak[ai,wi]+aak[wi,bi]-aak[wi,wi] if anchor else 0)
  components['rbf'+str(length)+str(anchor)]=K
# Protein WW topology is a hypothesis from motif; never imposed as a label.
def region(p):
 p+=1
 if 175<=p<=181:return 1
 if 186<=p<=190:return 2
 if 196<=p<=201:return 3
 if 182<=p<=185:return 4
 if 191<=p<=195:return 5
 return 0
B=np.zeros((n,6*30)); H=np.zeros((n,20*20))
for i,mi in enumerate(mut):
 for p,w,a in mi:
  ai=proposal.AA.index(a);wi=proposal.AA.index(w);r=region(p)
  B[i,30*r:30*r+10]+=(proposal.PROPS[ai]-proposal.PROPS[wi])/np.sqrt(10)
  B[i,30*r+10+ai]+=1; B[i,30*r+10+wi]-=1
  H[i,20*wi+ai]+=1
components['region']=B@B.T;components['substitution']=H@H.T
splits=[]
for seed in [21,83,127]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feedback):
  splits.append((np.r_[np.arange(32),feedback[tr]],feedback[te],'fb'))
for seed in [36,91]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(s):splits.append((tr,te,'all'))

def target(y,mode):
 if mode=='log':return np.log(y)
 if mode=='log.1':return np.log(y+.1)
 if mode=='sqrt':return np.sqrt(y)
 if mode=='raw':return y
 if mode=='rank':return norm.ppf((rankdata(y)-.5)/len(y))

def fitpred(K,y,tr,te,alpha,mode='log'):
 z=target(y[tr],mode); A=K[np.ix_(tr,tr)];C=K[np.ix_(te,tr)]
 m=A.mean(0);mean=A.mean();Ac=A-m[None,:]-m[:,None]+mean;Cc=C-m[None,:]-C.mean(1)[:,None]+mean
 return Cc@solve(Ac+alpha*np.eye(len(tr)),z-z.mean(),assume_a='pos')+z.mean()

def validate(K,alpha,mode):
 out=[];preds=[]
 for tr,te,typ in splits:
  pred=fitpred(K,y,tr,te,alpha,mode);r=spearmanr(y[te],pred).statistic
  out.append(r);preds.append(pred)
 return np.mean(out[:12]),np.mean(out[12:]),preds

results=[];t=time.time()
configs=[]
for exact in [.15,.5,1.5]:
 for prop in [0,.1,.4]:
  for glob in [0,.05,.2]:
   for alpha in [.03,.15,.6,2.]:
    configs.append({'exact':exact,'pos':1.,'prop':prop,'global':glob,'alpha':alpha,'target':'log'})
for key in [k for k in components if k.startswith('rbf')]:
 for exact in [.2,.8]:
  for glob in [0,.1]:
   for alpha in [.05,.3,1.]:
    configs.append({key:1.,'exact':exact,'pos':.2 if 'True' in key else 0,'global':glob,'alpha':alpha,'target':'log'})
for cfg in configs:
 K=sum(components[k]*v for k,v in cfg.items() if k in components)
 a,b,p=validate(K,cfg['alpha'],cfg['target']);results.append((.7*a+.3*b,a,b,cfg,p))
results.sort(key=lambda r:r[0],reverse=True)
print('First grid',len(results),'elapsed',time.time()-t)
for r in results[:15]:print(r[:4])
# Compare alternative output scales, region pooling, shared substitution effects,
# and second-order epistasis kernels on only a small shortlisted family.
short=[r[3] for r in results[:5]]
for base in short:
 K0=sum(components[k]*v for k,v in base.items() if k in components)
 for mode in ['log.1','sqrt','raw','rank']:
  cfg=dict(base,target=mode);a,b,p=validate(K0,cfg['alpha'],mode);results.append((.7*a+.3*b,a,b,cfg,p))
 for key in ['region','substitution','interaction']:
  for weight in [.05,.2,.7]:
   K=K0+weight*(K0*K0/np.maximum(np.diag(K0).mean(),1e-6) if key=='interaction' else components[key])
   cfg=dict(base,**{key:weight});a,b,p=validate(K,cfg['alpha'],cfg['target']);results.append((.7*a+.3*b,a,b,cfg,p))
results.sort(key=lambda r:r[0],reverse=True)
print('All grid',len(results),'elapsed',time.time()-t)
for r in results[:25]:print(r[:4])
# Save configurations and validation predictions only for exploration. Final
# predictor will neither read this file nor preserve fitted states.
np.savez('explore_results.npz',predictions=np.array([np.concatenate(r[4]) for r in results]),configs=np.array([json.dumps(r[3]) for r in results]),metrics=np.array([r[:3] for r in results]))
json.dump([{'metric':r[0],'fb':r[1],'all':r[2],'config':r[3]} for r in results],open('explore_results.json','w'),indent=2)
