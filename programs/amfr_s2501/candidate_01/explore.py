import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time,itertools
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json')); rows=c['support']; ref=c['reference_sequence'];y=np.array([r['fitness'] for r in rows]);n=len(y)
one,prop,site,glob=proposal.encode(rows,ref)
one=one.reshape(n,len(ref),20);prop=prop.reshape(n,len(ref),9)
kernels={}
kernels['O']=np.einsum('nla,mla->nm',one,one)
kernels['P']=np.einsum('nla,mla->nm',prop,prop)/9
kernels['S']=site@site.T
kernels['G']=glob@glob.T/9
# Cross-position chemical covariates provide cautious transfer to unobserved sites.
kernels['U']=site.sum(1)[:,None]@site.sum(1)[None,:]
D=proposal.descriptors(); dist=((D[:,None]-D[None,:])**2).mean(2)
for width in [0.3,1.,3.]:
 sim=np.exp(-dist/width);K=np.zeros((n,n));partials=[]
 for j,w in enumerate(ref):
  b=AAW=proposal.AA.index(w)
  center=sim-sim[:,b,None]-sim[None,b,:]+sim[b,b]
  part=one[:,j]@center@one[:,j].T
  partials.append(part)
  K+=part
 kernels['R'+str(width)]=K
 if width==1.:rpartials=partials
s_pair=(kernels['S']**2-kernels['S'])/2
kernels['pairS']=s_pair
# kernel for smooth two-site interactions, excluding same-site square.
partials=[r+0.5*(site[:,j,None]@site[None,:,j]) for j,r in enumerate(rpartials)]
add=sum(partials); kernels['pairR']=(add**2-sum(p*p for p in partials))/2
splits=[]
# Hold out independently sampled feedback; original 32 singles remain training.
for seed in range(4):
 rng=np.random.default_rng(600+seed)
 for train,v in KFold(4,shuffle=True,random_state=600+seed).split(np.arange(32,n)):
  splits.append((np.r_[np.arange(32),32+train],32+v))
# Less data availability: half of feedback kept, repeated 8 times.
for seed in range(8):
 ix=np.random.default_rng(1600+seed).permutation(np.arange(32,n));splits.append((np.r_[np.arange(32),ix[:32]],ix[32:]))
all_splits=list(KFold(5,shuffle=True,random_state=77).split(np.arange(n)))

def metric(K,target,reg,splits=splits):
 allp=[]; ally=[]; doublep=[]; doubley=[]; vals=[]
 for tr,va in splits:
  mean=target[tr].mean();a=cho_solve(cho_factor(K[np.ix_(tr,tr)]+reg*np.eye(len(tr))),target[tr]-mean)
  p=K[np.ix_(va,tr)]@a+mean
  vals.append(spearmanr(p,y[va]).statistic)
  allp.extend(p);ally.extend(y[va]);m=np.array([rows[i]['mutation_order']==2 for i in va]);doublep.extend(p[m]);doubley.extend(y[va][m])
 return np.mean(vals),spearmanr(allp,ally).statistic,spearmanr(doublep,doubley).statistic

targets={'raw':y,'rank':norm.ppf((rankdata(y)-.5)/n)}
for floor,tau in [(-3.5,.5),(-3.4,.8),(-3.3,1.),(-4.,1.),(-3.5,1.5)]:
 targets[f'floor{floor}t{tau}']=floor+tau*np.log(np.expm1(np.maximum(y-floor,.02)/tau))
results=[];start=time.time()
configs=[]
for aa in ['O','P','R0.3','R1.0','R3.0']:
 for s in [0.1,0.5,2.]:
  for g in [0.,0.1,0.5]:
   for rg in [0.03,0.1,0.3,1.,3.]:
    cfg={aa:1.,'S':s,'G':g}
    K=sum(kernels[k]*v for k,v in cfg.items())
    for t in ['raw','rank','floor-3.5t0.5','floor-3.4t0.8']:
     met=metric(K,targets[t],rg);results.append((met,cfg,rg,t))
for met,cfg,rg,t in sorted(results,key=lambda a:a[0][0],reverse=True)[:15]:print(met,cfg,rg,t)
json.dump([{'metric':list(met),'config':cfg,'reg':rg,'target':t} for met,cfg,rg,t in sorted(results,key=lambda a:a[0][0],reverse=True)],open('cv_initial.json','w'),indent=2)
print('elapsed',time.time()-start)
np.savez('exploration_kernels.npz',**kernels)
