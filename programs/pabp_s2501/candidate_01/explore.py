import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import proposal
threadpool_limits(1)
c=json.load(open('case.json')); ref=c['reference_sequence']; s=c['support']; n=len(s)
y=np.array([r['fitness'] for r in s]); ms=[proposal.mutations(r,ref) for r in s]
pos,exact,chem,reg,glob=proposal.features(s,ref)
# Separate positional chemistry into signed/absolute and baseline identity features.
chem=chem.reshape(n,len(ref),21);reg=reg.reshape(n,12,21)
kp=pos@pos.T;ke=exact@exact.T
kc=np.einsum('npd,mpd->nm',chem[:,:,1:],chem[:,:,1:])
kr=np.einsum('npd,mpd->nm',reg,reg)
kg=glob@glob.T
kernels=[kp,ke,kc,kr,kg]
for width in [1.5,3,6,12]:
 dist=np.arange(len(ref))[:,None]-np.arange(len(ref))[None,:]
 spatial=np.exp(-.5*(dist/width)**2)
 kernels.append(pos@spatial@pos.T)
 kernels.append(np.einsum('npd,pq,mqd->nm',chem[:,:,1:],spatial,chem[:,:,1:],optimize=True))
kernels=np.array(kernels)
# 4 independent feedback cross-validations, with originals always retained.
feedback=np.array([i for i,r in enumerate(s) if r['id'].startswith('M')]);orig=np.setdiff1d(np.arange(n),feedback)
splits=[]
for seed in [13,37,71,101]:
 for tr,va in KFold(4,shuffle=True,random_state=seed).split(feedback):splits.append((np.r_[orig,feedback[tr]],feedback[va]))
# General subset validation, including withheld single mutants.
for tr,va in KFold(5,shuffle=True,random_state=211).split(y):splits.append((tr,va))
def cv(k,target,alpha):
 out=np.zeros((5,n))
 for f,(tr,va) in enumerate(splits):
  mean=target[tr].mean(); coef=solve(k[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),target[tr]-mean,assume_a='pos')
  out[f//4 if f<16 else 4,va]=k[np.ix_(va,tr)]@coef+mean
 scores=[spearmanr(out[j,feedback],y[feedback]).statistic for j in range(4)]
 allscore=spearmanr(out[4],y).statistic
 return np.mean(scores),np.std(scores),allscore,out
rng=np.random.default_rng(1729)
configs=[]
for transform in ['raw','log05','log15','log01','rank']:
 for alpha in [.1,.3,1,3,10]:
  for mode in range(4):
   w=np.array([1.,.5,.1,.05,.02]+[0.]*8)
   if mode==1:w[5:7]=[.5,.05]
   if mode==2:w[7:9]=[1,.1]
   if mode==3:w[9:11]=[1,.1]
   configs.append((transform,alpha,w))
for z in range(450):
 w=np.zeros(13);w[0]=1;w[1:5]=10**rng.uniform([-1.5,-2.5,-3,-3], [.7,0,0,-.4])
 j=int(rng.integers(4));w[5+2*j:7+2*j]=10**rng.uniform([-1.5,-3],[.7,-.3])
 if rng.random()<.25:w[0]=0
 configs.append((rng.choice(['raw','log05','log15','log01','rank']),10**rng.uniform(-1.3,1.3),w))
results=[]; start=time.time()
for i,(tf,alpha,w) in enumerate(configs):
 target={'raw':y,'log05':np.log(y+.05),'log15':np.log(y+.15),'log01':np.log(y+.01),'rank':(rankdata(y)-.5)/n}[tf]
 k=np.einsum('k,kij->ij',w,kernels)
 sc,sd,allsc,out=cv(k,target,alpha)
 merit=.8*sc+.2*allsc-.1*sd
 results.append((merit,sc,sd,allsc,tf,float(alpha),w.tolist(),out.tolist()))
 if i%100==0:print(i,'elapsed',round(time.time()-start,1),flush=True)
results.sort(reverse=True,key=lambda v:v[0])
json.dump(results,open('cv_results.json','w'))
for r in results[:15]: print([round(x,4) if isinstance(x,float) else x for x in r[:7]])
