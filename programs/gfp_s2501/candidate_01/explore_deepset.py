from model_helpers import *
from scipy.special import expit
c=json.load(open('case.json'));r=c['support'];ref=c['reference_sequence'];y=np.array([v['fitness'] for v in r])
folds=[]
for seed in [41,1729,2026]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,len(y))):folds.append((np.r_[np.arange(32),tr+32],te+32))
for tr,te in KFold(4,shuffle=True,random_state=170).split(y):folds.append((tr,te))

def damage_fit(records,Y,query,ref,alpha=.03,mode=0):
 rows=records+query; feats=[];ri=[]
 for j,r in enumerate(rows):
  ms=mutations(r,ref);F=mfeat(ms,ref)
  if mode==0:F=F[:,:37]
  elif mode==1:F=F[:,:21]
  elif mode==2:
   F=F[:,:37]
   # smooth positional covariates provide modest sequence locality sharing
   pos=np.array([p for p,a,b in ms]);G=np.exp(-.5*((pos[:,None]-np.arange(15,239,20)[None,:])/15)**2);F=np.c_[F,G]
  feats.extend(F);ri.extend([j]*len(F))
 M=np.array(feats);ri=np.array(ri);N=len(records)
 low=np.quantile(Y,.04);hi=np.quantile(Y,.95)+.12;yy=(Y-low)/(hi-low)
 dim=M.shape[1]
 prior=np.zeros(dim+1);prior[0]=3.;prior[1]=-1.5
 def fun(b):
  u=M@b[1:];cost=np.logaddexp(0,u);latent=b[0]-np.bincount(ri,weights=cost,minlength=len(rows));pp=expit(latent[:N]);return np.r_[pp-yy,np.sqrt(alpha)*(b[1:]-prior[1:])]
 def jac(b):
  u=M@b[1:];cost=np.logaddexp(0,u);latent=b[0]-np.bincount(ri,weights=cost,minlength=len(rows));pp=expit(latent[:N]);MM=np.zeros((len(rows),dim));np.add.at(MM,ri,expit(u)[:,None]*M);J=np.c_[np.ones(N),-MM[:N]]*(pp*(1-pp))[:,None];return np.r_[J,np.c_[np.zeros(dim),np.eye(dim)*np.sqrt(alpha)]]
 opt=least_squares(fun,prior,jac=jac,max_nfev=120,ftol=1e-6,xtol=1e-6)
 return opt.x[0]-np.bincount(ri,weights=np.logaddexp(0,M@opt.x[1:]),minlength=len(rows))[N:]
res={}
for mode in [0,1,2,3]:
 for alpha in [.003,.03,.3]:
  name='damage'+str(mode)+'_'+str(alpha);st=time.time();ps=[];cs=[]
  for tr,te in folds:
   p=damage_fit([r[i] for i in tr],y[tr],[r[i] for i in te],ref,alpha,mode);ps.append(p);cs.append(spearmanr(y[te],p).statistic)
  res[name]=np.concatenate(ps);print(name,round(np.mean(cs[:12]),4),round(np.mean(cs[12:]),4),'time',round(time.time()-st,2),flush=True)
np.savez('damage_predictions.npz',**res)
