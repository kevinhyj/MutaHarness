from model_helpers import *
from scipy.linalg import eigh
c=json.load(open('case.json'));r=c['support'];ref=c['reference_sequence'];y=np.array([v['fitness'] for v in r]);base=features(r,ref);extra=extra_features(r,ref)
folds=[]
for seed in [41,1729,2026]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,len(y))):folds.append((np.r_[np.arange(32),tr+32],te+32))
for tr,te in KFold(4,shuffle=True,random_state=170).split(y):folds.append((tr,te))
results={}
def run(name,fn):
 st=time.time();o=[];cors=[]
 for tr,te in folds:
  p=fn(tr,te);o.append(p);cors.append(spearmanr(y[te],p).statistic)
 results[name]=np.concatenate(o)
 print(name,round(np.nanmean(cors[:12]),4),round(np.nanmean(cors[12:]),4),round(np.nanstd(cors[:12]),3),'time',round(time.time()-st,2),flush=True)

def ge_fit(X,Y,Z,alpha=.3):
 # Logistic fluorescence expectation with learned latent additive burden.
 low=np.quantile(Y,.04);hi=np.quantile(Y,.95)+.12
 yy=(Y-low)/(hi-low)
 # Remove constant columns and use compact SVD to optimize only supported dimensions.
 xm=X.mean(0);xc=X-xm;zc=Z-xm
 u,s,vt=np.linalg.svd(xc,full_matrices=False);keep=s>1e-8
 xt=u[:,keep]*s[keep];zt=zc@vt[keep].T
 init=Ridge(alpha=alpha*10).fit(xt,logit(np.clip(yy,.03,.97)))
 b=np.r_[init.intercept_,init.coef_]
 def fun(b):
  return np.r_[expit(b[0]+xt@b[1:])-yy,np.sqrt(alpha)*b[1:]]
 def jac(b):
  p=expit(b[0]+xt@b[1:]);g=p*(1-p)
  return np.r_[g[:,None]*np.c_[np.ones(len(Y)),xt],np.c_[np.zeros(len(b)-1),np.sqrt(alpha)*np.eye(len(b)-1)]]
 opt=least_squares(fun,b,jac=jac,max_nfev=70,ftol=1e-6,xtol=1e-6,gtol=1e-6)
 return opt.x[0]+zt@opt.x[1:]

# Raw counts preserve additive mutation effect interpretation; no arbitrary standardization.
for mode in ['sum','sum_sparse','summax','all']:
 for alpha in [.03,.3,3.]:
  def fn(tr,te,mode=mode,alpha=alpha):
   if mode=='sum':X=extra[:,:78]
   elif mode=='summax':X=extra
   elif mode=='all':X=np.c_[base,extra]/2
   else:
    ds=tokens([r[i] for i in tr],ref);e,p,t=ds(r);X=np.c_[extra[:,:78]/2,e*2,p,t/2]
   return ge_fit(X[tr],y[tr],X[te],alpha)
  run(mode+'_ge'+str(alpha),fn)
run('count',lambda tr,te:-extra[te,0])
np.savez('latent_predictions.npz',**results)
