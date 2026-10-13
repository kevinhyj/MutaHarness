import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata
from sklearn.model_selection import StratifiedKFold
c=json.load(open('case.json'));r=c['support'];n=len(r);y=np.array([a['fitness'] for a in r]);order=np.array([a['mutation_order'] for a in r]);data=np.load('kernel_data.npz');kernels=data['kernels'];keys=data['keys'].tolist()
raw=json.load(open('kernel_results.json'));base=json.load(open('cv_search.json'))
configs=[a for a in raw if a['target']=='raw'][:6]+[next(a for a in raw if a['target']=='rank'),next(a for a in raw if a['target']=='asinh'),base[0]]
# Round selected hyperparameters: keep only two significant figures.
for cf in configs:
 cf['weights']={k:float('%.2g'%v) for k,v in cf['weights'].items() if v>=.005};cf['alpha']=float('%.2g'%cf['alpha'])
json.dump(configs,open('chosen_configs.json','w'),indent=2)
splits=[]
for seed in [103,137,191,251,307]:
 for mode in ['all','feedback','thin']:
  ix=np.arange(32,n) if mode=='feedback' else np.arange(n)
  nf=2 if mode=='thin' else 4
  for tr,te in StratifiedKFold(nf,shuffle=True,random_state=seed).split(ix,order[ix]):
   if mode=='feedback':tr=np.r_[np.arange(32),ix[tr]];te=ix[te]
   else:tr=ix[tr];te=ix[te]
   splits.append((tr,te,mode,seed))
modelpred=[]
for cf in configs:
 pred=np.full((15,n),np.nan)
 for tr,te,mode,seed in splits:
  w=np.array([cf['weights'].get(k,0) for k in keys]);K=np.einsum('k,kij->ij',w,kernels[:,tr][:,:,tr]);Q=np.einsum('k,kij->ij',w,kernels[:,te][:,:,tr]);mean=K.mean(axis=0);grand=mean.mean();K=K-mean[:,None]-mean[None,:]+grand;Q=Q-Q.mean(axis=1,keepdims=True)-mean[None,:]+grand
  z=y[tr].copy()
  if cf['target']=='rank':z=rankdata(z)/(len(tr)+1)
  if cf['target']=='asinh':z=np.arcsinh(z)
  zm=z.mean();zs=z.std() or 1.;a=solve(K+cf['alpha']*np.eye(len(tr)),(z-zm)/zs,assume_a='pos',check_finite=False);v=(Q@a)*zs+zm
  if cf['target']=='asinh':v=np.sinh(v)
  if cf['target']=='rank':v=np.interp(v,np.arange(1,len(tr)+1)/(len(tr)+1),np.sort(y[tr]))
  jj=[103,137,191,251,307].index(seed)*3+['all','feedback','thin'].index(mode);pred[jj,te]=v
 modelpred.append(pred)
modelpred=np.array(modelpred);np.save('fresh_oof.npy',modelpred)
def report(pp,label):
 overall=[];double=[];single=[];by={m:[] for m in ['all','feedback','thin']}
 for i,v in enumerate(pp):
  ok=np.isfinite(v);ov=spearmanr(y[ok],v[ok]).statistic;overall.append(ov);by[['all','feedback','thin'][i%3]].append(ov)
  dd=ok&(order==2);ss=ok&(order==1);double.append(spearmanr(y[dd],v[dd]).statistic);single.append(spearmanr(y[ss],v[ss]).statistic)
 print(label,'overall',round(np.mean(overall),4),'double',round(np.mean(double),4),'single',round(np.mean(single),4),'regimes',{k:round(np.mean(v),4) for k,v in by.items()},flush=True)
for i,pp in enumerate(modelpred):report(pp,'model'+str(i))
for inds in [[0,1,2],[0,1,2,3,4,5],list(range(8)),[0,1,2,3,4,5,8],[0,1,2,3,4,5,6]]:
 pp=np.nanmean(modelpred[inds],axis=0);report(pp,'ensemble'+str(inds))
