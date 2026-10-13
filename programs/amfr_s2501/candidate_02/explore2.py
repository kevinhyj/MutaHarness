from exploration_utils import *
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
# BLOSUM62 in its conventional amino-acid order, a fixed biochemical similarity prior.
BORDER='ARNDCQEGHILKMFPSTWYV'
B=np.array([
[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4]])
ii=[BORDER.index(a) for a in proposal.AA]; B=B[np.ix_(ii,ii)]

def akernel(A,difference=False):
 K=np.zeros((n,n))
 for j,a in enumerate(ref):
  ids=np.where(mut[:,j]>=0)[0]; aa=mut[ids,j];wt=proposal.AA.index(a)
  v=A[np.ix_(aa,aa)]
  if difference:v=v-A[aa,wt][:,None]-A[wt,aa][None,:]+A[wt,wt]
  K[np.ix_(ids,ids)]+=v
 return K
for temp in [1,2,4,8]:
 A=np.exp((B-.5*(np.diag(B)[:,None]+np.diag(B)[None,:]))/temp)
 for diff in [False,True]:kernels[f'blosum{temp}-{diff}']=akernel(A,diff)
# More physically focused property kernels, including a helicity-only descriptor.
for name,weights in [('main',[2,1,2,1,.2,2,2,.2,.2,0]),('hydro',[3,1,.5,.5,.2,1,1,.2,.2,0]),('helix',[1,.5,3,.5,0,3,2,0,0,0])]:
 ds=((props[:,None]-props[None,:])**2*np.array(weights)).sum(2)/sum(weights)
 for width in [.5,1,2]:
  A=np.exp(-ds/(2*width**2))
  kernels[name+str(width)]=akernel(A)
# Whole-sequence smooth kernels allow nonadditive effects.
for v in [.1,.3,1]:
 k=kernels['phys']*.5+kernels['one']*.2+kernels['global']*v
 dist=np.diag(k)[:,None]+np.diag(k)[None,:]-2*k
 for gamma in [.1,.3,1]:kernels[f'seq{v}-{gamma}']=np.exp(-dist*gamma)
configs=[]
for aa in [k for k in kernels if k not in ['one','pos','phys','global','transfer']]:
 for pos,phys,glob,inter in [(.5,.25,.25,.1),(.5,.25,.25,0),(.5,0,.25,.1),(0,.25,.25,.1),(.5,.25,0,0)]:
  for alpha in [.1,.3,1,3]:
   configs.append(dict(aa=aa,pos=pos,phys=phys,transfer=0,global_=glob,inter=inter,alpha=alpha,target='rank'))
res=[];t=time.time()
for cfg in configs:
 v=evaluate(cfg);res.append((*v[:3],cfg))
res.sort(reverse=True,key=lambda x:x[0]);json.dump(res,open('exploration_second.json','w'),indent=2)
print('Kernel time',time.time()-t,flush=True)
for r in res[:25]:print(r,flush=True)
# SVR fits may be more robust to individual inconsistent double-mutant observations.
svrs=[]
for cfg in [res[0][3],dict(aa='rbf2',pos=.5,phys=.25,transfer=0,global_=.25,inter=.1,alpha=1,target='rank')]:
 for C,eps,tgt in itertools.product([.3,1,3,10],[.05,.2],['raw','rank']):
  K=matrix(cfg);out=[]
  for kind,tr,te in splits:
   model=SVR(kernel='precomputed',C=C,epsilon=eps).fit(K[np.ix_(tr,tr)],target(tgt,tr))
   pred=model.predict(K[np.ix_(te,tr)]);out.append((kind,spearmanr(pred,y[te]).statistic))
  a=np.mean([v for k,v in out if k=='all']);f=np.mean([v for k,v in out if k=='feedback'])
  svrs.append((.35*a+.65*f,a,f,C,eps,tgt,cfg))
svrs.sort(reverse=True,key=lambda x:x[0]);json.dump(svrs,open('exploration_svr.json','w'),indent=2)
print('SVR',flush=True)
for r in svrs[:10]:print(r,flush=True)
