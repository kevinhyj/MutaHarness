exec(open('refine.py').read().split('\nresults=[];start=time.time()')[0])
from scipy.optimize import minimize
from scipy.special import expit
z=np.load('exploration_kernels.npz');kernels={k:z[k] for k in z.files}

def rankfit(K,tr,va,reg,mode):
 kt=K[np.ix_(tr,tr)];e,v=np.linalg.eigh(kt);positive=e>1e-8;Z=v[:,positive]*np.sqrt(e[positive]);q=K[np.ix_(va,tr)]@v[:,positive]/np.sqrt(e[positive]);a,b=np.triu_indices(len(tr),1);sgn=np.sign(y[tr][a]-y[tr][b]);F=(Z[a]-Z[b])*sgn[:,None]
 if mode=='all':w=np.ones(len(a))
 elif mode=='delta':w=np.minimum(np.abs(y[tr][a]-y[tr][b]),1.)
 elif mode=='ordinal':
  r=rankdata(y[tr]);w=np.abs(r[a]-r[b])/len(tr)
 else:w=(np.abs(y[tr][a]-y[tr][b])>.25).astype(float)
 w/=w.sum()
 def obj(beta):
  marg=F@beta;p=expit(-marg)*w
  return np.logaddexp(0,-marg)@w+reg*(beta@beta)/2,-F.T@p+reg*beta
 opt=minimize(obj,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-8})
 return q@opt.x

basecfgs=[{'R3.0':1.,'S':1.,'G':1.,'pairS':.3},{'P':1.,'S':1.,'G':1.,'pairS':.3},{'R3.0':1.,'B':.5,'S':1.,'G':1.,'pairS':.3}]
results=[]
for cfg in basecfgs:
 K=sum(kernels[k]*v for k,v in cfg.items())
 for reg in [.003,.01,.03,.1]:
  for mode in ['all','delta','ordinal']:
   vals=[];ps=[];yy=[]
   for tr,va in splits[:8]:
    p=rankfit(K,tr,va,reg,mode);vals.append(spearmanr(p,y[va]).statistic);ps.extend(p);yy.extend(y[va])
   results.append((np.mean(vals),cfg,reg,mode,spearmanr(ps,yy).statistic))
print('RANKING',sorted(results,key=lambda a:a[0],reverse=True)[:15])
json.dump(results,open('cv_pairrank.json','w'))
