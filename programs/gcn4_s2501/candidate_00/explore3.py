import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,time
from scipy.stats import norm,rankdata,spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import proposal
threadpool_limits(1)
c=json.load(open('case.json'));s=c['support'];n=len(s);y=np.array([r['fitness'] for r in s]);o=np.array([r['mutation_order'] for r in s]);fb=np.where(o>1)[0];si=np.where(o==1)[0];f=proposal.features(s,c['reference_sequence'])
X={'phys':np.c_[f['phys'],f['basic']], 'comp':np.c_[f['comp'],f['basic']], 'global':np.c_[f['comp'][:,:20],f['basic']], 'phys_inter':np.c_[f['phys'],f['inter'],f['basic']]}
folds=[]
for seed in range(6):
 for tr,te in StratifiedKFold(4,shuffle=True,random_state=105+seed).split(fb,o[fb]>15):folds.append((seed,np.r_[si,fb[tr]],fb[te]))
def quality(p):return [round(float(np.mean([spearmanr(y[ix],z[ix]).statistic for z in p])),4) for ix in [fb,fb[o[fb]<15],fb[o[fb]>15]]]+[round(float(np.std([spearmanr(y[fb],z[fb]).statistic for z in p])),4)]
res=[];pred={};t0=time.time()
for name,x in X.items():
 for split in ['pooled','separate']:
  configs=[('en',a,l) for a in [.03,.1,.3] for l in [.2,.8,1]]+[('pls',a,0) for a in [1,2,3,4]]+[('svr',a,g) for a in [.1,1,10] for g in [.3,1,3]]
  for model,a,g in configs:
   p=np.zeros((6,n))
   for seed,tr,te in folds:
    ty=norm.ppf((rankdata(y[tr])-.5)/len(tr))
    for regime in ([False,True] if split=='separate' else [None]):
     tm=np.ones(len(tr),bool) if regime is None else (o[tr]>15)==regime;qm=np.ones(len(te),bool) if regime is None else (o[te]>15)==regime;ix=tr[tm];jx=te[qm]
     mu=x[ix].mean(0);sd=np.maximum(x[ix].std(0),1);xx=(x[ix]-mu)/sd;qq=(x[jx]-mu)/sd
     if model=='en':m=ElasticNet(alpha=a,l1_ratio=g,max_iter=2000,tol=.0001)
     if model=='pls':m=PLSRegression(n_components=a,scale=False)
     if model=='svr':m=SVR(C=a,gamma=g/max(np.mean((xx*xx).sum(1)),1),epsilon=.1)
     m.fit(xx,ty[tm]);p[seed,jx]=m.predict(qq).ravel()
   key=(name,split,model,a,g);q=quality(p);res.append((q,key));pred[str(key)]=p
  print(name,split,sorted([v for v in res if v[1][:2]==(name,split)],reverse=True)[:3],round(time.time()-t0),flush=True)
ranked=sorted(res,reverse=True);print('TOP',ranked[:20]);json.dump([{'quality':q,'config':k} for q,k in ranked],open('validation3_results.json','w'),indent=2);np.savez('exploration3_predictions.npz',**{'p'+str(i):pred[str(k)] for i,(_,k) in enumerate(ranked)})
