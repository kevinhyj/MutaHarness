from experiment import *
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')
c=json.load(open('case.json'));rs=c['support'];ref=c['reference_sequence'];n=len(rs);y=np.array([r['fitness'] for r in rs]);m=np.array([len(mutations(r,ref)) for r in rs]);multi=np.flatnonzero(m>1);single=np.flatnonzero(m==1)
folds=[]
for seed in [11,47,101]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(multi):folds.append((np.r_[single,multi[tr]],multi[te]))
predictions={};configs={}
for feat in [('mut',0,0),('context',0,0),('context',.5,0)]:
 X=design(rs,ref,*feat)
 for mode in ['raw','sqrt','rank','logit']:
  for l1 in [.2,.7,1.]:
   for alpha in [.001,.004,.015,.05]:
    name=str(('en',feat,mode,l1,alpha));pred=np.zeros((3,n))
    for z,(tr,te) in enumerate(folds):
     yt=trans(y[tr],mode);model=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=3000,tol=1e-5);model.fit(X[tr],yt);pred[z//4,te]=model.predict(X[te])
    predictions[name]=pred;configs[name]=('en',feat,mode,l1,alpha)
 print('elastic done',feat,flush=True)
# Stationary and polynomial kernels; effects and epistasis share information.
for feat in [('mut',0,0),('context',.5,0)]:
 X=design(rs,ref,*feat);lin=X@X.T;D=euclidean_distances(X,squared=True)
 for kernel,param in [('rbf',.03),('rbf',.1),('rbf',.3),('poly',.05),('poly',.2),('poly',1.)]:
  if kernel=='rbf':K=np.exp(-param*D)
  else:K=lin+param*(lin*lin)/8.
  for mode in ['raw','sqrt','rank','logit']:
   for alpha in ([.005,.05,.5] if kernel=='rbf' else [.1,1,10]):
    name=str((kernel,feat,mode,param,alpha));pred=np.zeros((3,n))
    for z,(tr,te) in enumerate(folds):
     yt=trans(y[tr],mode);base=np.median(yt[m[tr]==1]);coef=solve(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),yt-base,assume_a='pos');pred[z//4,te]=K[np.ix_(te,tr)]@coef+base
    predictions[name]=pred;configs[name]=(kernel,feat,mode,param,alpha)
 print('kernel done',feat,flush=True)
scores={k:[metric(y[multi],p[multi]) for p in v] for k,v in predictions.items()}
order=sorted(scores,key=lambda k:np.mean(scores[k]),reverse=True)
for k in order[:25]:print('BEST',round(np.mean(scores[k]),4),np.round(scores[k],3),k,flush=True)
np.savez('validation2_predictions.npz',**predictions)
json.dump({'configs':configs,'scores':scores},open('validation2_results.json','w'),indent=2)
