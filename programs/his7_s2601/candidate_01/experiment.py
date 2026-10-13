import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata,spearmanr,norm
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.model_selection import KFold
from proposal import mutations
# Statistical pooling families from the mutation positions; no structural evidence assumed.
GROUPS=[set(range(1,36)),{56,59,60,62,76,77,78,79,80,81,82,85},{66,67,73,86,87,92,93},set(range(96,123)),{137,139,140,141,142,143,144,157,159,160,161,163},{145,147,148,149,151,152,154,164,165,168,169,170},{171,172,173,175,177,178,180,195,196,198,200},{181,183,185,186,187,189,201,202,203,204,205,206,207,208,209,210,211}]
AA='ACDEFGHIKLMNPQRSTVWY'
HY=[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOL=[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
PROP={a:np.array([HY[i]/4.5,VOL[i]/170.,float(a in 'KR')-float(a in 'DE'),float(a in 'FWY'),float(a=='P'),float(a=='G'),float(a in 'STNQ'),float(a=='C')]) for i,a in enumerate(AA)}
def group(m):
 counts=[sum(p in g for p,a,b in m) for g in GROUPS]
 return int(np.argmax(counts)) if m else 0

def design(records,ref,kind='context',phys=0.,pair=0.):
 ds=[]
 for r in records:
  ms=mutations(r,ref); n=len(ms); g=group(ms)
  d={'bias':3.,'multi':2.*(n>1),'count':n/6.}
  d['region:%s'%g]=1.*(n>1)
  d['regioncount:%s'%g]=n/6.
  for p,a,b in ms:
   d['m:%s:%s'%(p,b)]=1.
   if kind in ['site','context']: d['site:%s'%p]=.35
   if kind=='context': d['cm:%s:%s'%(p,b)]=float(n>1)
   if phys:
    delta=PROP[b]-PROP[a]
    for j,v in enumerate(delta):
     d['p:%s:%s'%(p,j)]=phys*v
     d['agg:%s:%s'%(g,j)]=d.get('agg:%s:%s'%(g,j),0.)+phys*v/3.
     d['abs:%s:%s'%(g,j)]=d.get('abs:%s:%s'%(g,j),0.)+phys*abs(v)/3.
  if pair:
   for ii,(p,a,b) in enumerate(ms):
    for q,c,e in ms[ii+1:]: d['pair:%s:%s:%s:%s'%(p,b,q,e)]=pair
  ds.append(d)
 keys=sorted({k for d in ds for k in d}); ix={k:i for i,k in enumerate(keys)}
 X=np.zeros((len(ds),len(keys)))
 for i,d in enumerate(ds):
  for k,v in d.items(): X[i,ix[k]]=v
 return X

def trans(y,mode):
 if mode=='raw':return y
 if mode=='sqrt':return np.sqrt(np.maximum(y,0)+.01)
 if mode=='rank':return (rankdata(y)-.5)/len(y)
 if mode=='normal':return norm.ppf((rankdata(y)-.5)/len(y))
 if mode=='logit':return np.log((np.maximum(y,0)+.05)/(np.maximum(1.15-y,.02)))
 if mode=='clip':return np.minimum(y,.95)
 if mode=='log':return np.log(y+.1)

def metric(y,p):return spearmanr(y,p).statistic

def run():
 c=json.load(open('case.json'));rs=c['support']; ref=c['reference_sequence']; n=len(rs);y=np.array([r['fitness'] for r in rs]); m=np.array([len(mutations(r,ref)) for r in rs]);multi=np.flatnonzero(m>1);single=np.flatnonzero(m==1)
 folds=[]
 for seed in [11,47,101]:
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(multi):folds.append((np.r_[single,multi[tr]],multi[te]))
 predictions={};configs={}
 for feat in [('mut',0,0),('context',0,0),('context',.5,0),('context',1,0),('context',0,.5),('context',0,1)]:
  X=design(rs,ref,*feat);K=X@X.T
  for mode in ['raw','sqrt','rank','logit']:
   for alpha in [.1,.5,2,8,32]:
    name=str((feat,mode,alpha)); pred=np.zeros((3,n))
    for z,(tr,te) in enumerate(folds):
     yt=trans(y[tr],mode);baseline=np.median(yt[m[tr]==1]);coef=solve(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),yt-baseline,assume_a='pos');pred[z//4,te]=K[np.ix_(te,tr)]@coef+baseline
    predictions[name]=pred;configs[name]=('kernel',feat,mode,alpha)
 for feat in [('mut',0,0),('context',0,0),('context',.5,0)]:
  X=design(rs,ref,*feat)
  for mode in ['raw','rank']:
   for meth in ['et','rf','gb']:
    for depth in [1,2,3,5,None] if meth!='gb' else [1,2,3]:
     if meth=='gb' and depth is None: continue
     name=str((meth,feat,mode,depth));pred=np.zeros((3,n))
     for z,(tr,te) in enumerate(folds):
      yt=trans(y[tr],mode)
      if meth=='et':mdl=ExtraTreesRegressor(n_estimators=160,max_depth=depth,min_samples_leaf=2,max_features=1.,random_state=123,n_jobs=1)
      if meth=='rf':mdl=RandomForestRegressor(n_estimators=160,max_depth=depth,min_samples_leaf=2,max_features=.7,random_state=123,n_jobs=1)
      if meth=='gb':mdl=GradientBoostingRegressor(n_estimators=120,max_depth=depth,min_samples_leaf=3,learning_rate=.04,loss='huber',random_state=123)
      mdl.fit(X[tr],yt);pred[z//4,te]=mdl.predict(X[te])
     predictions[name]=pred;configs[name]=(meth,feat,mode,depth)
     print('done',name,round(np.mean([metric(y[multi],p[multi]) for p in pred]),4),flush=True)
 scores={k:[metric(y[multi],p[multi]) for p in v] for k,v in predictions.items()}
 order=sorted(scores,key=lambda k:np.mean(scores[k]),reverse=True)
 for k in order[:25]: print('BEST',round(np.mean(scores[k]),4),np.round(scores[k],3),k)
 np.savez('validation_predictions.npz',**predictions)
 json.dump({'configs':configs,'scores':scores,'folds':[(tr.tolist(),te.tolist()) for tr,te in folds]},open('validation_results.json','w'),indent=2)
if __name__=='__main__':run()
