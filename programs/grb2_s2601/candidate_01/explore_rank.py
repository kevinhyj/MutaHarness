import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,itertools,pickle,time,numpy as np
from scipy.stats import spearmanr
import proposal
from ranking import fit_rank
c=json.load(open('case.json'));rs=c['support'];y=np.array([r['fitness'] for r in rs]);ref=c['reference_sequence'];fs=proposal.features(rs,ref);base=[f@f.T for f in fs]+[np.ones((len(rs),len(rs)))]
WT=np.array([[float(a==b) for b in proposal.AA] for a in ref]);Kw=fs[0]@WT@(fs[0]@WT).T
sp=np.load('cv_splits.npz');splits=[(sp[f'tr{i}'],sp[f'te{i}']) for i in range(15)];res=[];start=time.time()
for wt,deg,alpha,gap,power in itertools.product((0,8),(.0,.3),(.1,1),(.0,.15),(0,1)):
 K=4*base[0]+.1*base[1]+.25*base[2]+base[4]+base[5]+wt*Kw;K+=deg*K*K/4
 ps=[];ss=[]
 for tr,te in splits:
  p=fit_rank(K[np.ix_(tr,tr)],K[np.ix_(te,tr)],y[tr],alpha,gap,power)
  ps.append(p);ss.append(spearmanr(p,y[te]).statistic)
 res.append((.75*np.mean(ss[:10])+.25*np.mean(ss[10:]),np.mean(ss[:10]),np.mean(ss[10:]),(wt,deg,alpha,gap,power),ps,ss))
res.sort(key=lambda t:-t[0])
for r in res[:20]:print(r[:4])
with open('rank_results.pkl','wb') as f:pickle.dump(res,f)
print('elapsed',time.time()-start)
