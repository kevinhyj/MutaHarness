import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);n=len(s)
f=proposal.features(s,c['reference_sequence'])
# Extra similarities for conservative substitution generalization.
L=len(c['reference_sequence']);ab=np.zeros_like(f['prop']);dest=np.zeros_like(ab);sev=np.zeros((n,L*3)); context=np.zeros((n,24))
for i,r in enumerate(s):
 for p,w,m in proposal.mutations(r,c['reference_sequence']):
  d=proposal.DESC[proposal.AA.index(m)]-proposal.DESC[proposal.AA.index(w)]
  ab[i,p*8:p*8+8]=abs(d);dest[i,p*8:p*8+8]=proposal.DESC[proposal.AA.index(m)]
  sev[i,p*3:p*3+3]=[np.linalg.norm(d)/3,float(m=='P'),float(m=='G')]
  cls=0 if w in 'AVLIMFWY' else 1 if w in 'GP' else 2
  context[i,cls*8:cls*8+8]+=d
f.update(abs=ab,dest=dest,severity=sev,context=context)
bases={k:v@v.T for k,v in f.items()}
feedback=np.array([i for i,r in enumerate(s) if not r['id'].startswith('S')]); originals=np.array([i for i,r in enumerate(s) if r['id'].startswith('S')])
splits=[]
for seed in [13,29,47]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feedback):splits.append((np.r_[originals,feedback[tr]],feedback[te]))
for tr,te in KFold(4,shuffle=True,random_state=73).split(np.arange(n)):splits.append((tr,te))
np.savez('validation_splits.npz',**{str(i):te for i,(_,te) in enumerate(splits)})

def pred(K,tr,te,alpha,target):
 k=K[np.ix_(tr,tr)];q=K[np.ix_(te,tr)];mean=k.mean(0);qm=q.mean(1);km=k.mean();k=k-mean[:,None]-mean[None,:]+km;q=q-mean[None,:]-qm[:,None]+km
 z=target[tr];return q@solve(k+np.eye(len(tr))*alpha,z-z.mean(),assume_a='pos')+z.mean()
def evaluate(K,alpha,target):
 p=np.full((4,n),np.nan);rs=[]
 for i,(tr,te) in enumerate(splits):
  z=pred(K,tr,te,alpha,target);p[i//4,te]=z;rs.append(spearmanr(y[te],z).statistic)
 r=[spearmanr(y[feedback],p[j,feedback]).statistic for j in range(3)];mixed=spearmanr(y,p[3]).statistic
 return np.mean(r)*.8+mixed*.2,r,mixed,p

rng=np.random.default_rng(612)
configs=[]
for prop in [0,.05,.2,.7]:
 for exact in [0,.3,1]:
  for a in [.1,.4,1,3]:configs.append({'site':1.,'prop':prop,'exact':exact,'order':.2,'alpha':a})
for i in range(150):
 d={'site':1.,'exact':rng.choice([0.,.1,.4,1.,2.]),'prop':rng.choice([0.,.02,.1,.4]),'abs':rng.choice([0.,.02,.1,.4]),'severity':rng.choice([0.,.1,.5]),'global':rng.choice([0.,.005,.02,.08]),'pair':rng.choice([0.,.05,.2,.5]),'aa':rng.choice([0.,.05,.2]),'context':rng.choice([0.,.01,.05,.2]),'order':.2,'alpha':rng.choice([.1,.4,1.,3.,10.])};configs.append(d)
res=[];start=time.time()
for idx,cfg in enumerate(configs):
 K=sum(v*bases[k] for k,v in cfg.items() if k!='alpha')
 for scale in ['raw','floor2','floor16','rank']:
  target=y if scale=='raw' else np.log(y+2) if scale=='floor2' else np.log(y+1.6) if scale=='floor16' else norm.ppf((rankdata(y)-.5)/n)
  score,r,m,p=evaluate(K,cfg['alpha'],target)
  res.append({'score':score,'feedback':r,'mixed':m,'cfg':cfg,'scale':scale});
 if idx%50==0:print(idx,round(time.time()-start,2),flush=True)
res.sort(key=lambda r:-r['score']);json.dump(res,open('validation_grid.json','w'),indent=2)
for r in res[:15]:print(r)
