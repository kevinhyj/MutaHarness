import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np,time
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr,rankdata,norm
import proposal
c=json.load(open('case.json'));s=c['support'];n=len(s);y=np.array([r['fitness'] for r in s]);order=np.array([r['mutation_order'] for r in s]);feedback=np.arange(32,n)
rng=np.random.default_rng(2026)
folds=[]
for repeat in range(4):
 pp=rng.permutation(feedback)
 for va in np.array_split(pp,4): folds.append((np.setdiff1d(np.arange(n),va),va))
parts={l:proposal.kernels(s,c['reference_sequence'],l) for l in [.25,.5,.8,1.2,1.8,3.]}
trans={'raw':lambda y:y,'rank':lambda y:norm.ppf((rankdata(y)-.5)/len(y)),'asinh':lambda y:np.arcsinh(y), 'floor':lambda y:-4+np.log(np.expm1(np.maximum(y+4,.01))),'floor2':lambda y:-3.5+.6*np.log(np.expm1(np.maximum((y+3.5)/.6,.01)))}
def evaluate(config):
 l,ident,pos,glob,count,pair,noise,transform=config
 p=parts[l]; k=p[0]+ident*p[1]+pos*p[2]+glob*p[3]+count*p[4]+pair*p[5]+1
 pred=np.zeros((4,n));corr=[];dcor=[];scor=[]
 for j,(tr,va) in enumerate(folds):
  yy=trans[transform](y[tr]);a=k[np.ix_(tr,tr)]+noise*np.eye(len(tr))
  co=cho_solve(cho_factor(a,lower=True,check_finite=False),yy,check_finite=False)
  yh=k[np.ix_(va,tr)]@co;pred[j//4,va]=yh
 for pr in pred:
  corr.append(spearmanr(y[32:],pr[32:]).statistic)
  dcor.append(spearmanr(y[order==2],pr[order==2]).statistic)
  mask=(order==1)&(np.arange(n)>=32);scor.append(spearmanr(y[mask],pr[mask]).statistic)
 # Query predominantly doubles: emphasize feedback combinations.
 score=.75*np.mean(dcor)+.25*np.mean(corr)
 return score,np.mean(corr),np.mean(dcor),np.mean(scor),pred
configs=[]
for l in parts:
 for noise in [.01,.03,.1,.3,1]:
  for pair in [0,.1,.5]: configs.append((l,.2,.2,.1,.1,pair,noise,'raw'))
for _ in range(400):
 configs.append((rng.choice(list(parts)),rng.choice([0,.03,.1,.3,1]),rng.choice([0,.1,.3,1]),rng.choice([0,.03,.1,.3]),rng.choice([0,.1,.3,1]),rng.choice([0,.03,.1,.3,1]),rng.choice([.01,.03,.1,.3,1]),rng.choice(list(trans))))
results=[];t=time.time()
for cfg in configs:
 score,a,b,d,p=evaluate(cfg);results.append((score,a,b,d,cfg));
results.sort(reverse=True)
print('seconds',time.time()-t)
for r in results[:35]: print(r)
json.dump(results,open('cv_results.json','w'),indent=2)
# Model ensemble rank predictions; avoid mixing differing target units.
for num in [3,5,10,20,40,80]:
 prs=[evaluate(r[-1])[-1] for r in results[:num]]
 pred=np.mean([np.array([rankdata(p[32:]) for p in pr]) for pr in prs],0)
 print('ensemble',num,'all',np.mean([spearmanr(y[32:],p).statistic for p in pred]),'double',np.mean([spearmanr(y[32:][order[32:]==2],p[order[32:]==2]).statistic for p in pred]))
