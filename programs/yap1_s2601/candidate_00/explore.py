import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json, time, itertools
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json')); S=c['support']; y=np.array([r['fitness'] for r in S]); ref=c['reference_sequence']
ms=[m for r in S+c['query'] for m in proposal.mutations(r,ref)]
pos=sorted({m[0] for m in ms}); al=sorted(set(ms)); blocks=proposal.design(S,ref,pos,al)
# Generalized physicochemical sharing: residue identity across sites and same-reference-AA groups.
pc=proposal.chemistry(); globaa=np.zeros((len(S),40)); refprop=np.zeros((len(S),20*10)); group=np.zeros((len(S),6*10))
# Approximate sequence-derived WW elements, an explicit hypothesis only.
def region(p):
 if p<=174:return 0
 if p<=181:return 1
 if p<=185:return 2
 if p<=191:return 3
 if p<=195:return 4
 return 5
for i,r in enumerate(S):
 for p,a,b in proposal.mutations(r,ref):
  globaa[i,proposal.AA.index(b)]+=1; globaa[i,20+proposal.AA.index(a)]+=1
  j=proposal.AA.index(a)*10; refprop[i,j:j+10]+=pc[b]-pc[a]
  j=region(p)*10; group[i,j:j+10]+=pc[b]-pc[a]
B=[x@x.T for x in [*blocks[:4],globaa,refprop,group,blocks[4]]]
# Validation: original singles remain in training, independently sampled feedback is held out.
folds=[]
for seed in [11,29,47,71,103]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,96)):
  folds.append((np.r_[np.arange(32),32+tr],32+te))
foldall=[]
for seed in [17,83]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(y):foldall.append((tr,te))
np.savez('design_explore.npz',**{'B'+str(i):b for i,b in enumerate(B)},y=y)
def evaluate(K,target,fs=folds):
 vals=[]; preds=[]; truth=[]
 for tr,te in fs:
  try: pr=K[np.ix_(te,tr)]@cho_solve(cho_factor(K[np.ix_(tr,tr)]+noise*np.eye(len(tr)),check_finite=False),target[tr],check_finite=False)
  except Exception:return -1,-1
  vals.append(spearmanr(y[te],pr).statistic);preds.extend(pr);truth.extend(y[te])
 return np.mean(vals),spearmanr(truth,preds).statistic
res=[];t=time.time()
for loc,exact,prop,glo,noise in itertools.product([.3,1,3],[.1,.5,2],[0,.05,.2,.7],[0,.05,.2],[.05,.2,.8,3]):
 K=10+loc*B[0]+exact*B[1]+prop*B[2]+glo*B[3]
 sc,po=evaluate(K,np.log(y));res.append((sc,po,(loc,exact,prop,glo,noise)))
res.sort(reverse=True);print('elapsed',time.time()-t)
for x in res[:20]: print(x)
# A limited grid of shared chemistry, compare raw vs transformed targets and nonlinear effects.
best=[]
for base in res[:5]:
 loc,exact,prop,glo,noise=base[2]
 for aa,rf,gr,poly in itertools.product([0,.2],[0,.1,.4],[0,.1,.4],[0,.05,.2]):
  K=loc*B[0]+exact*B[1]+prop*B[2]+glo*B[3]+aa*B[4]+rf*B[5]+gr*B[6]
  K=K+poly*K*K/4+10
  sc,po=evaluate(K,np.log(y));best.append((sc,po,(*base[2],aa,rf,gr,poly)))
best.sort(reverse=True)
print('EXTENDED')
for x in best[:20]:print(x)
json.dump({'basic':res[:30],'extended':best[:30]},open('validation_search.json','w'),indent=2)
print('TARGETS AND ALL-SUPPORT HOLDOUT')
for z in best[:4]:
 loc,exact,prop,glo,noise,aa,rf,gr,poly=z[2]
 K=loc*B[0]+exact*B[1]+prop*B[2]+glo*B[3]+aa*B[4]+rf*B[5]+gr*B[6];K=K+poly*K*K/4+10
 for name,target in [('log',np.log(y)),('log1p',np.log1p(y)),('raw',y),('rank',norm.ppf((rankdata(y)-.5)/len(y)))]:
  print(z[2],name,evaluate(K,target),evaluate(K,target,foldall))
