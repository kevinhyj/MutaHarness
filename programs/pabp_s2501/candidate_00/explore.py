import json,time,itertools
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import rankdata,spearmanr
from sklearn.model_selection import KFold
import proposal_initial as p
c=json.load(open('case.json')); s=c['support']; y=np.array([r['fitness'] for r in s]);f=p.features(s,c['reference_sequence'])
P,E,D,H=f
# All transformations are refit using the training fold only.
kernels={'pos':P@P.T,'exact':E@E.T,'chem':D@D.T,'global':H@H.T}
idx=np.arange(len(s));feedback=np.array([i for i,r in enumerate(s) if r['mutation_order']>1]);singles=np.setdiff1d(idx,feedback)
splits=[]
for seed in [31,71,117]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(feedback): splits.append((np.r_[singles,feedback[tr]],feedback[te]))
for seed in [21,67]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(idx): splits.append((tr,te))
# Optional topology is an assumed sequence annotation, not observed structure.
def group(pos):
 for i,(lo,hi) in enumerate([(126,133),(134,139),(140,149),(150,151),(152,158),(159,165),(166,173),(174,178),(179,190),(191,192),(193,200)]):
  if lo<=pos<=hi:return i
 return -1
loc=np.arange(len(c['reference_sequence']))+1
G=np.array([group(x) for x in loc]); gm=((G[:,None]==G[None,:]) & (G[:,None]>=0)).astype(float)
dist=np.abs(loc[:,None]-loc[None,:])
for width in [1,2,4,8,16]:
 smooth=np.exp(-dist**2/(2*width**2))
 kernels['smooth'+str(width)]=P@smooth@P.T
 # Local chemistry borrowing across neighboring sites.
 kernels['dsm'+str(width)]=sum(D[:,a::8]@smooth@D[:,a::8].T for a in range(8))
kernels['group']=P@gm@P.T
kernels['dgroup']=sum(D[:,a::8]@gm@D[:,a::8].T for a in range(8))
# Mutation destination/origin shared across positions.
A=np.zeros((len(s),40))
for i,r in enumerate(s):
 for pos,a,b in p.mutations(r,c['reference_sequence']):A[i,p.INDEX[b]]+=1;A[i,20+p.INDEX[a]]+=1
kernels['aa']=A@A.T
np.savez('explore_kernels.npz',**kernels)

def evaluate(K,alpha,mode,offset=.1,return_pred=False):
 scores=[];oof=np.full((len(splits),len(s)),np.nan)
 for z,(tr,te) in enumerate(splits):
  yy=y[tr].copy()
  if mode=='log':yy=np.log(np.maximum(yy,0)+offset)
  elif mode=='rank':yy=(rankdata(yy)-.5)/len(yy)
  elif mode=='sqrt':yy=np.sqrt(np.maximum(yy,0))
  elif mode=='asinh':yy=np.arcsinh(yy/.2)
  m=yy.mean();wt=cho_solve(cho_factor(K[np.ix_(tr,tr)]+alpha*np.eye(len(tr)),check_finite=False),yy-m,check_finite=False)
  pred=m+K[np.ix_(te,tr)]@wt
  scores.append(spearmanr(y[te],pred).statistic);oof[z,te]=pred
 scores=np.array(scores)
 if return_pred:return scores,oof
 return round(scores[:12].mean(),5),round(scores[12:].mean(),5),round(scores.std(),5)
results=[]
for chem,exact,glob,smooth,alpha,mode in itertools.product([0,.3,1.5],[0,.5,2],[0,.05],[0,.2],[.15,.6,2],['raw','log','rank']):
 cfg=dict(chem=chem,exact=exact,glob=glob,smooth=smooth,alpha=alpha,mode=mode)
 K=kernels['pos']+chem*kernels['chem']+exact*kernels['exact']+glob*kernels['global']+smooth*kernels['smooth2']
 scores=evaluate(K,alpha,mode)
 results.append((.7*scores[0]+.3*scores[1],scores,cfg))
results.sort(reverse=True,key=lambda x:x[0]);json.dump(results,open('explore_results.json','w'),indent=2)
for r in results[:25]: print(r,flush=True)
