import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np,itertools,time
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));S=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in S]);ms=sorted(set(m for r in S for m in proposal.mutations(r,ref)));pc=proposal.chemistry();C=np.zeros((len(S),len(ms)))
for i,r in enumerate(S):
 for m in proposal.mutations(r,ref):C[i,ms.index(m)]=1
pos=np.array([m[0] for m in ms]);A=np.array([pc[m[1]] for m in ms]);D=np.array([pc[m[2]] for m in ms]);dif=D-A;L=(pos[:,None]==pos).astype(float);R=np.array([[a[1]==b[1] for b in ms] for a in ms],float)
def dist(X,Y):return np.maximum(0,(X*X).sum(1)[:,None]+(Y*Y).sum(1)[None,:]-2*X@Y.T)
k=np.exp(-dist(D,D)/16)-np.exp(-dist(D,A)/16)-np.exp(-dist(A,D)/16)+np.exp(-dist(A,A)/16)
B=[C@L@C.T,C@C.T,C@(dif@dif.T)@C.T,C@(R*(dif@dif.T))@C.T,C@(L*k)@C.T,C@(R*k)@C.T,C.sum(1)[:,None]*C.sum(1)[None,:]]
# Discovery folds used only for configuration search.
fs=[]
for seed in [11,29,47,71,103]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,96)):fs.append((np.r_[np.arange(32),32+tr],32+te))
Y=[]
for tr,te in fs:
 log=np.log(y[tr]); z=norm.ppf((rankdata(y[tr])-.5)/len(tr)); Y.append(.5*(log/(np.std(log)+1e-12)+z))
def eval(K,noise):
 sc=[]
 for (tr,te),yt in zip(fs,Y):
  pr=K[np.ix_(te,tr)]@cho_solve(cho_factor(K[np.ix_(tr,tr)]+noise*np.eye(len(tr)),check_finite=False),yt,check_finite=False)
  sc.append(spearmanr(pr,y[te]).statistic)
 return float(np.mean(sc))
results=[]
for loc,exact,glo,rf,noise,poly,cnt in itertools.product([.02,.1,.3],[.02,.1,.3],[.1,.3,1],[0,.03,.1],[.03,.1,.3],[0,.2],[0,.3]):
 K=loc*B[0]+exact*B[1]+glo*B[2]+rf*B[3]+.1*B[4]+.1*B[5]+cnt*B[6];K=K+poly*K*K/4+10
 results.append((eval(K,noise),(loc,exact,glo,rf,noise,poly,cnt)))
results.sort(reverse=True)
for x in results[:25]:print(x)
json.dump(results[:50],open('validation_refine.json','w'),indent=2)
