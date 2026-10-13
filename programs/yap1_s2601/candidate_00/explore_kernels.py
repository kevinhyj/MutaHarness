import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,itertools,time
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
import proposal
c=json.load(open('case.json'));S=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in S]);pc=proposal.chemistry()
allms=sorted(set(m for r in S for m in proposal.mutations(r,ref)));midx={m:i for i,m in enumerate(allms)};n=len(S);m=len(allms)
C=np.zeros((n,m))
for i,r in enumerate(S):
 for mut in proposal.mutations(r,ref):C[i,midx[mut]]=1
pos=np.array([t[0] for t in allms]);orig=[t[1] for t in allms];dest=[t[2] for t in allms];L=(pos[:,None]==pos).astype(float);A=np.array([pc[a] for a in orig]);D=np.array([pc[a] for a in dest]);diff=D-A
source=np.array([[a==b for b in orig] for a in orig],float)
def dist(X,Y):return np.maximum(0,(X*X).sum(1)[:,None]+(Y*Y).sum(1)[None,:]-2*X@Y.T)
B=[C@L@C.T,C@C.T,C@(L*(diff@diff.T))@C.T,C@(diff@diff.T)@C.T,C@(source*(diff@diff.T))@C.T]
for scale in [.5,1,2,4,8]:
 k=np.exp(-dist(D,D)/(2*scale))-np.exp(-dist(D,A)/(2*scale))-np.exp(-dist(A,D)/(2*scale))+np.exp(-dist(A,A)/(2*scale))
 B.extend([C@(L*k)@C.T,C@k@C.T,C@(source*k)@C.T])
fs=[]
for seed in [11,29,47,71,103]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(32,96)):fs.append((np.r_[np.arange(32),32+tr],32+te))
# Transform outcomes within each training fold: validation labels cannot affect target.
def target(v,mode):
 if mode=='log':return np.log(v)
 if mode=='rank':return norm.ppf((rankdata(v)-.5)/len(v))
 if mode=='log1p':return np.log1p(v)
 return (np.log(v)/np.std(np.log(v)) + norm.ppf((rankdata(v)-.5)/len(v)))/2
cached={mode:[target(y[tr],mode) for tr,te in fs] for mode in ['log','rank','mix','log1p']}
def evaluate(K,noise,mode='log'):
 vals=[]
 for i,(tr,te) in enumerate(fs):
  pr=K[np.ix_(te,tr)]@cho_solve(cho_factor(K[np.ix_(tr,tr)]+noise*np.eye(len(tr)),check_finite=False),cached[mode][i],check_finite=False)
  vals.append(spearmanr(y[te],pr).statistic)
 return float(np.mean(vals))
results=[]
for loc,exact,linear,glo,rbf,share,scale,noise in itertools.product([.1,.3,1],[.05,.3],[0,.03],[.1,.3],[.1,.5,2],[0,.1],[0,2,4],[.1,.4]):
 K=10+loc*B[0]+exact*B[1]+linear*B[2]+glo*B[3]+rbf*B[5+3*scale]+share*B[7+3*scale]
 score=evaluate(K,noise);results.append((score,(loc,exact,linear,glo,rbf,share,scale,noise)))
results.sort(reverse=True);print('RBF');
for x in results[:15]:print(x)
# Global nonlinearities and calibrated target comparisons.
res2=[]
for sc,cfg in results[:15]:
 loc,exact,linear,glo,rbf,share,scale,noise=cfg
 base=loc*B[0]+exact*B[1]+linear*B[2]+glo*B[3]+rbf*B[5+3*scale]+share*B[7+3*scale]
 for poly,mode in itertools.product([0,.05,.2],['log','rank','mix','log1p']):
  K=base+poly*base*base/4+10; res2.append((evaluate(K,noise,mode),cfg,poly,mode))
res2.sort(reverse=True);print('RBF EXTENDED')
for x in res2[:20]:print(x)
json.dump({'rbf':results[:30],'extended':res2[:30]},open('validation_rbf.json','w'),indent=2)
