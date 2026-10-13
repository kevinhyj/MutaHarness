import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from scipy.stats import spearmanr,rankdata,norm
from sklearn.model_selection import KFold
from explore import kernels,predict,region
from proposal import mutations,properties,HYDRO

c=json.load(open('case.json'));ref=c['reference_sequence'];rec=c['support'];n=len(rec);y=np.array([r['fitness'] for r in rec]);ks=kernels(rec,ref)
vocab=sorted({m for r in rec for m in mutations(r,ref)});vi={m:i for i,m in enumerate(vocab)};B=np.zeros((n,len(vocab)))
for i,r in enumerate(rec):
 for m in mutations(r,ref):B[i,vi[m]]=1
p=np.array([m[1] for m in vocab]);dp=abs(p[:,None]-p[None,:]);d=np.array([properties(m[2])-properties(m[0]) for m in vocab]);dc=((d[:,None]-d[None,:])**2).sum(2)
env=[]
for a,pos,b in vocab:
 h=np.array([HYDRO[x]/4.5 for x in ref]);i=pos-1
 env.append([np.mean(h[max(0,i-w):min(len(h),i+w+1)]) for w in [2,4,8,16]]+[float(pos<=9 or 41<=pos<=46 or 51<=pos<=58 or 77<=pos<=86 or 111<=pos<=119),float(18<=pos<=25 or 67<=pos<=75 or 87<=pos<=104 or 127<=pos<=138 or 151<=pos<=166),float(pos>166),float(pos>184)])
env=np.array(env);de=((env[:,None]-env[None,:])**2).sum(2);ks['environment']=B@(np.exp(-de)*np.exp(-dc/4))@B.T

def kval(cfg):
 kk=ks.copy();length=cfg.get('length',12)
 if 'length' in cfg:
  kk['local']=B@(np.exp(-dp/length) if cfg['form']=='exp' else np.exp(-.5*(dp/length)**2))@B.T
  kk['localchem']=B@(np.exp(-dp/length)*np.exp(-dc/4))@B.T
 K=np.ones((n,n))*cfg.get('bias',1)
 for name,w in cfg.items():
  if name in kk:K+=w*kk[name]
 if cfg.get('poly',0):K+=cfg['poly']*(ks['exact']+.3*ks['site']+.1*ks['localchem'])**2
 return K

old=json.load(open('kernel_search.json'));new=json.load(open('length_search.json'))
fb=np.array([i for i,r in enumerate(rec) if r['id'].startswith('M')]);ss=[]
for seed in [101,211,307,409]:
 for _,te in KFold(4,shuffle=True,random_state=seed).split(fb):
  test=fb[te];ss.append((np.setdiff1d(np.arange(n),test),test,'feedback'))
for seed in [113,223,317]:
 for tr,te in KFold(4,shuffle=True,random_state=seed).split(y):ss.append((tr,te,'all'))

def metric(ps):
 vals={'feedback':[],'all':[]}
 for (_,te,k),p in zip(ss,ps):vals[k].append(spearmanr(y[te],p).statistic)
 f=np.mean(vals['feedback']);a=np.mean(vals['all']);return [.7*f+.3*a,f,a]

psets={}
for label,pool in [('old',old),('new',new)]:
 for top in [1,3,5,10,20]:
  for transform in ['raw','rank','normal']:
   ps=[]
   for tr,te,_ in ss:
    yy=y.copy()
    if transform=='rank':yy[tr]=(rankdata(y[tr])-.5)/len(tr)
    if transform=='normal':yy[tr]=norm.ppf((rankdata(y[tr])-.5)/len(tr))
    ps.append(np.mean([predict(kval(cfg),yy,tr,te,cfg['noise']) for _,cfg in pool[:top]],axis=0))
   key=f'{label}_{top}_{transform}';psets[key]=ps;print(key,metric(ps),flush=True)
for w in [0,.25,.5,.75,1]:
 pp=[w*a+(1-w)*b for a,b in zip(psets['new_10_raw'],psets['old_5_raw'])];print('blend_new_old',w,metric(pp),flush=True)

# Nested validation repeats model selection inside each outer training fold.
# Candidate pool was developed on this data; this is a sensitivity audit, not
# an independent experimental result.
pool=[cfg for _,cfg in old[:30]]+[cfg for _,cfg in new[:30]]
mat=[kval(cfg) for cfg in pool];outer=[]
for seed in [607,811]:
 for _,te in KFold(4,shuffle=True,random_state=seed).split(fb):
  test=fb[te];tr=np.setdiff1d(np.arange(n),test);trainfb=np.intersect1d(tr,fb)
  scores=[]
  for cfg,K in zip(pool,mat):
   sc=[]
   for _,it in KFold(3,shuffle=True,random_state=seed+1).split(trainfb):
    va=trainfb[it];tt=np.setdiff1d(tr,va);pr=predict(K,y,tt,va,cfg['noise']);sc.append(spearmanr(y[va],pr).statistic)
   scores.append(np.mean(sc))
  best=np.argsort(scores)[-10:];pr=np.mean([predict(mat[j],y,tr,test,pool[j]['noise']) for j in best],axis=0)
  outer.append(float(spearmanr(y[test],pr).statistic))
print('nested_feedback',np.mean(outer),outer,flush=True)
