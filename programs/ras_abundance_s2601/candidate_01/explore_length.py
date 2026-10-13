import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,numpy as np
from explore import kernels,splits,evaluate
from proposal import mutations,properties,HYDRO

c=json.load(open('case.json'));rec=c['support'];ref=c['reference_sequence'];y=np.array([r['fitness'] for r in rec]);n=len(y)
ks=kernels(rec,ref);vocab=sorted({m for r in rec for m in mutations(r,ref)});vi={m:i for i,m in enumerate(vocab)}
B=np.zeros((n,len(vocab)))
for i,r in enumerate(rec):
 for m in mutations(r,ref):B[i,vi[m]]=1
p=np.array([m[1] for m in vocab]);dp=abs(p[:,None]-p[None,:]);d=np.array([properties(m[2])-properties(m[0]) for m in vocab]);dc=((d[:,None]-d[None,:])**2).sum(2)
env=[]
for a,pos,b in vocab:
 h=np.array([HYDRO[x]/4.5 for x in ref]);i=pos-1
 vals=[np.mean(h[max(0,i-w):min(len(h),i+w+1)]) for w in [2,4,8,16]]
 vals+=[float(pos<=9 or 41<=pos<=46 or 51<=pos<=58 or 77<=pos<=86 or 111<=pos<=119),float(18<=pos<=25 or 67<=pos<=75 or 87<=pos<=104 or 127<=pos<=138 or 151<=pos<=166),float(pos>166),float(pos>184)]
 env.append(vals)
env=np.array(env);de=((env[:,None]-env[None,:])**2).sum(2)
ks['environment']=B@(np.exp(-de)*np.exp(-dc/4))@B.T
fb=np.array([i for i,r in enumerate(rec) if r['id'].startswith('M')]);ss=splits(n,fb)
prev=json.load(open('kernel_search.json'));rng=np.random.default_rng(718)
out=[]
for i in range(850):
 cfg=dict(prev[i%20][1])
 if i>40:
  for name in list(cfg):cfg[name]*=float(10**rng.uniform(-.35,.35))
 length=float(rng.choice([3,6,10,16,24,40,64])); form=str(rng.choice(['exp','rbf']))
 kk=ks.copy();kk['local']=B@(np.exp(-dp/length) if form=='exp' else np.exp(-.5*(dp/length)**2))@B.T
 kk['localchem']=B@(np.exp(-dp/length)*np.exp(-dc/4))@B.T
 if rng.random()<.5:cfg['environment']=float(10**rng.uniform(-2,.3))
 if rng.random()<.25:cfg['region']=0;cfg['regionchem']=0
 sc=evaluate(kk,y,ss,cfg);cfg['length']=length;cfg['form']=form
 out.append((sc,cfg))
out.sort(key=lambda x:x[0][0],reverse=True);json.dump(out,open('length_search.json','w'),indent=2)
for sc,cfg in out[:12]:print(sc,cfg,flush=True)
