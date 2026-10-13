import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from proposal import mutations,properties,features,AA

def region(p):
    # Broad canonical Ras secondary-structure hypotheses, not measured structure.
    bounds=[9,17,25,40,46,50,58,66,75,86,104,110,119,126,138,150,166,184,188]
    return sum(p>b for b in bounds)

def kernels(records,ref):
    mm=[mutations(r,ref) for r in records]
    vocab=sorted({m for ms in mm for m in ms});vi={m:i for i,m in enumerate(vocab)}
    B=np.zeros((len(mm),len(vocab)))
    for i,ms in enumerate(mm):
        for m in ms:B[i,vi[m]]=1
    p=np.array([m[1] for m in vocab]);r=np.array([region(x) for x in p])
    wt=np.array([properties(m[0]) for m in vocab]);mt=np.array([properties(m[2]) for m in vocab]);d=mt-wt
    dist=lambda x:np.maximum(0,((x[:,None]-x[None,:])**2).sum(2))
    dc=dist(d);dm=dist(mt)
    samep=(p[:,None]==p[None,:]);samer=(r[:,None]==r[None,:]);dp=abs(p[:,None]-p[None,:])
    C=features(records,ref)[2]
    C=C[:,:55];C[:,0]=0
    result={
      'exact':B@B.T,'site':B@samep@B.T,
      'sitechem':B@(samep*np.exp(-dm/3))@B.T,
      'localchem':B@(np.exp(-dp/10)*np.exp(-dc/4))@B.T,
      'regionchem':B@(samer*np.exp(-dc/4))@B.T,
      'globalchem':B@np.exp(-dc/4)@B.T,
      'chemlinear':C@C.T/12,
      'local':B@np.exp(-dp/12)@B.T,
      'region':B@samer@B.T,
      'count':B.sum(1)[:,None]*B.sum(1)[None,:],
    }
    return result

def splits(n,feedback):
    out=[]
    for seed in [13,37,71]:
        for tr,te in KFold(4,shuffle=True,random_state=seed).split(feedback):
            test=feedback[te]; train=np.setdiff1d(np.arange(n),test)
            out.append((train,test,'feedback'))
    for seed in [19,43]:
        for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(n)):
            out.append((tr,te,'all'))
    return out

def predict(K,y,tr,te,noise):
    return K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+np.eye(len(tr))*noise,y[tr],assume_a='pos')

def evaluate(ks,y,ss,cfg,ret=False):
    K=np.ones((len(y),len(y)))*cfg.get('bias',1)
    for name,w in cfg.items():
        if name in ks:K+=w*ks[name]
    if cfg.get('poly',0):
        z=ks['exact']+.3*ks['site']+.1*ks['localchem']
        K+=cfg['poly']*z*z
    vals={'feedback':[],'all':[]};pred=[]
    for tr,te,kind in ss:
        pp=predict(K,y,tr,te,cfg['noise']);vals[kind].append(spearmanr(y[te],pp).statistic);pred.append(pp)
    fb=np.mean(vals['feedback']);al=np.mean(vals['all'])
    return ((.7*fb+.3*al,fb,al),pred) if ret else (.7*fb+.3*al,fb,al)

if __name__=='__main__':
    c=json.load(open('case.json'));rec=c['support'];y=np.array([r['fitness'] for r in rec]);ks=kernels(rec,c['reference_sequence'])
    fb=np.array([i for i,r in enumerate(rec) if r['id'].startswith('M')]);ss=splits(len(y),fb)
    rng=np.random.default_rng(20261013);configs=[]
    for noise in [.1,.3,1,3]:
        for chem in [0,.1,.4,1]:
            configs.append(dict(exact=1,site=.3,chemlinear=chem,noise=noise))
    for i in range(450):
        cfg={'exact':1.,'noise':float(10**rng.uniform(-1.5,.6)),'bias':float(10**rng.uniform(-.5,1))}
        for name in ['site','sitechem','localchem','regionchem','globalchem','chemlinear','local','region','count','poly']:
            if rng.random()<.55:cfg[name]=float(10**rng.uniform(-2,.4))
        configs.append(cfg)
    result=[];t=time.time()
    for i,cfg in enumerate(configs):
        sc=evaluate(ks,y,ss,cfg);result.append((sc,cfg))
        if i%100==0:print(i,'seconds',round(time.time()-t,1),flush=True)
    result.sort(key=lambda x:x[0][0],reverse=True)
    json.dump(result,open('kernel_search.json','w'),indent=2)
    for score,cfg in result[:15]:print(score,cfg,flush=True)
