import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata,spearmanr
from sklearn.model_selection import KFold
from proposal import basic_features

AA='ACDEFGHIKLMNPQRSTVWY'
hyd=[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
vol=[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
pol=[8.1,5.5,13,12.3,5.2,9,10.4,5.2,11.3,4.9,5.7,11.6,8,10.5,10.5,9.2,8.6,5.9,5.4,6.2]
props=np.array([hyd,vol,pol,[1 if a in 'KR' else -1 if a in 'DE' else .25 if a=='H' else 0 for a in AA],
 [float(a in 'FWY') for a in AA],[float(a=='G') for a in AA],[float(a=='P') for a in AA],
 [float(a in 'NQST') for a in AA],[float(a in 'ILV') for a in AA],[float(a=='C') for a in AA]]).T
props=(props-props.mean(0))/props.std(0)

def region(p):
    for i,(a,b) in enumerate([(126,133),(134,140),(141,149),(150,156),(157,165),(166,173),(174,180),(181,189),(190,193),(194,199),(200,205)]):
        if a<=p<=b:return i
    return 11

def kernels(records,ref):
    p,t,pos,ms=basic_features(records,ref)
    n=len(records);z=np.zeros((n,len(pos),30));g=np.zeros((n,400));res=np.zeros((n,20));reg=np.zeros((n,12))
    pi={v:i for i,v in enumerate(pos)}
    for i,m in enumerate(ms):
        for po,a,b in m:
            x,y=AA.index(a),AA.index(b);delta=props[y]-props[x]
            z[i,pi[po],:10]+=delta
            z[i,pi[po],10:20]+=abs(delta)
            z[i,pi[po],20:]+=props[y]
            g[i,x*20+y]+=1
            res[i,y]+=1
            reg[i,region(po)]+=1
    out={'p':p@p.T,'t':t@t.T,'pair':g@g.T,'res':res@res.T,'reg':reg@reg.T}
    for w in [1.5,3,6,12]:
        s=np.exp(-.5*((pos[:,None]-pos[None,:])/w)**2)
        out['s'+str(w)]=p@s@p.T
    zp=z.reshape(n,-1)/np.sqrt(30)
    out['chem']=zp@zp.T
    zg=z.sum(1)/np.sqrt(30)
    out['global']=zg@zg.T
    for w in [3,6,12]:
        s=np.exp(-.5*((pos[:,None]-pos[None,:])/w)**2)
        out['c'+str(w)]=np.einsum('ipf,pq,jqf->ij',z,s,z,optimize=True)/30
    s=np.array([[region(a)==region(b) for b in pos] for a in pos],float)
    out['creg']=np.einsum('ipf,pq,jqf->ij',z,s,z,optimize=True)/30
    # Sharing between corresponding RRM beta strands is a tested hypothesis.
    rr=np.array([0 if region(a) in (0,3,5,9) else 1 if region(a) in (2,7) else 2 for a in pos])
    s=(rr[:,None]==rr[None,:]).astype(float)
    out['ctopo']=np.einsum('ipf,pq,jqf->ij',z,s,z,optimize=True)/30
    count=p.sum(1)
    out['count']=np.outer(count,count)
    return out

def transform(y,mode):
    if mode=='raw':return y
    if mode=='sqrt':return np.sqrt(np.maximum(y,0)+.01)
    if mode=='rank':return rankdata(y)/len(y)
    if mode.startswith('log'):return np.log(np.maximum(y,0)+float(mode[3:]))

def fitpred(k,y,tr,te,alpha,mode):
    a=k[np.ix_(tr,tr)];b=k[np.ix_(te,tr)]
    mean=a.mean(0);grand=mean.mean()
    a=a-mean[:,None]-mean[None,:]+grand
    b=b-b.mean(1)[:,None]-mean[None,:]+grand
    target=transform(y[tr],mode)
    coef=solve(a+alpha*np.eye(len(tr)),target-target.mean(),assume_a='pos',check_finite=False)
    return b@coef

def splits(y,orders):
    n=len(y);single=np.where(orders==1)[0];double=np.where(orders>1)[0]
    out=[]
    for seed in [17,42,93]:
        for tr,te in KFold(4,shuffle=True,random_state=seed).split(np.arange(n)):
            out.append((tr,te,'all'))
        for tr,te in KFold(4,shuffle=True,random_state=seed).split(double):
            out.append((np.r_[single,double[tr]],double[te],'feedback'))
    return out

def evaluate(k,y,sp,alpha,mode,ret=False):
    vals=[];preds=[]
    for tr,te,kind in sp:
        pred=fitpred(k,y,tr,te,alpha,mode)
        vals.append(spearmanr(y[te],pred).statistic)
        preds.append(pred)
    return (np.mean(vals),np.mean(vals[::2]),np.asarray(vals),preds) if ret else np.mean(vals)

if __name__=='__main__':
    c=json.load(open('case.json'));recs=c['support'];y=np.array([r['fitness'] for r in recs]);orders=np.array([r['mutation_order'] for r in recs]);ks=kernels(recs,c['reference_sequence']);sp=splits(y,orders)
    results=[]
    for t in [0,.25,1,3]:
      for s in [0,.3,1,3]:
       for w in ['s1.5','s3','s6','s12'] if s else ['s3']:
        k=ks['p']+t*ks['t']+s*ks[w]
        for mode in ['raw','sqrt','log0.03','log0.1','log0.3','rank']:
         for a in [.1,.5,2,8]:
          score=evaluate(k,y,sp,a,mode)
          results.append((score,{'p':1,'t':t,w:s},a,mode))
    results.sort(key=lambda x:-x[0]);print('BASE TOP',results[:15],flush=True)
    json.dump(results,open('base_results.json','w'))
    best=results[0];weights=best[1]
    for stage in range(2):
      candidates=[]
      for name in ['chem','global','c3','c6','c12','creg','ctopo','pair','res','reg','count']:
       for v in [.1,.3,1,3]:
        ww=dict(weights);ww[name]=v
        k=sum(val*ks[key] for key,val in ww.items())
        for a in [.25,1,4]:
         for mode in ['sqrt','log0.03','log0.1','log0.3','rank']:
          score=evaluate(k,y,sp,a,mode);candidates.append((score,ww,a,mode))
      candidates.sort(key=lambda x:-x[0]);print('CHEM STAGE',stage,candidates[:12],flush=True)
      results+=candidates
      if candidates[0][0]>best[0]:best=candidates[0];weights=best[1]
    results.sort(key=lambda x:-x[0]);json.dump(results,open('model_results.json','w'))
    print('BEST',best,flush=True)
