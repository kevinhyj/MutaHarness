import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.special import expit,logit
from scipy.stats import rankdata,spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from proposal import AA,INDEX,HYD,VOL,CHG,mutations

# Assumed GFP secondary-structure intervals; these are priors, not coordinates.
STRANDS=[(11,22),(25,37),(40,50),(79,89),(94,104),(110,118),(123,132),(135,142),(145,153),(161,172),(177,189),(198,207),(219,229)]
# Hypothetical inward alternating parity, to be tested rather than asserted.
PARITIES=[0,1,0,0,0,0,1,1,0,1,0,1,1]

def annotation(ref,p):
    pos=p+1; strand=next((k for k,(a,b) in enumerate(STRANDS) if a<=pos<=b),-1)
    beta=strand>=0; helix=58<=pos<=75
    local=[HYD[INDEX[ref[t]]] for t in range(max(0,p-4),min(len(ref),p+5)) if (t-p)%2==0]
    other=[HYD[INDEX[ref[t]]] for t in range(max(0,p-4),min(len(ref),p+5)) if (t-p)%2]
    amph=expit((np.mean(local)-np.mean(other))/2) if beta else .4
    inward=float(beta and pos%2==PARITIES[strand])
    return beta,helix,amph,inward

def featurize(records,ref):
    ms=[mutations(r,ref) for r in records]
    annotations=[annotation(ref,p) for p in range(len(ref))]
    props=[]; sparse=[]
    for mm in ms:
        arr=[]; z=np.zeros(len(ref)*4+400+40)
        for p,a,b in mm:
            i,j=INDEX[a],INDEX[b]; beta,helix,burial,inward=annotations[p]
            dh=(HYD[j]-HYD[i])/5; dv=(VOL[j]-VOL[i])/100; dc=CHG[j]-CHG[i]
            hydloss=max(0,-dh); volabs=abs(dv); charged=float(b in 'DEKR'); aromaticloss=float(a in 'FWY' and b not in 'FWY'); pro=float(b=='P'); gly=float(b=='G')
            conserv=float((a in 'ILVM' and b in 'ILVM') or (a in 'DE' and b in 'DE') or(a in 'KR' and b in 'KR') or(a in 'ST' and b in 'ST') or(a in 'NQ' and b in 'NQ') or(a in 'FWY' and b in 'FWY'))
            dist=np.sqrt(dh**2+dv**2+dc**2)
            corehyd=float(a in 'ILVMFWYAC')
            v=[1,dh,abs(dh),dv,volabs,dc,abs(dc),pro,gly,a=='P',a=='G',charged,aromaticloss,conserv,dist,beta,helix,burial,inward,hydloss,hydloss*burial,hydloss*inward,charged*burial,charged*inward,pro*beta,pro*helix,gly*beta,gly*helix,volabs*burial,volabs*inward,aromaticloss*burial,aromaticloss*inward,corehyd*charged,corehyd*float(b in 'NSTQDEKR'),float(65<=p+1<=67),float(p+1 in [69,94,96,148,203,205,222]),float(p<10 or p>229),float(p+1 in range(90,105)),float(p+1 in range(130,143)),float(p+1 in range(155,173)),float(p+1 in range(215,230))]
            arr.append(v)
            z[p]=1; z[len(ref)+p]=dh;z[len(ref)*2+p]=dv;z[len(ref)*3+p]=dc
            z[4*len(ref)+20*i+j]+=1
            z[4*len(ref)+400+i]+=1;z[4*len(ref)+420+j]+=1
        x=np.array(arr,float).reshape(-1,41)
        n=len(mm)
        sums=x.sum(0); maxs=x.max(0) if n else np.zeros(41)
        # Counts, maxima, pair burden: shared features allow unseen-site generalization.
        props.append(np.r_[n,n*n,np.sqrt(n),sums,maxs, sums[[7,14,19,20,21,24,32]]**2])
        sparse.append(z)
    return np.array(props),np.array(sparse),ms

def target(y,mode):
    if mode=='rank':return (rankdata(y)-.5)/len(y)
    if mode=='latent':return logit(np.clip((y-1.28)/(max(3.9,float(np.max(y))) -1.28),.01,.99))
    if mode=='soft':return np.arcsinh((y-2.5)/.25)
    return y

def ridge_pred(x,q,y,alpha=20,mode='raw'):
    scale=np.maximum(x.std(0),.2)
    return Ridge(alpha=alpha).fit(x/scale,target(y,mode)).predict(q/scale)

def kernel_pred(x,q,z,zq,y,alpha=1,mode='raw',zw=1,bio=1,norm=False,rbf=0):
    scale=np.maximum(x.std(0),.2); xs=x/scale;qs=q/scale
    # Dimensional blocks have separate prior weights, fitted coefficients support-only.
    xs=(xs-xs.mean(0))/np.sqrt(x.shape[1]);qs=(qs-(x/scale).mean(0))/np.sqrt(x.shape[1])
    zs=z*zw;zqs=zq*zw
    if norm:
        zs=zs/np.maximum(np.linalg.norm(zs,axis=1,keepdims=True),1);zqs=zqs/np.maximum(np.linalg.norm(zqs,axis=1,keepdims=True),1)
    k=bio*xs@xs.T+zs@zs.T; kq=bio*qs@xs.T+zqs@zs.T
    if rbf:
        from scipy.spatial.distance import cdist
        k+=rbf*np.exp(-cdist(xs,xs,'sqeuclidean')/2);kq+=rbf*np.exp(-cdist(qs,xs,'sqeuclidean')/2)
    yy=target(y,mode); mu=yy.mean()
    return mu+kq@np.linalg.solve(k+alpha*np.eye(len(y)),yy-mu)

def et_pred(x,q,y,leaf=3,mode='raw',mf=1.,trees=200):
    return ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=leaf,max_features=mf,random_state=1729,n_jobs=1).fit(x,target(y,mode)).predict(q)

if __name__=='__main__':
 import json,time
 c=json.load(open('case.json'));s=c['support'];y=np.array([r['fitness'] for r in s]);x,z,ms=featurize(s,c['reference_sequence']);multi=np.array([i for i,m in enumerate(ms) if len(m)>1]);single=np.setdiff1d(np.arange(len(y)),multi)
 splits=[]
 for seed in [12,47,103]:
  for tr,te in KFold(4,shuffle=True,random_state=seed).split(multi):splits.append((np.r_[single,multi[tr]],multi[te]))
 configs=[]
 for mode in ['raw','rank','latent','soft']:
  for alpha in [3,20,100]:configs.append(('ridge',alpha,mode))
  for alpha in [.3,3,20]:
   for zw in [.3,1,3]:configs.append(('kernel',alpha,mode,zw))
  for leaf in [2,5,10]:configs.append(('et',leaf,mode))
 predictions={};results=[]
 for cfg in configs:
  out=[];cors=[]
  for tr,te in splits:
   if cfg[0]=='ridge':p=ridge_pred(x[tr],x[te],y[tr],cfg[1],cfg[2])
   elif cfg[0]=='kernel':p=kernel_pred(x[tr],x[te],z[tr],z[te],y[tr],cfg[1],cfg[2],cfg[3])
   else:p=et_pred(x[tr],x[te],y[tr],cfg[1],cfg[2],trees=100)
   out.append(p);cors.append(spearmanr(p,y[te]).statistic)
  predictions[str(cfg)]=out;results.append((np.mean(cors),np.std(cors),cfg))
  print(round(np.mean(cors),4),cfg,flush=True)
 results.sort(reverse=True);print('TOP',results[:20]);
 np.savez('cv_exploration.npz',y=y,x=x,z=z)
 with open('validation_results.json','w') as f:json.dump({'results':results,'predictions':predictions,'splits':[(a.tolist(),b.tolist()) for a,b in splits]},f,default=lambda v:np.asarray(v).tolist())

def ge_pred(x,q,z,zq,y,alpha=.3,zw=.3,ceiling=4.0,floor=1.30,kind='sigmoid',bio=1.):
    from scipy.optimize import least_squares
    from scipy.spatial.distance import cdist
    scale=np.maximum(x.std(0),.2); mean=x.mean(0)
    xs=(x-mean)/scale/np.sqrt(x.shape[1]);qs=(q-mean)/scale/np.sqrt(x.shape[1]);zs=z*zw;zqs=zq*zw
    k=bio*xs@xs.T+zs@zs.T;kq=bio*qs@xs.T+zqs@zs.T
    vals,vecs=np.linalg.eigh(k);keep=vals>1e-8; vals=vals[keep];vecs=vecs[:,keep]
    design=vecs*np.sqrt(vals);designq=(kq@vecs)/np.sqrt(vals)
    design=np.c_[np.ones(len(y)),design];designq=np.c_[np.ones(len(q)),designq]
    sy=logit(np.clip((y-floor)/(ceiling-floor),.015,.985))
    penalty=np.r_[.001,np.ones(len(vals))]*alpha
    init=np.linalg.solve(design.T@design+np.diag(penalty),design.T@sy)
    def fun(w):
        a=design@w; fit=floor+(ceiling-floor)*expit(a)
        return np.r_[fit-y,np.sqrt(penalty)*w]
    def jac(w):
        a=expit(design@w); deriv=(ceiling-floor)*a*(1-a)
        return np.r_[design*deriv[:,None],np.diag(np.sqrt(penalty))]
    opt=least_squares(fun,init,jac=jac,max_nfev=100,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    return designq@opt.x

def pair_pred(x,q,z,zq,y,alpha=1,zw=.3,power=.5):
    from scipy.optimize import minimize
    scale=np.maximum(x.std(0),.2);mean=x.mean(0)
    xs=(x-mean)/scale/np.sqrt(x.shape[1]);qs=(q-mean)/scale/np.sqrt(x.shape[1]);zs=z*zw;zqs=zq*zw
    k=xs@xs.T+zs@zs.T;kq=qs@xs.T+zqs@zs.T
    vals,vecs=np.linalg.eigh(k);keep=vals>1e-8; vals=vals[keep];vecs=vecs[:,keep]
    d=vecs*np.sqrt(vals);dq=(kq@vecs)/np.sqrt(vals)
    i,j=np.triu_indices(len(y),1);dif=y[i]-y[j];keep=abs(dif)>1e-8;i=i[keep];j=j[keep];dif=dif[keep]
    dd=(d[i]-d[j])*np.sign(dif)[:,None];weight=np.maximum(np.abs(dif),.01)**power;weight=weight/weight.mean()
    def fun(w):
        pred=dd@w
        loss=np.sum(weight*np.logaddexp(0,-pred))/len(i)+alpha*np.sum(w*w)/2
        grad=-(dd.T@(weight*expit(-pred)))/len(i)+alpha*w
        return loss,grad
    opt=minimize(fun,np.zeros(d.shape[1]),method='L-BFGS-B',jac=True,options={'maxiter':100,'ftol':1e-9})
    return dq@opt.x
