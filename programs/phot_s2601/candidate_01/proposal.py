"""Support-only fluorescence ranking, with deterministic validation and shrinkage.

No fitted parameters or outcomes are stored here. The modest configuration grid
was screened with repeated held-out support ranking. Sequence positions identify
categorical sites, not experimentally established structural contacts.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v,'1')
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

def _mutations(record, ref):
    seq=record.get('sequence')
    if seq and len(seq)==len(ref):
        return tuple(f'{a}{i+1}{b}' for i,(a,b) in enumerate(zip(ref,seq)) if a!=b)
    return tuple(m for m in record.get('mutation','').split(':') if m)

def _matrix(records,ref,vocab):
    idx={m:i for i,m in enumerate(vocab)}; x=np.zeros((len(records),len(vocab)))
    for j,r in enumerate(records):
        for m in _mutations(r,ref):
            if m in idx:x[j,idx[m]]=1.
    return x

def _corr(a,b):
    a=rankdata(a);b=rankdata(b);a-=a.mean();b-=b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-12 else 0.

def _ridge(x,y,q,alpha):
    xm=x.mean(0);ym=y.mean();z=x-xm
    co=z.T@solve(z@z.T+alpha*np.eye(len(y)),y-ym,assume_a='pos')
    return ym+(q-xm)@co

# Small, prespecified families; all site importance and coefficients are refit.
_CONFIGS=(
    [('add',0.,a,'raw') for a in (.03,.1,.3)]+
    [('pair',p,a,'raw') for p in (.05,.1,.2) for a in (.01,.03)]+
    [('hub',p,a,'raw') for p in (.15,.3) for a in (.01,.03,.1)]+
    [('strong',p,a,'raw') for p in (.1,.3) for a in (.01,.03,.1)]+
    [('rbf',p,a,'raw') for p in (.02,.05) for a in (.003,.03)]+
    [('pair',.1,a,'rank') for a in (.01,.03,.1)]
)

def _fitpredict(x,y,q,positions,config):
    family,p,alpha,scale=config
    # A rank-response sensitivity model reduces dependence on assay magnitudes.
    yt=(rankdata(y)-.5)/len(y) if scale=='rank' else y
    if family=='add':out=_ridge(x,yt,q,alpha)
    elif family=='rbf':
        d=((x[:,None,:]-x[None,:,:])**2).sum(2)
        dq=((q[:,None,:]-x[None,:,:])**2).sum(2)
        mean=yt.mean()
        out=mean+np.exp(-p*dq)@solve(np.exp(-p*d)+alpha*np.eye(len(y)),yt-mean,assume_a='pos')
    else:
        xm=x.mean(0);xc=x-xm;qc=q-xm
        ii,jj=np.triu_indices(x.shape[1],1)
        mask=positions[ii]!=positions[jj];ii,jj=ii[mask],jj[mask]
        w=np.ones(len(ii))
        if family in ('hub','strong'):
            # Hierarchical epistasis: stronger measured main-effect sites receive
            # more interaction flexibility. This is evidence-based shrinkage,
            # not an assertion of physical residue contacts.
            co=xc.T@solve(xc@xc.T+.1*np.eye(len(y)),y-y.mean(),assume_a='pos')
            sites=np.unique(positions)
            imp=np.array([np.linalg.norm(co[positions==s]) for s in sites])
            if family=='hub':
                active=positions==sites[np.argmax(imp)]
                mask=active[ii]|active[jj];ii,jj=ii[mask],jj[mask];w=np.ones(len(ii))
            else:
                lookup=dict(zip(sites,imp));v=np.array([lookup[s] for s in positions])
                v/=max(v.max(),1e-9);w=np.maximum(v[ii],v[jj])
        z=np.column_stack((x,p*w*xc[:,ii]*xc[:,jj]))
        zq=np.column_stack((q,p*w*qc[:,ii]*qc[:,jj]))
        out=_ridge(z,yt,zq,alpha)
    if scale=='rank':
        # Put sensitivity predictions back onto a common support-derived scale
        # solely to combine orderings. No WT or calibrated probabilities assumed.
        out=np.interp(out,(np.arange(len(y))+.5)/len(y),np.sort(y))
    return out

def _ensemble(x,y,q,positions,validate=True):
    configs=list(_CONFIGS)
    if len(y)<16:
        return _fitpredict(x,y,q,positions,('add',0.,1.,'raw'))
    if not validate:
        chosen=[('pair',.1,.01,'raw'),('hub',.3,.03,'raw'),('strong',.1,.03,'raw')]
        return np.mean([_fitpredict(x,y,q,positions,c) for c in chosen],axis=0)
    folds=[]
    for seed in (1729,2718):
        folds.extend(KFold(n_splits=4,shuffle=True,random_state=seed).split(x))
    quality=[]
    for config in configs:
        values=[]
        for tr,va in folds:
            pred=_fitpredict(x[tr],y[tr],x[va],positions,config)
            values.append(_corr(y[va],pred))
        # A small uncertainty penalty discourages fragile configurations.
        quality.append(np.mean(values)-.15*np.std(values)/np.sqrt(len(values)))
    quality=np.array(quality)
    families=[(c[0],c[3]) for c in configs]
    keys=list(dict.fromkeys(families));fp=[];fq=[]
    for key in keys:
        ix=np.array([i for i,k in enumerate(families) if k==key])
        weights=np.exp((quality[ix]-quality[ix].max())/.025);weights/=weights.sum()
        fp.append(sum(w*_fitpredict(x,y,q,positions,configs[i]) for w,i in zip(weights,ix)))
        fq.append(float(weights@quality[ix]))
    fq=np.array(fq);fw=np.exp((fq-fq.max())/.04);fw/=fw.sum()
    return fw@np.array(fp)

# Standard biochemical constants, used only as cautious similarity priors for
# isolated single mutants outside the combinatorial design.
_HYDRO=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
_VOL=dict(zip('ACDEFGHIKLMNPQRSTVWY',[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]))
def _aa(a):
    return np.array([_HYDRO.get(a,0)/4.5,_VOL.get(a,140)/180.,float(a in 'KR')-float(a in 'DE'),float(a in 'FYW'),float(a=='P'),float(a=='G')])
def _single_predict(support,query,ref,default):
    singles=[r for r in support if len(_mutations(r,ref))==1]
    if not singles:return np.full(len(query),default)
    muts=[_mutations(r,ref)[0] for r in singles];y=np.array([float(r['fitness']) for r in singles]);median=np.median(y)
    out=[]
    for r in query:
        mm=_mutations(r,ref)
        if not mm:out.append(median);continue
        m=mm[0];pos=int(m[1:-1]);target=_aa(m[-1]);source=_aa(m[0]);weights=[]
        for a in muts:
            dist=np.sum((target-_aa(a[-1]))**2)+.5*np.sum((source-_aa(a[0]))**2)
            # Nearby sequence positions are weak evidence; exact position is
            # more informative but still shrunk because anchors are sparse.
            proximity=.04*np.exp(-abs(pos-int(a[1:-1]))/8.)
            if int(a[1:-1])==pos:proximity+=1.
            weights.append(proximity*np.exp(-dist/1.5))
        w=np.array(weights);out.append((.6*median+w@y)/(.6+w.sum()))
    return np.array(out)

def propose(case,assignment):
    """Refit on supplied supports each invocation; return every current query."""
    query=list(case.get('query',[]));ref=case['reference_sequence']
    if not query:return {'scores':{},'selected_ids':[]}
    support=[r for r in case.get('support',[]) if np.isfinite(float(r['fitness']))]
    if not support:return {'scores':{str(r['id']):0. for r in query},'selected_ids':[]}
    multi=[r for r in support if len(_mutations(r,ref))>1]
    train=multi if len(multi)>=8 else support
    # Feature vocabulary is outcome-free; zero columns for unseen alleles cannot
    # acquire a learned effect. It is recreated from the supplied invocation.
    vocab=sorted({m for r in train+query for m in _mutations(r,ref)})
    x=_matrix(train,ref,vocab);q=_matrix(query,ref,vocab)
    positions=np.array([int(m[1:-1]) for m in vocab]);y=np.array([float(r['fitness']) for r in train])
    with threadpool_limits(limits=1):
        pred=_ensemble(x,y,q,positions,validate=False)
    isolated=np.array([len(_mutations(r,ref))<=1 for r in query])
    if isolated.any():
        sub=[r for r,b in zip(query,isolated) if b]
        pred[isolated]=_single_predict(support,sub,ref,float(np.median(y)))
    pred=np.nan_to_num(pred,nan=float(np.median(y)),posinf=float(y.max()),neginf=float(y.min()))
    scores=2*(rankdata(pred,method='average')-.5)/len(query)-1
    return {'scores':{str(r['id']):float(s) for r,s in zip(query,scores)},'selected_ids':[]}
