"""Support-only, deterministic protein variant ranking.
No external data, stored labels, model state or query-ID rules are used.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata, norm
import random

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Standard amino-acid chemical constants: Kyte-Doolittle hydropathy,
# side-chain volume, charge, helix propensity, aromatic/proline/glycine indicators.
HYD = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
HEL = [1.45,.77,.98,1.53,1.12,.53,1.24,1,1.07,1.34,1.2,.73,.59,1.17,.79,.79,.82,1.14,1.14,.61]
PROP = np.array([[HYD[i],VOL[i],(-1 if a in 'DE' else 1 if a in 'KR' else .1 if a=='H' else 0),HEL[i],float(a in 'FWY'),float(a=='P'),float(a=='G'),float(a in 'STNQ'),float(a in 'NQHSTYKR')] for i,a in enumerate(AA)],float)
PROP = (PROP-PROP.mean(0))/PROP.std(0)
INDEX = {a:i for i,a in enumerate(AA)}

def encode(records, reference):
    x = np.zeros((len(records),len(reference),20))
    for i,r in enumerate(records):
        seq=r.get('sequence')
        if seq:
            for p,(a,b) in enumerate(zip(reference,seq)):
                if a!=b: x[i,p,INDEX[b]]=1
        else:
            for m in r['mutation'].split(':'):
                x[i,int(m[1:-1])-1,INDEX[m[-1]]]=1
    return x

def kernels(records,reference,length=1.5):
    x=encode(records,reference)
    wt=np.array([INDEX[a] for a in reference])
    z=x-np.sum(x,2)[:,:,None]*np.eye(20)[wt][None,:,:]
    dist=np.sum((PROP[:,None,:]-PROP[None,:,:])**2,2)/PROP.shape[1]
    sim=np.exp(-dist/(2*length**2))
    per=np.einsum('ipa,ab,jpb->pij',z,sim,z,optimize=True)
    ident=np.einsum('ipa,jpa->pij',x,x,optimize=True)
    presence=x.sum(2)
    kp=presence@presence.T
    ka=per.sum(0)
    ki=ident.sum(0)
    global_delta=z.sum(1)@PROP
    kg=global_delta@global_delta.T/PROP.shape[1]
    count=presence.sum(1)
    kc=np.outer(count,count)
    # ANOVA second-order kernel: cross-position combinations only.
    kpair=(ka**2-np.sum(per**2,0))/2
    return [ka,ki,kp,kg,kc,kpair]

# Fixed hyperparameters selected using support-only, seeded validation.
# These are kernel prior strengths, NOT learned coefficients or label values.
# First five members specialize in combinations; remaining four provide a
# more conservative expert for sparse single substitutions. All fit afresh.
CONFIGS = [[[3.0, 0.1, 0.1, 3.0, 1.0, 0.3, 3.0, 'rank'], [0.1, 0.1, 0.3, 0.0, 0.3]], [[0.25, 0.3, 0.3, 10.0, 0.3, 0.1, 3.0, 'rank'], [0.0, 0.1, 0.1, 0.1, 1.0]], [[1.8, 0.0, 0.1, 3.0, 1.0, 0.1, 3.0, 'asinh'], [0.0, 0.1, 0.0, 0.0, 3.0]], [[1.8, 0.3, 0.3, 3.0, 0.3, 0.1, 1.0, 'rank'], [0.0, 0.0, 0.0, 0.0, 3.0]], [[0.25, 0.1, 1.0, 3.0, 0.3, 0.3, 0.1, 'rank'], [0.0, 0.0, 0.3, 0.0, 3.0]], [[0.8, 0.3, 3.0, 3.0, 0.0, 0.0, 3.0, 'rank'], [0.1, 0.1, 0.0, 0.0, 1.0]], ([3.0, 0.1, 0.3, 0.3, 0.1, 0.1, 1.0, 'rank'], [0, 0, 0, 0, 0]), ([1.8, 0.03, 0.3, 0.3, 0.0, 0.3, 0.1, 'rank'], [0, 0, 0, 0, 0]), ([1.8, 0.0, 1.0, 0.3, 1.0, 1.0, 0.3, 'rank'], [0, 0, 0, 0, 0])]

def extra_kernels(records,reference):
    x=encode(records,reference)
    wt=np.array([INDEX[a] for a in reference])
    z=x-x.sum(2)[:,:,None]*np.eye(20)[wt][None,:,:]
    delta=z@PROP
    source=np.c_[np.ones(len(wt)),PROP[wt]]
    f=np.einsum('npd,pe->nde',delta,source).reshape(len(records),-1)
    transfer=f@f.T/f.shape[1]
    magnitude=np.abs(delta).sum(1)
    kmag=magnitude@magnitude.T/PROP.shape[1]
    total=delta.sum(1)
    pairs=(np.einsum('nd,ne->nde',total,total)-np.einsum('npd,npe->nde',delta,delta))/2
    pairs=pairs.reshape(len(records),-1)
    kchem=pairs@pairs.T/pairs.shape[1]
    f=np.einsum('npa,pb->nab',delta,np.eye(20)[wt]).reshape(len(records),-1)
    kref=f@f.T/PROP.shape[1]
    # Sequence-only alpha-helical-face hypothesis. Residues +/-3 and +/-4
    # approximately share a helix face. Their mean hydropathy is a proxy,
    # not a supplied structure, burial measurement, or established mechanism.
    h=np.array([HYD[INDEX[a]] for a in reference])
    context=[]
    for p in range(len(reference)):
        nearby=[h[j] for j in range(max(0,p-4),min(len(reference),p+5)) if abs(p-j) in (3,4)]
        context.append(np.mean(nearby) if nearby else 0.)
    context=np.array(context)
    context=(context-context.mean())/max(float(context.std()),1.e-8)
    f=(delta*context[None,:,None]).sum(1)
    environment=f@f.T/PROP.shape[1]
    return [transfer,kmag,kchem,kref,environment]

def transform_target(y,mode):
    if mode=='rank': return norm.ppf((rankdata(y)-.5)/len(y))
    if mode=='asinh': return np.arcsinh(y)
    return y.copy()

def propose(case,assignment):
    """Refit from the supplied support and score every supplied query.

    Returned numbers encode an ordering only. They are neither probabilities
    nor estimates of absolute stability, and no additional assay is requested.
    """
    random.seed(2026)
    np.random.seed(2026)
    support=[r for r in case.get('support',[]) if np.isfinite(float(r['fitness']))]
    support=sorted(support,key=lambda r:(r.get('sequence',''),r.get('mutation',''),float(r['fitness'])))
    query=list(case.get('query',[]))
    if not query: return {'scores':{},'selected_ids':[]}
    if not support: return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    y=np.array([float(r['fitness']) for r in support])
    records=support+query
    reference=case['reference_sequence']
    nt=len(support)
    extras=extra_kernels(records,reference)
    # Local per-invocation memoization of sequence features only. No fitted
    # state survives invocation, and no filesystem or network reads occur.
    cache={}
    predictions=[]
    for base,weights in CONFIGS:
        length,ident,pos,glob,count,pair,noise,mode=base
        if length not in cache: cache[length]=kernels(records,reference,length)
        parts=cache[length]
        k=parts[0]+ident*parts[1]+pos*parts[2]+glob*parts[3]+count*parts[4]+pair*parts[5]+1.
        for weight,extra in zip(weights,extras): k+=weight*extra
        a=k[:nt,:nt]+(noise+1.e-9)*np.eye(nt)
        yy=transform_target(y,mode)
        coef=cho_solve(cho_factor(a,lower=True,check_finite=False),yy,check_finite=False)
        pred=k[nt:,:nt]@coef
        # Rank blending is invariant to differing response transform scales.
        predictions.append((rankdata(pred)-.5)/len(query))
    main=np.mean(predictions[:5],axis=0)
    specialist=np.mean(predictions[5:],axis=0)
    single=np.array([sum(a!=b for a,b in zip(reference,r['sequence']))==1 if r.get('sequence') else len(r['mutation'].split(':'))==1 for r in query])
    values=main.copy()
    values[single]=.25*main[single]+.75*specialist[single]
    ordinal=2*(rankdata(values)-.5)/len(query)-1
    ordinal=np.clip(np.nan_to_num(ordinal),-1.,1.)
    return {'scores':{r['id']:float(v) for r,v in zip(query,ordinal)},'selected_ids':[]}
