"""Support-fitted, deterministic ordinal predictor; no stored outcomes or fitted state."""
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Standard physical descriptors: Kyte-Doolittle hydropathy, side-chain volume,
# formal charge, aromaticity, glycine, proline, and hydrogen-bond polarity.
_HYD = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
_VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
_P = np.array([[ _HYD[i]/3, _VOL[i]/60, float(a in 'KR')-float(a in 'DE')+.15*(a=='H'),float(a in 'FWY'),float(a=='G'),float(a=='P'),float(a in 'NQSTDEKRH')] for i,a in enumerate(AA)])
_P = (_P-_P.mean(0))/np.maximum(_P.std(0),.1)
_AI = {a:i for i,a in enumerate(AA)}

def mutations(record, reference):
    seq = record.get('sequence')
    if seq:
        return [(i,_AI[b]) for i,(a,b) in enumerate(zip(reference,seq)) if a!=b and b in _AI]
    import re
    return [(int(p)-1,_AI[b]) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

def features(records,reference):
    n=len(records); l=len(reference)
    pos=np.zeros((n,l)); exact=np.zeros((n,l*20)); chem=np.zeros((n,l*len(_P[0])))
    glob=np.zeros((n,23))
    for row,r in enumerate(records):
        for p,a in mutations(r,reference):
            if p>=l or reference[p] not in _AI: continue
            w=_AI[reference[p]]; d=_P[a]-_P[w]
            pos[row,p]=1; exact[row,p*20+a]=1; chem[row,p*7:(p+1)*7]=d/np.sqrt(7)
            glob[row,:7]+=d/np.sqrt(7); glob[row,7:14]+=np.abs(d)/np.sqrt(7)
            glob[row,14:21]+=_P[a]/np.sqrt(7); glob[row,21]+=1; glob[row,22]+=np.linalg.norm(d)/3
    return pos,exact,chem,glob

# Coarse C-terminal SH3 hypotheses inferred from canonical topology and this
# reference sequence. These are uncertain priors, not coordinates or mechanisms.
_CORE = frozenset([161,163,175,177,182,183,185,193,194,205,210])
_POCKET = frozenset([160,165,209])

def _group(position, reference):
    if position + 1 in _CORE: return 0
    if position + 1 in _POCKET: return 1
    if reference[position] == 'G': return 2
    return 3

def _kernels(records, reference):
    p,e,chem,g=features(records, reference)
    all_mut=[mutations(r,reference) for r in records]
    atoms=sorted(set(m for ms in all_mut for m in ms))
    index={m:j for j,m in enumerate(atoms)}
    incidence=np.zeros((len(records),len(atoms)))
    for i,ms in enumerate(all_mut):
        for m in ms: incidence[i,index[m]]=1
    if atoms:
        positions=np.array([m[0] for m in atoms]);aas=np.array([m[1] for m in atoms])
        wt=np.array([_AI[reference[m[0]]] for m in atoms])
        delta=_P[aas]-_P[wt]
        distance=np.sum((delta[:,None,:]-delta[None,:,:])**2,axis=2)/7
        groups=np.array([_group(int(p),reference) for p in positions])
        same=(groups[:,None]==groups[None,:])
        shared=incidence@(same*np.exp(-distance))@incidence.T
        broad=incidence@(same*np.exp(-distance/3))@incidence.T
    else: shared=broad=np.zeros((len(records),len(records)))
    base=.3*(p@p.T)+.1*(e@e.T)
    return base,g@g.T,shared,broad

# A small predeclared ensemble neighborhood around support-validation-selected
# configurations. All coefficients, intercepts, and ensemble weights are refit.
_CONFIGS=((.5,1.5,1,'rank'),(.15,1.5,1,'rank'),(.5,.4,1,'rank'),
          (.5,1.5,3,'rank'),(.5,5.,1,'rank'),(.5,1.5,1,'logit'))
_PRIOR=np.array([.50,.12,.10,.08,.05,.15])

def _target(y, mode):
    if mode=='rank':
        from scipy.special import ndtri
        return ndtri(np.clip((rankdata(y)-.5)/len(y),1e-5,1-1e-5))
    # Empirical assay-scale saturation hypothesis, allowing negative fitness.
    # No WT value is inserted. Bounds are model settings, not assay endpoints.
    z=np.clip((y+1.35)/1.85,.025,.975)
    return np.log(z/(1-z))

def _fit_predict(K, train, test, y, alpha, mode):
    kt=K[np.ix_(train,train)];kx=K[np.ix_(test,train)]
    row=kt.mean(1);grand=row.mean()
    kt=kt-row[:,None]-row[None,:]+grand
    kx=kx-kx.mean(1)[:,None]-row[None,:]+grand
    target=_target(y[train],mode);mean=target.mean()
    weights=solve(kt+alpha*np.eye(len(train)),target-mean,assume_a='pos')
    return kx@weights+mean

def _ordinal(values):
    if not len(values): return np.array([],dtype=float)
    return (rankdata(values)-.5)/len(values)

def propose(case, assignment):
    """Refit solely on supplied supports, returning every supplied query ID.

    Ordinal output is intentionally not a calibrated probability or predicted
    experimental gain. Independent invocations have no persistent model state.
    """
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits
    support=[r for r in case.get('support',[]) if r.get('fitness') is not None
             and np.isfinite(float(r['fitness']))]
    query=case.get('query',[]);n=len(support)
    if not query:return {'scores':{},'selected_ids':[]}
    if n<2:return {'scores':{str(r['id']):0. for r in query},'selected_ids':[]}
    with threadpool_limits(limits=1):
        y=np.array([float(r['fitness']) for r in support])
        base,global_k,shared,broad=_kernels(support+query,case['reference_sequence'])
        matrices=[base+gw*global_k+2*(shared if bw==1 else broad)
                  for gw,alpha,bw,mode in _CONFIGS]
        weights=_PRIOR.copy()
        # Feedback holdout keeps original single-mutant information available.
        # With singleton-only public subsets, use ordinary support holdout.
        orders=np.array([len(mutations(r,case['reference_sequence'])) for r in support])
        eligible=np.flatnonzero(orders>=2)
        if len(eligible)<12:eligible=np.arange(n)
        if len(eligible)>=12:
            rng=np.random.default_rng(2601)
            partition=np.array_split(rng.permutation(eligible),4)
            oof=np.zeros((len(_CONFIGS),n))
            for test in partition:
                train=np.setdiff1d(np.arange(n),test)
                for j,(gw,alpha,bw,mode) in enumerate(_CONFIGS):
                    pred=_fit_predict(matrices[j],train,test,y,alpha,mode)
                    # Normalize within a simultaneous held-out query batch;
                    # avoids comparing arbitrary units across link functions.
                    oof[j,test]=_ordinal(pred)
            quality=np.array([spearmanr(y[eligible],v[eligible]).statistic for v in oof])
            quality=np.nan_to_num(quality,nan=0.)
            learned=_PRIOR*np.exp(np.clip((quality-quality.max())/.08,-20,0))
            learned/=learned.sum()
            weights=.6*_PRIOR+.4*learned
        prediction=np.zeros(len(query));train=np.arange(n);test=np.arange(n,n+len(query))
        for w,K,(_,alpha,_,mode) in zip(weights,matrices,_CONFIGS):
            prediction+=w*_ordinal(_fit_predict(K,train,test,y,alpha,mode))
        scores=2*_ordinal(prediction)-1
    return {'scores':{str(r['id']):float(s) for r,s in zip(query,scores)},'selected_ids':[]}
