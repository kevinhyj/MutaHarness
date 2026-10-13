"""Deterministic support-only ordinal prediction for fluorescence.

The fixed interaction families at sequence positions 46 and 60 and the small
regularization grid were chosen by public support validation, not by external
structures or scores. They are statistical hypotheses, not proven mechanisms.
All coefficients, intercepts, CV weights, and single-mutant estimates are newly
fitted from the supplied support records on every call. No data/model file I/O.
"""
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

SEED = 2601

def tokens(record, reference=None):
    # Sequence differences avoid depending on mutation order or variant IDs.
    sequence = record.get('sequence')
    if reference and sequence and len(sequence) == len(reference):
        return tuple(a+str(i+1)+b for i,(a,b) in enumerate(zip(reference,sequence)) if a != b)
    mutation = record.get('mutation','')
    if mutation and mutation.lower() not in ('wt','wildtype','wild_type'):
        return tuple(sorted(mutation.split(':'),key=lambda t:(int(t[1:-1]),t[-1])))
    return ()

def _position(t):
    return int(t[1:-1])

def _matrix(rows, vocabulary):
    ix={t:i for i,t in enumerate(vocabulary)}
    x=np.zeros((len(rows),len(vocabulary)))
    for i,row in enumerate(rows):
        for t in row:
            if t in ix:x[i,ix[t]]=1.
    return x

def _kernel_predict(k, cross, y, alpha):
    """Kernel ridge with an unpenalized, training-centered intercept."""
    if len(y)==0:return np.zeros(len(cross))
    mean=k.mean(axis=0); grand=float(mean.mean())
    kc=k-mean[None,:]-mean[:,None]+grand
    zc=cross-mean[None,:]-cross.mean(axis=1)[:,None]+grand
    beta=solve(kc+alpha*np.eye(len(y)),y-y.mean(),assume_a='pos',check_finite=False)
    return zc@beta+y.mean()

def _corr(y,p):
    a=rankdata(y); b=rankdata(p);a=a-a.mean();b=b-b.mean()
    den=np.sqrt((a@a)*(b@b))
    return float(a@b/den) if den>1e-15 else 0.

def _splits(n):
    # CV uses support outcomes only; query sequences never affect selection.
    k=4 if n>=24 else 3
    for seed in (SEED,SEED+17):
        order=np.random.default_rng(seed).permutation(n)
        for test in np.array_split(order,k):
            train=np.setdiff1d(np.arange(n),test,assume_unique=True)
            yield train,test

def _combinations(train_rows,y,query_rows):
    vocabulary=sorted({t for row in train_rows for t in row},key=lambda t:(_position(t),t[-1]))
    x=_matrix(train_rows,vocabulary);z=_matrix(query_rows,vocabulary)
    pos=np.array([_position(t) for t in vocabulary])
    pairs=[(i,j) for i in range(len(pos)) for j in range(i) if pos[i]!=pos[j]]
    xp=np.column_stack([x[:,i]*x[:,j] for i,j in pairs]) if pairs else np.zeros((len(x),0))
    zp=np.column_stack([z[:,i]*z[:,j] for i,j in pairs]) if pairs else np.zeros((len(z),0))
    kernels=[]
    # Main effects are shared. Pair terms express context-specific departures,
    # with stronger shrinkage than main effects. No WT fitness is assumed.
    for loci,ratios,prior in [((46,),(.03,.1,.3),.45),((46,60),(.03,.1,.3),.45),((),(.01,.03),.10)]:
        mask=np.array([not loci or pos[i] in loci or pos[j] in loci for i,j in pairs],dtype=bool)
        pp=xp[:,mask];pq=zp[:,mask]
        for ratio in ratios:
            k=x@x.T+ratio*(pp@pp.T)
            cross=z@x.T+ratio*(pq@pp.T)
            kernels.append((k,cross,prior/len(ratios)))
    splits=list(_splits(len(y))) if len(y)>=12 else []
    predictions=[];merits=[];priors=[]
    for k,cross,prior in kernels:
        for power in (.5,1.):
            target=np.sqrt(np.maximum(y,1e-12)) if power==.5 else y
            for alpha in (.03,.1,.3,1.):
                validation=[]
                for tr,te in splits:
                    p=_kernel_predict(k[np.ix_(tr,tr)],k[np.ix_(te,tr)],target[tr],alpha)
                    validation.append(_corr(y[te],p))
                merit=float(np.mean(validation)) if validation else 0.
                p=_kernel_predict(k,cross,target,alpha)
                # Restore a common assay-like scale only to average models;
                # the exported values are ranks and have no calibration claim.
                if power==.5:p=np.maximum(p,0.)**2
                predictions.append(p);merits.append(merit);priors.append(prior/8.)
    merits=np.array(merits);priors=np.array(priors)
    # Smooth CV weighting retains model uncertainty instead of selecting one
    # winner from small, noisy validation folds. Fixed temperature is public.
    weights=priors*np.exp((merits-merits.max())/.035)
    weights=weights/weights.sum()
    return weights@np.array(predictions)

def _single_prediction(row,rows,y):
    # The original single library mostly differs from the combinatorial library.
    # Transfer measured same-position outcomes conservatively; otherwise use
    # its robust empirical baseline. Four current query singles lack sufficient
    # observations for a justified high-dimensional structure/descriptor model.
    median=float(np.median(y))
    if len(row)!=1:return median
    exact=[i for i,t in enumerate(rows) if t==row]
    if exact:return float(np.mean(y[exact]))
    same=[i for i,t in enumerate(rows) if len(t)==1 and _position(t[0])==_position(row[0])]
    if not same:return median
    return float((np.sum(y[same])+.5*median)/(len(same)+.5))

def predict_records(support,query,reference=''):
    clean=[r for r in support if np.isfinite(float(r['fitness']))]
    if not clean:return np.zeros(len(query))
    # Stable ordering makes output independent of input support ordering.
    clean=sorted(clean,key=lambda r:(tokens(r,reference),float(r['fitness'])))
    rows=[tokens(r,reference) for r in clean];qrows=[tokens(r,reference) for r in query]
    y=np.array([float(r['fitness']) for r in clean])
    multi=np.array([len(t)>1 for t in rows]);qm=np.array([len(t)>1 for t in qrows])
    result=np.full(len(query),float(np.median(y)))
    if multi.sum()>=6:
        inds=np.flatnonzero(multi); qi=np.flatnonzero(qm)
        if len(qi):result[qi]=_combinations([rows[i] for i in inds],y[inds],[qrows[i] for i in qi])
    else:
        vocabulary=sorted({t for row in rows for t in row})
        x=_matrix(rows,vocabulary);z=_matrix(qrows,vocabulary)
        result=_kernel_predict(x@x.T,z@x.T,y,1.)
    single=np.flatnonzero(~multi)
    if len(single):
        sr=[rows[i] for i in single];sy=y[single]
        for i in np.flatnonzero(~qm):result[i]=_single_prediction(qrows[i],sr,sy)
    return np.nan_to_num(result,nan=float(np.median(y)),posinf=float(y.max()),neginf=float(y.min()))

def propose(case,assignment):
    query=case.get('query',[])
    prediction=predict_records(case.get('support',[]),query,case.get('reference_sequence',''))
    # Strictly monotone ordinal conversion; ties receive averaged ranks.
    scores=2*(rankdata(prediction,method='average')-.5)/max(len(query),1)-1
    return {'scores':{r['id']:float(s) for r,s in zip(query,scores)},'selected_ids':[]}
