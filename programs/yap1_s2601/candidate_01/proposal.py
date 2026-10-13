#!/usr/bin/env python3
"""Deterministic support-only ordinal regression for protein substitutions.

Fit on each invocation; no I/O, fitted caches, pretrained scores or WT label.
The small fixed ensemble was selected using repeated support-only validation.
Chemical sharing is an inductive assumption, not an established mechanism.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {a: i for i, a in enumerate(AA)}
# Standard scientific descriptors: Kyte-Doolittle hydropathy and approximate
# residue volume (angstrom^3); the remaining columns are chemical indicators.
HYDRO = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]
CATEGORIES = ('AVLIM','FWY','STNQ','DE','KRH','P','G','C','AGST','NQ')


def _properties():
    raw = np.array([[HYDRO[i],VOL[i],{'D':-1,'E':-1,'K':1,'R':1,'H':.1}.get(a,0),float(a in 'FWY'),float(a=='P'),float(a=='G'),float(a in 'STNQH'),float(a in 'DE'),float(a in 'KR'),float(a=='C')] for i,a in enumerate(AA)])
    # Standardization over the twenty scientific descriptor rows is fixed and
    # independent of observations. No data or predictor embedding is imported.
    return (raw-raw.mean(0))/np.maximum(raw.std(0),.1)


PROPS = _properties()
CATEGORY_FLAGS = np.array([[float(a in cat) for cat in CATEGORIES] for a in AA])


def _mutations(record, reference):
    sequence = record.get('sequence')
    if sequence and len(sequence) == len(reference):
        return [(i,reference[i],a) for i,a in enumerate(sequence) if a != reference[i]]
    return [(int(p)-1,w,a) for w,p,a in re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]


def _design(records, reference):
    mutations = [_mutations(r,reference) for r in records]
    # Column indexing is only bookkeeping; it is not fitted from query data.
    # Any query-only position has zero cross-covariance in its local features.
    positions = sorted({p for ms in mutations for p,w,a in ms})
    ix = {p:i for i,p in enumerate(positions)}
    n,l = len(records),len(positions)
    out = {name:np.zeros((n,width)) for name,width in
           [('pos',l),('exact',l*20),('prop',l*10),('global',30),
            ('wtcat',200),('origprop',100),('origcat',100),('substitution',400)]}
    for i,ms in enumerate(mutations):
        for p,w,a in ms:
            if w not in AA_INDEX or a not in AA_INDEX:
                continue
            ai,wi,j = AA_INDEX[a],AA_INDEX[w],ix[p]
            d = (PROPS[ai]-PROPS[wi])/np.sqrt(10)
            cd = CATEGORY_FLAGS[ai]-CATEGORY_FLAGS[wi]
            out['pos'][i,j] = 1
            out['exact'][i,j*20+ai] = 1
            out['prop'][i,j*10:(j+1)*10] = d
            out['global'][i,:10] += d
            out['global'][i,10+ai] += 1
            out['global'][i,10+wi] -= 1
            out['wtcat'][i,wi*10:(wi+1)*10] += cd
            out['substitution'][i,wi*20+ai] += 1
            for k in np.flatnonzero(CATEGORY_FLAGS[wi]):
                out['origprop'][i,k*10:(k+1)*10] += d
                out['origcat'][i,k*10:(k+1)*10] += cd
    return out


# Configuration constants are validation-selected regularization assumptions,
# never learned coefficients, fitness values, variant IDs or prediction lists.
# Members deliberately vary pooling, output scale and presence of epistasis.
CONFIGS = (
    dict(exact=.15,pos=1.,prop=.4,glob=.2,alpha=.03,offset=0.,interaction=.2,wtcat=1.,origprop=.3),
    dict(exact=.15,pos=1.,prop=.4,glob=.2,alpha=.03,offset=.1,interaction=.2,wtcat=1.),
    dict(exact=.15,pos=1.,prop=.4,glob=.2,alpha=.03,offset=0.,interaction=.2,wtcat=1.,origcat=.3),
    dict(exact=.5,pos=1.,prop=.4,glob=.2,alpha=.15,offset=.1,interaction=.2,wtcat=1.),
    dict(exact=.15,pos=1.,prop=.4,glob=.2,alpha=.15,offset=0.,interaction=0.,substitution=.2),
)


def _kernel(parts, cfg, n):
    core = (cfg['pos']*parts['pos']+cfg['exact']*parts['exact']+
            cfg['prop']*parts['prop']+cfg['glob']*parts['global'])
    # The second-order PSD kernel includes smooth chemical interactions. It
    # cannot identify unconstrained individual pair effects with these data.
    scale = max(float(np.trace(core[:n])/n),1e-8)
    k = core + cfg['interaction']*core*core/scale
    for key in ('wtcat','origprop','origcat','substitution'):
        k = k + cfg.get(key,0.)*parts[key]
    return k


def _predict(k, y, cfg, n):
    # A learned intercept leaves the reference's assay value unspecified.
    # Centering is computed strictly from this invocation's support rows.
    train = k[:n]
    train_mean = train.mean(axis=0)
    grand_mean = float(train_mean.mean())
    kc = k-train_mean[None,:]-k.mean(axis=1)[:,None]+grand_mean
    # Positive enrichment-like observations permit a multiplicative null in
    # log space. The offset member reduces leverage of near-floor outcomes.
    # The signed fallback keeps the reusable contract valid for other scales.
    if np.all(y>0):
        z = np.log(y+cfg['offset'])
    else:
        scale = max(float(np.median(np.abs(y))),1e-8)
        z = np.arcsinh(y/scale)
    mean = float(z.mean())
    weights = solve(kc[:n]+cfg['alpha']*np.eye(n),z-mean,assume_a='pos',check_finite=False)
    return kc[n:]@weights+mean


def propose(case, assignment):
    """Return finite ordinal scores for all supplied queries; select no IDs."""
    support,query = case.get('support',[]),case.get('query',[])
    if not query:
        return {'scores':{},'selected_ids':[]}
    if not support:
        return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    n = len(support)
    y = np.array([float(r['fitness']) for r in support],dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError('Support fitness must be finite.')
    if n == 1 or np.ptp(y) == 0:
        return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    design = _design(support+query,case['reference_sequence'])
    # Compute only train and train-to-query covariances. Each coefficient is
    # freshly fitted; no stochastic operations or persistent state are used.
    parts = {name:f@f[:n].T for name,f in design.items()}
    ranks = []
    for cfg in CONFIGS:
        pred = _predict(_kernel(parts,cfg,n),y,cfg,n)
        if not np.all(np.isfinite(pred)):
            raise ArithmeticError('Non-finite model prediction.')
        ranks.append((rankdata(pred,method='average')-.5)/len(query))
    ensemble = np.mean(ranks,axis=0)
    # Ordinal scores, not assay units, probabilities or predicted improvement.
    scores = 2*(rankdata(ensemble,method='average')-.5)/len(query)-1
    return {'scores':{r['id']:float(v) for r,v in zip(query,scores)},'selected_ids':[]}
