"""Deterministic support-only ranking for the supplied KRAS abundance assay.

All regression targets and coefficients are recomputed at every call. The only
fixed quantities are chemical constants, coarse sequence/domain annotations,
and hyperparameters selected by public support-only validation. No file I/O,
external sequences, model states, fitness constants, or variant IDs are used.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Standard Kyte-Doolittle hydropathy and residue volume (angstrom cubed).
HYD = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL = dict(zip(AA, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))
# Coarse sequence partitions reflecting KRAS fold/loop/domain knowledge.
# These are assumed annotations, not structure coordinates or measured burial.
EDGES = (9,17,25,40,58,76,86,104,120,138,166)
# Balanced local ensemble selected by support-only ranking validation.
CONFIGS = tuple((alpha, region_weight) for region_weight in (.03,.1)
                for alpha in (.03,.1,.3))

def _mutations(record, reference):
    sequence = record.get('sequence')
    if sequence and len(sequence) == len(reference):
        return [(i+1,a,b) for i,(a,b) in enumerate(zip(reference,sequence)) if a != b]
    return [(int(p),a,b) for a,p,b in
            re.findall(r'([A-Z])(\d+)([A-Z])',record.get('mutation',''))]

def _chem(a):
    return np.array([HYD[a]/3, VOL[a]/80, float(a in 'KR')-float(a in 'DE'),
                     float(a in 'FWY'),float(a=='G'),float(a=='P'),
                     float(a in 'STNQ'),float(a=='C')], dtype=float)

def _features(rows, reference):
    mutations = [_mutations(r,reference) for r in rows]
    # Query sequences supply only feature vocabulary, never outcomes. Adding
    # unseen columns is algebraically equivalent to assigning zero fitted effect.
    vocabulary = sorted({(p,b) for mm in mutations for p,a,b in mm})
    token_index = {token:i for i,token in enumerate(vocabulary)}
    n = len(rows)
    exact = np.zeros((n,len(vocabulary)))
    position = np.zeros((n,len(reference)))
    regional = np.zeros((n,12*24))
    count = np.zeros((n,2))
    for i,mm in enumerate(mutations):
        count[i] = [len(mm),len(mm)**2]
        for p,a,b in mm:
            exact[i,token_index[(p,b)]] += 1
            position[i,p-1] += 1
            ca,cb = _chem(a),_chem(b)
            descriptor = np.concatenate((cb-ca,np.abs(cb-ca),cb*ca))
            region = int(np.searchsorted(EDGES,p,side='left'))
            regional[i,24*region:24*(region+1)] += descriptor
    return exact,position,regional,count

def _ordinal(prediction):
    if len(prediction) == 0: return np.empty(0)
    return 2*(rankdata(prediction,method='average')-.5)/len(prediction)-1

def propose(case, assignment):
    """Fit on current supports and rank every current query; request no batch."""
    # No stochastic operations are used. Fixed ordering ensures reproducibility
    # even when the supplied lists are reordered (no random seed is needed).
    supports = sorted(case.get('support',[]),key=lambda r:str(r['id']))
    queries = sorted(case.get('query',[]),key=lambda r:str(r['id']))
    if not queries:
        return {'scores':{}, 'selected_ids':[]}
    reference = case['reference_sequence']
    features = _features(supports+queries,reference)
    n = len(supports)
    if n < 2:
        # A single outcome cannot identify comparative sequence effects.
        return {'scores':{r['id']:0.0 for r in queries}, 'selected_ids':[]}
    target = np.array([float(r['fitness']) for r in supports])
    # Learn the assay order directly. Ties retain equal targets. No WT fitness
    # or probability interpretation is assumed, and direction is preserved.
    y = (rankdata(target,method='average')-.5)/n
    y -= y.mean()
    train_k = [x[:n]@x[:n].T for x in features]
    query_k = [x[n:]@x[:n].T for x in features]
    identity = np.eye(n)
    predictions = []
    for alpha, region_weight in CONFIGS:
        weights = (1.0,1.0,region_weight,.1)
        kt = sum(w*k for w,k in zip(weights,train_k))
        kq = sum(w*k for w,k in zip(weights,query_k))
        # Training-only centering corresponds to an unpenalized fitted intercept.
        mu = kt.mean(axis=0)
        grand = mu.mean()
        kt = kt-mu[:,None]-mu[None,:]+grand
        kq = kq-kq.mean(axis=1)[:,None]-mu[None,:]+grand
        coefficients = solve(kt+alpha*identity,y,assume_a='pos',check_finite=True)
        predictions.append(_ordinal(kq@coefficients))
    # Rank aggregation prevents arbitrary scale differences between members.
    scores = _ordinal(np.mean(predictions,axis=0))
    return {'scores':{r['id']:float(v) for r,v in zip(queries,scores)},
            'selected_ids':[]}
