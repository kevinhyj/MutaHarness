"""Support-only deterministic ordinal ranking with an additive kernel ensemble.
All fitted quantities are recomputed from case['support']; no data files are read.
"""
import re
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Hydropathy (Kyte-Doolittle), volume, charge, polarity, and sidechain flags.
_H = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
_V = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
_Q = [0,0,-1,-1,0,0,.25,0,1,0,0,0,0,0,1,0,0,0,0,0]
_P = [8.1,5.5,13,12.3,5.2,9,10.4,5.2,11.3,4.9,5.7,11.6,8,10.5,10.5,9.2,8.6,5.9,5.4,6.2]
PROP = np.array([_H,_V,_Q,_P,[a in 'FYW' for a in AA],[a=='P' for a in AA],[a=='G' for a in AA],[a in 'STNQ' for a in AA]]).T.astype(float)
PROP = (PROP-PROP.mean(0))/PROP.std(0)
INDEX = {a:i for i,a in enumerate(AA)}

def mutations(record, reference):
    text = record.get('mutation','')
    out = [(int(p),a,b) for a,p,b in re.findall(r'([A-Z])(\d+)([A-Z])',text)]
    if not out and record.get('sequence'):
        out = [(i+1,a,b) for i,(a,b) in enumerate(zip(reference,record['sequence'])) if a!=b]
    return out

# Fixed configurations selected by support-only repeated validation. These are
# regularization and feature priors, not fitted coefficients or assay outcomes.
# No structural annotation, external sequence, or predictor is required.
CONFIGS = (
    # chemical linear + local nonlinear substitution similarity
    dict(weight=.4, chem=1.5, exact=.5, smooth=.2, alpha=1.,
         offset=.03, intercept="mean", rbf=0., local=1., localbw=12.),
    # same-site and neighboring-site nonlinear substitution similarity
    dict(weight=.3, chem=0., exact=2., smooth=.2, alpha=.6,
         offset=.1, intercept="mean", rbf=2., local=1., localbw=5.),
    # more strongly regularized same-site kernel, with learned free intercept
    dict(weight=.3, chem=0., exact=.5, smooth=.2, alpha=2.,
         offset=.03, intercept="gls", rbf=4., local=0., localbw=5.),
)


def _kernel_parts(records, reference):
    """Aggregate mutation-pair kernels into variant-pair kernels.

    Features depend only on the amino acids and positions in each input record.
    Query sequences introduce no information into the support fit. A sequence-
    distance kernel is a weak transfer prior, not a spatial contact model.
    """
    parsed = [mutations(r, reference) for r in records]
    tokens = sorted(set(m for ms in parsed for m in ms
                        if m[1] in INDEX and m[2] in INDEX))
    if not tokens:
        z = np.zeros((len(records), len(records)))
        return {name:z.copy() for name in ('pos','exact','chem','smooth','rbf','local5','local12')}
    lookup = {m:i for i,m in enumerate(tokens)}
    incidence = np.zeros((len(records), len(tokens)))
    for i,ms in enumerate(parsed):
        for m in ms:
            if m in lookup:
                incidence[i,lookup[m]] += 1.
    positions = np.array([m[0] for m in tokens])
    dest = np.array([INDEX[m[2]] for m in tokens])
    delta = np.array([PROP[INDEX[b]]-PROP[INDEX[a]] for _,a,b in tokens])
    position_dist2 = (positions[:,None]-positions[None,:])**2
    same = (position_dist2 == 0).astype(float)
    chem = delta @ delta.T
    norm = np.sum(delta*delta,axis=1)
    delta_dist2 = np.maximum(norm[:,None]+norm[None,:]-2*chem,0.)
    local = np.exp(-position_dist2/8.)  # bandwidth two residues
    def aggregate(k):
        return incidence @ k @ incidence.T
    return dict(pos=aggregate(same),
                exact=aggregate(same*(dest[:,None]==dest[None,:])),
                chem=aggregate(same*chem/8.),
                smooth=aggregate(local),
                rbf=aggregate(same*np.exp(-delta_dist2/5.)),
                local5=aggregate(local*np.exp(-delta_dist2/5.)),
                local12=aggregate(local*np.exp(-delta_dist2/12.)))


def _predict(support, query, reference):
    n = len(support)
    parts = _kernel_parts(list(support)+list(query), reference)
    observed = np.array([float(r['fitness']) for r in support])
    result = np.zeros(len(query))
    for cfg in CONFIGS:
        k = (parts['pos'] + cfg['exact']*parts['exact']
             + cfg['chem']*parts['chem'] + cfg['smooth']*parts['smooth']
             + cfg['rbf']*parts['rbf']
             + cfg['local']*parts['local'+str(int(cfg['localbw']))])
        # The shift is a validation-selected link parameter in the supplied
        # assay units. Neither zero nor one is assumed to be measured WT.
        target = np.log(np.maximum(observed,0.)+cfg['offset'])
        cf = cho_factor(k[:n,:n]+cfg['alpha']*np.eye(n),check_finite=False)
        mean = float(target.mean())
        if cfg['intercept']=='gls':
            inv = cho_solve(cf,np.column_stack([target,np.ones(n)]),check_finite=False)
            mean = float(inv[:,0].sum()/inv[:,1].sum())
        coef = cho_solve(cf,target-mean,check_finite=False)
        pred = mean + k[n:,:n] @ coef
        # All ensemble scaling is learned exclusively from the current support.
        result += cfg['weight']*(pred-target.mean())/max(float(target.std()),1e-8)
    return result


def propose(case, assignment):
    """Refit from this invocation's supports; score every current query ID.

    No random algorithms are used (seed convention: 2026). No caches, data-file
    reads, model-file reads, or external tools are used. Scores are ordinal;
    their magnitudes are neither fitness estimates nor probabilities.
    """
    query = list(case.get('query',[]))
    support = [r for r in case.get('support',[])
               if 'fitness' in r and np.isfinite(float(r['fitness']))]
    if not query:
        return {'scores':{},'selected_ids':[]}
    if not support:
        return {'scores':{r['id']:0. for r in query},'selected_ids':[]}
    pred = _predict(support,query,case['reference_sequence'])
    pred = np.nan_to_num(pred,nan=0.,posinf=1e10,neginf=-1e10)
    scores = 2.*(rankdata(pred,method='average')-.5)/len(query)-1.
    return {'scores':{r['id']:float(v) for r,v in zip(query,scores)},
            'selected_ids':[]}
