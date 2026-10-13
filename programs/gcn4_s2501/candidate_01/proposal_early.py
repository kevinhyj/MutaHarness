"""Deterministic support-only ordinal predictor for the supplied protein assay."""
import numpy as np
from scipy.stats import rankdata
from scipy.linalg import solve

AA = 'ACDEFGHIKLMNPQRSTVWY'
GROUPS = ('DE','KR','FWY','LIVM','FWYLIVM','STNQ','PG','DEKR','AG','H')

def _features(records, reference):
    # The assayed activation-domain library spans reference residues 101--144.
    seqs = [r['sequence'][100:144] for r in records]
    out = []
    for s in seqs:
        row = []
        for lo, hi in ((0,44),(0,22),(22,44),(0,14),(14,29),(29,44)):
            t = s[lo:hi]
            row.extend(sum(a==b for a in t) for b in AA)
            row.extend(sum(a in g for a in t) for g in GROUPS)
        row.append(sum(a!=b for a,b in zip(s, reference[100:144])))
        out.append(row)
    return np.asarray(out, dtype=float)

def propose(case, assignment):
    supports = [r for r in case.get('support',[]) if np.isfinite(float(r['fitness']))]
    queries = case.get('query',[])
    if not queries:
        return {'scores':{},'selected_ids':[]}
    if len(supports)<2:
        return {'scores':{r['id']:0.0 for r in queries},'selected_ids':[]}
    reference = case['reference_sequence']
    x = _features(supports, reference)
    q = _features(queries, reference)
    mean, scale = x.mean(0), np.maximum(x.std(0), 0.5)
    x, q = (x-mean)/scale, (q-mean)/scale
    y = np.asarray([r['fitness'] for r in supports])
    y = (rankdata(y)-0.5)/len(y)
    y -= y.mean()
    coefficients = solve(x@x.T + 30*np.eye(len(x)), y, assume_a='pos')
    pred = q@x.T@coefficients
    score = 2*(rankdata(pred)-0.5)/len(pred)-1
    return {'scores':{r['id']:float(v) for r,v in zip(queries,score)},'selected_ids':[]}
