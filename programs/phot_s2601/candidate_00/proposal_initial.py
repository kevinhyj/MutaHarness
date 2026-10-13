"""Deterministic support-only ordinal fitness predictor."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import numpy as np
from scipy.stats import rankdata


def mutations(record, reference):
    seq = record.get('sequence')
    if seq and len(seq) == len(reference):
        return frozenset(f'{a}{i+1}{b}' for i, (a, b) in enumerate(zip(reference, seq)) if a != b)
    return frozenset(x for x in record.get('mutation', '').split(':') if x)


def propose(case, assignment):
    support = list(case['support'])
    query = list(case['query'])
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not support:
        return {'scores': {r['id']: 0.0 for r in query}, 'selected_ids': []}
    sets = [mutations(r, case['reference_sequence']) for r in support + query]
    alphabet = sorted(set().union(*sets))
    x = np.array([[float(m in s) for m in alphabet] for s in sets])
    n = len(support)
    y = np.array([r['fitness'] for r in support], dtype=float)
    mean = y.mean()
    xc = x - x[:n].mean(0)
    k = xc @ xc.T
    k += 0.05 * k * k
    k += 5 * np.outer([len(s)>1 for s in sets], [len(s)>1 for s in sets])
    pred = mean + k[n:, :n] @ np.linalg.solve(k[:n,:n] + 0.3*np.eye(n), y-mean)
    scores = (rankdata(pred)-0.5)/len(query)*2-1
    return {'scores': {r['id']: float(s) for r,s in zip(query, scores)}, 'selected_ids': []}
