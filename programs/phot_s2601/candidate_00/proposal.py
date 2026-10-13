"""Support-only prediction of ordinal fluorescence fitness.

No WT measurement, external sequence information, stored training data, or
pretrained parameters are used. Configurations below were selected with public
support cross-validation; all phenotype-dependent quantities are refit per call.
"""
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


def _fit_kernel(k, train, test, target, noise):
    """Universal kriging: an unpenalized, correctly noise-weighted intercept."""
    a = k[np.ix_(train, train)] + np.diag(noise)
    rhs = np.column_stack((target, np.ones(len(train))))
    sol = np.linalg.solve(a, rhs)
    intercept = sol[:, 0].sum() / max(sol[:, 1].sum(), 1e-12)
    coef = sol[:, 0] - intercept * sol[:, 1]
    return intercept + k[np.ix_(test, train)] @ coef


def _combo_predict(sets, y, n, multi, single):
    # Alleles are inferred from supplied measurements, never from cached lists.
    alphabet = sorted(set().union(*(sets[i] for i in multi)))
    positions = sorted({int(m[1:-1]) for m in alphabet})
    x = np.array([[float(m in s) for m in alphabet] for s in sets])
    xp = np.array([[float(any(int(m[1:-1]) == p for m in s))
                    for p in positions] for s in sets])
    x -= x[multi].mean(0)
    xp -= xp[multi].mean(0)
    ka = x @ x.T
    kp = xp @ xp.T
    # A single mutant in the combinatorial alphabet supplies a weak anchor.
    # Unrelated singles cannot identify effects of the recurring alleles.
    shared = [i for i in single if sets[i] and sets[i].issubset(alphabet)]
    train = np.array(list(multi) + shared, dtype=int)
    test = np.arange(n, len(sets))
    # Stability at very small sample sizes; full and >=32-feedback fits use .01.
    reg = .01 * max(1., (32. / len(multi)) ** 2)
    noise = np.full(len(train), reg)
    noise[len(multi):] = reg / .1
    # Main effects have much larger prior variance than epistatic effects.
    # Allele-by-site interactions pool information across alleles at a site;
    # the RBF component permits modest higher-order smooth deviations.
    configs = [('allele', .003, 1., .20), ('allele', .01, 1., .20),
               ('mixed', .003, 1., .20), ('rbf', 10., 1., .20),
               ('allele', .003, 1.5, .10), ('allele', .01, 1.5, .10)]
    dist = np.maximum(np.diag(ka)[:, None] + np.diag(ka)[None, :] - 2 * ka, 0.)
    result = np.zeros(len(test))
    for kind, strength, power, weight in configs:
        if kind == 'allele':
            kernel = ka + strength * ka * ka
        elif kind == 'mixed':
            kernel = ka + strength * ka * kp
        else:
            kernel = ka + strength * np.exp(-dist / 32.)
        # Signed powers also work for alternative supplied signed assay scales.
        target = np.sign(y[train]) * np.abs(y[train]) ** power
        p = _fit_kernel(kernel, train, test, target, noise)
        result += weight * np.sign(p) * np.abs(p) ** (1. / power)
    return result


def _single_predict(s, support_sets, y, single):
    """Conservative transfer for isolated variants outside the combination grid.

    Physicochemical/sequence-neighborhood extrapolation failed leave-one-out
    validation on the original singles. Thus no structural effect is asserted.
    Same-position observations get only a small, shrunk contribution.
    """
    if len(single):
        base = float(np.median(y[single]))
    else:
        base = float(np.median(y))
    if not s:
        return base
    positions = {int(m[1:-1]) for m in s}
    nearby = [i for i in single if any(int(m[1:-1]) in positions for m in support_sets[i])]
    return .8 * base + .2 * float(np.median(y[nearby])) if nearby else base


def propose(case, assignment):
    """Fit from this invocation's supports and score every current query ID."""
    np.random.seed(1729)
    support = sorted(case.get('support', []), key=lambda r: str(r.get('id', '')))
    query = list(case.get('query', []))
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not support:
        return {'scores': {r['id']: 0.0 for r in query}, 'selected_ids': []}
    reference = case['reference_sequence']
    sets = [mutations(r, reference) for r in support + query]
    n = len(support)
    y = np.array([float(r['fitness']) for r in support])
    if not np.isfinite(y).all():
        raise ValueError('Support fitness must be finite')
    single = np.array([i for i in range(n) if len(sets[i]) <= 1], dtype=int)
    multi = np.array([i for i in range(n) if len(sets[i]) > 1], dtype=int)
    pred = np.array([_single_predict(s, sets[:n], y, single) for s in sets[n:]])
    if len(multi) >= 3:
        combo = _combo_predict(sets, y, n, multi, single)
        for j, s in enumerate(sets[n:]):
            if len(s) > 1:
                pred[j] = combo[j]
    elif len(multi):
        for j, s in enumerate(sets[n:]):
            if len(s) > 1:
                pred[j] = float(np.median(y[multi]))
    # Exact repeat measurements, if provided, dominate generic extrapolation.
    observed = {}
    for s, value in zip(sets[:n], y):
        observed.setdefault(s, []).append(value)
    for j, s in enumerate(sets[n:]):
        if s in observed:
            pred[j] = float(np.mean(observed[s]))
    pred = np.nan_to_num(pred, nan=float(np.median(y)), posinf=float(np.max(y)), neginf=float(np.min(y)))
    scores = 2. * (rankdata(pred, method='average') - .5) / len(query) - 1.
    return {'scores': {r['id']: float(v) for r, v in zip(query, scores)}, 'selected_ids': []}
