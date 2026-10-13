"""Support-only ordinal ensemble for GCN4_YEAST_Staller_2018.

No fitted state, training records, external predictors, or query IDs are stored.
Every call refits the ensemble. See decision_note.txt for validation and caveats.
"""
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

# Assumed library annotation: the variable activation-domain segment is 101--144.
# These are sequence descriptors, not a claim of experimentally resolved structure.
GROUPS = ('DE', 'KR', 'FWY', 'FWYLIVM', 'P', 'G', 'STNQ')
SEED = 123


def _features(records, reference):
    seqs = [r['sequence'][100:144] for r in records]
    p = np.asarray([[[float(a in g) for a in s] for g in GROUPS]
                    for s in seqs], dtype=float)
    n = len(records)
    if not n:
        return np.empty((0, 51)), np.empty((0, 42)), np.empty((0, 35)), np.empty(0)
    # Low-frequency positional profiles share information across nearby residues.
    basis = np.cos(np.arange(44)[:, None] * np.pi * np.arange(6)[None, :] / 44)
    smooth6 = (p @ basis).reshape(n, -1)
    smooth5 = (p @ basis[:, :5]).reshape(n, -1)
    comp = p.sum(2)
    # The WxxLF neighborhood is annotated from the supplied reference sequence.
    # Keep its residues 120--124 separate from the two flanking regions.
    domain = np.concatenate([p[:, :, lo:hi].sum(2)
                             for lo, hi in ((0, 19), (19, 24), (24, 44))], axis=1)
    # Maximal local counts allow nonlinear clustering/exposure hypotheses to be
    # tested statistically without assuming a structure or enforcing their sign.
    windows = np.stack([np.stack([p[:, gi, i:i+w].sum(1)
                                  for i in range(45-w)], axis=1).max(1)
                        for gi in range(len(GROUPS)) for w in (5, 9, 15)], axis=1)
    ref = reference[100:144]
    order = np.asarray([sum(a != b for a, b in zip(s, ref)) for s in seqs])
    tree = np.concatenate([comp, domain, windows, order[:, None],
                           (order > 15)[:, None]], axis=1)
    return tree, smooth6, smooth5, order


def _ordinal(y):
    return (rankdata(y, method='average') - .5) / len(y)


def _kernel_predict(x, q, y, alpha=1., gamma=.3):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), .5)
    x = (x - mean) / scale
    q = (q - mean) / scale
    if gamma == 0:
        k, kq = x @ x.T, q @ x.T
    else:
        xx, qq = (x*x).sum(1), (q*q).sum(1)
        d = np.maximum(0., xx[:, None] + xx[None, :] - 2*x@x.T)
        dq = np.maximum(0., qq[:, None] + xx[None, :] - 2*q@x.T)
        k = np.exp(-gamma*d/x.shape[1])
        kq = np.exp(-gamma*dq/x.shape[1])
    ym = y.mean()
    coef = solve(k + alpha*np.eye(len(x)), y-ym, assume_a='pos',
                 check_finite=False)
    return ym + kq @ coef


def _predict(supports, queries, reference):
    xt, xs, xl, order = _features(supports, reference)
    qt, qs, ql, qorder = _features(queries, reference)
    fitness = np.asarray([float(r['fitness']) for r in supports])
    y = _ordinal(fitness)
    # Train-only empirical ranks retain the signed assay orientation, while
    # reducing leverage of the isolated zero and the extreme high measurement.
    # The four seeded tree fits average algorithm and single-mutant weighting.
    treepred = np.zeros(len(queries))
    leaf = 5 if len(supports) >= 40 else max(2, len(supports)//12)
    for cls in (ExtraTreesRegressor, RandomForestRegressor):
        for single_weight in (.3, 1.):
            model = cls(n_estimators=256, min_samples_leaf=leaf,
                        max_features=.8, random_state=SEED, n_jobs=1)
            model.fit(xt, y, sample_weight=np.where(order == 1, single_weight, 1.))
            treepred += .25 * model.predict(qt)
    # Gaussian similarities model nonlinear combinations of smooth composition
    # profiles. Two bandwidths and two global-composition weights hedge scale.
    comp, qcomp = xt[:, :7], qt[:, :7]
    joint = np.zeros(len(queries))
    for duplicate_comp in (False, True):
        x = np.concatenate([xs, comp], axis=1) if duplicate_comp else xs
        q = np.concatenate([qs, qcomp], axis=1) if duplicate_comp else qs
        for gamma in (.3, 1.):
            joint += .25 * _kernel_predict(x, q, y, 1., gamma)
    # The high-distance regime has different measured composition relationships.
    # Fit a regularized local expert when there is enough observed feedback.
    high = order > 15
    qhigh = qorder > 15
    expert = joint.copy()
    if high.sum() >= 8 and qhigh.any():
        local_y = _ordinal(fitness[high])
        local_linear = _kernel_predict(xl[high], ql[qhigh], local_y, 100., 0.)
        local_rbf = _kernel_predict(xl[high], ql[qhigh], local_y, 1., .3)
        # Conditional empirical ranks provide an ordinal rather than physical
        # cross-regime calibration; global trees anchor the ensemble's scale.
        expert[qhigh] = .5*(local_linear + local_rbf)
    # Broad, stable validation-selected mixture; no coefficients are hardcoded.
    return .25*treepred + .75*expert


def propose(case, assignment):
    """Return one bounded ordinal score per current query, and no selections."""
    queries = list(case.get('query', []))
    if not queries:
        return {'scores': {}, 'selected_ids': []}
    supports = [r for r in case.get('support', [])
                if 'fitness' in r and np.isfinite(float(r['fitness']))]
    supports.sort(key=lambda r: (int(r.get('mutation_order', 2)) != 1,
                                 str(r.get('id', '')), r['sequence']))
    if len(supports) < 2:
        return {'scores': {r['id']: 0. for r in queries}, 'selected_ids': []}
    with threadpool_limits(limits=1):
        prediction = _predict(supports, queries, case['reference_sequence'])
    prediction = np.nan_to_num(prediction, nan=0., posinf=1., neginf=-1.)
    scores = 2*(_ordinal(prediction)) - 1
    return {'scores': {r['id']: float(v) for r, v in zip(queries, scores)},
            'selected_ids': []}
