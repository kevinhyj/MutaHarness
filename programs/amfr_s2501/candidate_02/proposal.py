"""Deterministic, support-only ordinal protein stability predictor.

This module performs no file or network reads. Every invocation reconstructs
features and fits five regularized kernel models using only case['support'].
Hyperparameters and ensemble weights were selected by repeated support-only
mixed, feedback, and mutation-token holdouts. No learned coefficients, outcomes,
IDs, or fitted states are stored here. See decision_note.txt for limitations.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

RANDOM_SEED = 1729  # The final fitting algorithms themselves have no random steps.
AA = 'ACDEFGHIKLMNPQRSTVWY'
# Fixed biochemical descriptors, not externally trained predictor scores.
HYDRO = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
VOLUME = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.,116.1,140.,227.8,193.6]
HELIX = [1.45,.77,.98,1.53,1.12,.53,1.24,1.,1.07,1.34,1.2,.73,.59,1.17,.79,.79,.82,1.14,1.14,.61]


def properties():
    """Standardize over the fixed 20 amino acids, never over held-out outcomes."""
    p = np.array([
        [HYDRO[i], VOLUME[i], HELIX[i],
         float(c in 'KR') - float(c in 'DE') + .15*(c == 'H'),
         float(c in 'FWY'), float(c == 'P'), float(c == 'G'),
         float(c in 'STNQ'), float(c in 'IVT'), float(c == 'C')]
        for i, c in enumerate(AA)], dtype=float)
    return (p - p.mean(axis=0)) / p.std(axis=0)


def _sequence(record, reference):
    if record.get('sequence'):
        return record['sequence']
    seq = list(reference)
    for token in record.get('mutation', '').split(':'):
        if token:
            seq[int(token[1:-1])-1] = token[-1]
    return ''.join(seq)


def _features(records, reference, p):
    n, L, d = len(records), len(reference), p.shape[1]
    identity = np.zeros((n, L*20))
    physical = np.zeros((n, L*d))
    positions = np.zeros((n, L))
    global_physical = np.zeros((n, d))
    mutations = []
    for i, record in enumerate(records):
        changes = []
        seq = _sequence(record, reference)
        for j, (wt, aa) in enumerate(zip(reference, seq)):
            if wt == aa:
                continue
            u, v = AA.index(wt), AA.index(aa)
            delta = p[v] - p[u]
            identity[i, j*20+v] = 1.
            positions[i, j] = 1.
            physical[i, j*d:(j+1)*d] = delta
            global_physical[i] += delta
            changes.append((j, v))
        mutations.append(changes)
    return identity, physical, positions, global_physical, mutations


def _chemical_kernel(mutations, reference, p, difference=False):
    """Sum site-specific amino-acid similarity; optionally anchor at reference.

    The difference kernel is an RKHS feature difference, so it remains positive
    semidefinite. Anchoring features does not assume a measured WT outcome.
    """
    n = len(mutations)
    distance = ((p[:, None, :] - p[None, :, :])**2).mean(axis=2)
    similarity = np.exp(-distance / 8.)  # amino-acid RBF length scale = 2
    K = np.zeros((n, n))
    sites = {}
    for i, changes in enumerate(mutations):
        for j, aa in changes:
            sites.setdefault(j, []).append((i, aa))
    for j, entries in sites.items():
        idx = np.array([i for i, _ in entries], dtype=int)
        aa = np.array([a for _, a in entries], dtype=int)
        block = similarity[np.ix_(aa, aa)]
        if difference:
            wt = AA.index(reference[j])
            block = (block - similarity[aa, wt][:, None]
                     - similarity[wt, aa][None, :] + similarity[wt, wt])
        K[np.ix_(idx, idx)] += block
    return K


def _ordinal(prediction):
    return (rankdata(prediction, method='average') - .5) / len(prediction)


def _predict(K, n, target, regularization):
    # A ridge noise term regularizes all effects including the fitted intercept.
    coef = solve(K[:n, :n] + regularization*np.eye(n), target,
                 assume_a='pos', check_finite=False)
    return _ordinal(K[n:, :n] @ coef)


def propose(case, assignment):
    """Return finite ordinal scores for every supplied query, without selection."""
    train = case.get('support', [])
    query = case.get('query', [])
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not train:
        return {'scores': {r['id']: 0. for r in query}, 'selected_ids': []}
    reference = case['reference_sequence']
    p = properties()
    I, P, Z, G, mutations = _features(train + query, reference, p)
    n = len(train)
    y = np.array([r['fitness'] for r in train], dtype=float)
    # Labels are transformed within this invocation's training set only.
    target_rank = 3.*(rankdata(y, method='average')-.5)/n
    # An alternative monotone scale expands the least stable observations.
    # The offset is an empirical regularization choice, NOT an assay threshold
    # or a calibrated folding free energy. Stable inverse-softplus evaluation.
    floor = float(y.min()) - .8
    t = y - floor
    target_link = t + np.log(-np.expm1(-t))

    Kidentity = I @ I.T
    Kphysical = (P @ P.T) / p.shape[1]
    Kposition = Z @ Z.T
    Kglobal = (G @ G.T) / p.shape[1]
    main = .5*Kposition + .25*Kphysical + .25*Kglobal
    intercept = np.ones_like(main)  # no measured WT or fixed WT score

    # Two smooth whole-sequence kernels permit nonlinear combination effects,
    # while their shared additive part limits extrapolation from only 96 labels.
    smooth = []
    for exact_weight in (0., .1):
        geometry = .5*Kphysical + exact_weight*Kidentity
        diag = np.diag(geometry)
        distance = np.maximum(diag[:, None]+diag[None, :]-2.*geometry, 0.)
        K = intercept + main + np.exp(-distance)
        smooth.append(_predict(K, n, target_rank, .3))
        if exact_weight == .1:
            linked = _predict(K, n, target_link, .3)
    smooth_rank = .5*(smooth[0]+smooth[1])

    # Chemically smoothed additive effects generalize better to some mutations
    # with no exact observations. A small polynomial term permits epistasis.
    chemical = []
    for difference in (False, True):
        local = _chemical_kernel(mutations, reference, p, difference)
        k = main + local
        K = intercept + k + (.1/3.)*(k*k)
        chemical.append(_predict(K, n, target_rank, 1.))
    chemical_rank = .5*(chemical[0]+chemical[1])

    # This gate uses sequence coverage only, never outcomes from query records.
    # Weights were frozen after support validation; no fitting is cached.
    observed_tokens = set(t for changes in mutations[:n] for t in changes)
    novel = np.array([any(t not in observed_tokens for t in changes)
                      for changes in mutations[n:]], dtype=bool)
    chemical_weight = np.where(novel, .5, 0.)
    blended = (.6*((1.-chemical_weight)*smooth_rank
                   + chemical_weight*chemical_rank) + .4*linked)
    # Absolute magnitudes have no assay calibration; preserve the final order.
    scores = 2.*_ordinal(blended)-1.
    return {'scores': {r['id']: float(s) for r, s in zip(query, scores)},
            'selected_ids': []}
