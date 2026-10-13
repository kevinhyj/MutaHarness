"""Support-only, deterministic ordinal predictor for HIS7 growth fitness.

Configuration was selected by local held-out-support comparisons. No fitted
parameters, outcomes, variant IDs, or prediction lists are stored here.
The sequence windows below describe mutation-library co-occurrence; they are
not asserted structural domains or experimentally established mechanisms.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata, norm
from sklearn.linear_model import ElasticNet
from threadpoolctl import threadpool_limits


def _mutations(record, reference):
    sequence = record.get('sequence')
    if sequence and len(sequence) == len(reference):
        return [(i + 1, b) for i, (a, b) in enumerate(zip(reference, sequence)) if a != b]
    return [(int(p), b) for _, p, b in
            re.findall(r'([A-Z])(\d+)([A-Z])', record.get('mutation', ''))]


def _block(position):
    # Convenient design windows; fitted effects are always learned afresh.
    if position < 40:
        return 0
    if position <= 65 or 76 <= position <= 85:
        return 1
    if position < 96:
        return 2
    if position <= 125:
        return 3
    if position <= 144 or 157 <= position <= 163:
        return 4
    if position <= 170:
        return 5
    if position <= 180 or 195 <= position <= 200:
        return 6
    return 7


def _features(records, reference, vocabulary=None):
    """Allele features; retained for reuse in local exploration scripts."""
    mutations = [_mutations(r, reference) for r in records]
    if vocabulary is None:
        vocabulary = sorted({m for ms in mutations for m in ms})
    lookup = {m: i for i, m in enumerate(vocabulary)}
    x = np.zeros((len(records), len(vocabulary)))
    for i, ms in enumerate(mutations):
        for m in ms:
            if m in lookup:
                x[i, lookup[m]] = 1.0
    return x, vocabulary


def _representation(records, reference):
    mutations = [_mutations(r, reference) for r in records]
    x, vocabulary = _features(records, reference)
    positions = sorted({p for p, a in vocabulary})
    position_lookup = {p: i for i, p in enumerate(positions)}
    occupied = np.zeros((len(records), len(positions)))
    counts = np.zeros((len(records), 8))
    for i, ms in enumerate(mutations):
        for p, a in ms:
            occupied[i, position_lookup[p]] = 1.0
            counts[i, _block(p)] += 1.0
    # The >=2 gate lets combinations have a different background from singles;
    # it is a nonlinear sequence feature, not an assumed fitness threshold.
    features = np.column_stack((x, 0.5 * occupied,
                                (counts >= 2).astype(float), 0.25 * counts))
    return x, occupied, features


def _ordinal(values):
    n = len(values)
    if n == 0:
        return np.empty(0)
    return (rankdata(values, method='average') - 0.5) / n


def _predict(supports, queries, reference):
    n = len(supports)
    x, occupied, features = _representation(supports + queries, reference)
    y = np.asarray([float(r['fitness']) for r in supports], dtype=float)
    # These are empirical ordinal targets, not calibrated probabilities.
    # Identical observed outcomes (including zeros) receive identical ranks.
    ranked = norm.ppf(_ordinal(y))
    sparse = ElasticNet(alpha=0.003, l1_ratio=0.9, fit_intercept=True,
                        max_iter=6000, tol=1e-6, selection='cyclic',
                        random_state=2601)
    sparse.fit(features[:n], ranked)
    sparse_prediction = sparse.predict(features[n:])

    # Exact categorical Hamming distances. A different allele at the same
    # position counts as one difference, including alleles absent in training.
    # Query genotypes define feature columns only; their outcomes are never used.
    xs = x[:n]
    ps = occupied[:n]
    counts = occupied.sum(axis=1)
    distance = (counts[:, None] + counts[:n][None, :]
                - occupied @ ps.T - x @ xs.T)
    kernel = np.exp(-0.8 * np.maximum(distance, 0.0))
    scale = float(np.std(y))
    standardized = (y - np.mean(y)) / scale if scale > 1e-12 else np.zeros(n)
    # Combining raw and ordinal information in this component retained useful
    # interpolation while the main sparse component optimizes ranking directly.
    target = 0.5 * ranked + 0.5 * standardized
    mean = float(np.mean(target))
    coefficients = solve(kernel[:n] + np.eye(n), target - mean,
                         assume_a='pos', check_finite=False)
    nonlinear_prediction = mean + kernel[n:] @ coefficients
    # A fixed broad mixture avoids estimating many ensemble weights from 64
    # feedback variants. Component ranks remove arbitrary score-unit differences.
    return (0.60 * _ordinal(sparse_prediction)
            + 0.40 * _ordinal(nonlinear_prediction))


def propose(case, assignment):
    """Refit from supplied measured supports and score every current query ID.

    Assignment does not affect ranking: no new measurements are selected.
    Feature construction may inspect unlabeled query sequences, never outcomes.
    """
    queries = list(case.get('query', []))
    if not queries:
        return {'scores': {}, 'selected_ids': []}
    supports = []
    for r in case.get('support', []):
        try:
            if np.isfinite(float(r['fitness'])):
                supports.append(r)
        except (KeyError, TypeError, ValueError):
            pass
    if len(supports) < 2:
        return {'scores': {r['id']: 0.0 for r in queries}, 'selected_ids': []}
    with threadpool_limits(limits=1):
        prediction = _predict(supports, queries, case['reference_sequence'])
    # Values are arbitrary ordinal scores in [-1,1]; ties stay tied.
    prediction = np.nan_to_num(prediction, nan=0.5, posinf=1.0, neginf=0.0)
    scores = 2.0 * _ordinal(prediction) - 1.0
    return {'scores': {r['id']: float(s) for r, s in zip(queries, scores)},
            'selected_ids': []}
