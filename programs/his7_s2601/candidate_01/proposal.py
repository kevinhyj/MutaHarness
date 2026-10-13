"""Support-only deterministic ranking of substitution combinations.

No fitted state, outcomes, sequences, or prediction lists are stored here.
The pooling sets below are assumed statistical mutation neighborhoods identified
from the visible sequence positions, NOT experimentally verified structures.
Configuration was selected with repeated support holdouts; see decision_note.txt.
Dependencies: numpy, scipy, scikit-learn. There are no runtime file reads.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata
from sklearn.linear_model import ElasticNet
from threadpoolctl import threadpool_limits

SEED = 20261013
GROUPS = (
    frozenset(range(1, 36)),
    frozenset((56, 59, 60, 62, 76, 77, 78, 79, 80, 81, 82, 85)),
    frozenset((66, 67, 73, 86, 87, 92, 93)),
    frozenset(range(96, 123)),
    frozenset((137, 139, 140, 141, 142, 143, 144, 157, 159, 160, 161, 163)),
    frozenset((145, 147, 148, 149, 151, 152, 154, 164, 165, 168, 169, 170)),
    frozenset((171, 172, 173, 175, 177, 178, 180, 195, 196, 198, 200)),
    frozenset((181, 183, 185, 186, 187, 189, 201, 202, 203, 204, 205,
               206, 207, 208, 209, 210, 211)),
)


def mutations(record, reference):
    sequence = record.get('sequence')
    if sequence and len(sequence) == len(reference):
        return [(i + 1, a, b) for i, (a, b) in enumerate(zip(reference, sequence))
                if a != b]
    return sorted((int(p), a, b) for a, p, b in
                  re.findall(r'([A-Z])(\d+)([A-Z])', record.get('mutation', '')))


def _dictionary(record, reference):
    changes = mutations(record, reference)
    count = len(changes)
    overlaps = [sum(p in neighborhood for p, a, b in changes)
                for neighborhood in GROUPS]
    region = int(np.argmax(overlaps)) if max(overlaps, default=0) else 8
    # Feature scaling expresses shrinkage strengths, without assuming a WT value.
    d = {'bias': 3., 'multi': 2. * (count > 1), 'count': count / 6.}
    d['region:%s' % region] = float(count > 1)
    d['regioncount:%s' % region] = count / 6.
    for p, a, b in changes:
        d['m:%s:%s' % (p, b)] = 1.
    return d, count


def _design(records, reference):
    rows = [_dictionary(r, reference) for r in records]
    # This vocabulary uses sequence information only. Unobserved mutant features
    # have zero fitted linear coefficients. Their norm contributes appropriately
    # to RBF distance, so novel substitutions increase distance from supports.
    keys = sorted({k for d, n in rows for k in d})
    index = {key: i for i, key in enumerate(keys)}
    x = np.zeros((len(rows), len(keys)), dtype=float)
    for i, (d, count) in enumerate(rows):
        for key, value in d.items():
            x[i, index[key]] = value
    return x, np.array([count for d, count in rows])


def _fit_predict(support, query, reference):
    n = len(support)
    x, counts = _design(support + query, reference)
    train, test = x[:n], x[n:]
    fitness = np.asarray([r['fitness'] for r in support], dtype=float)
    # Empirical midranks preserve assay direction and ties, with no WT assumption
    # and no probability calibration. Recomputed strictly from current supports.
    target = (rankdata(fitness, method='average') - .5) / n
    single = counts[:n] == 1
    baseline = float(np.median(target[single] if single.any() else target))
    residual = target - baseline
    linear = train @ train.T
    cross = test @ train.T
    squared_train = np.sum(train * train, axis=1)
    squared_test = np.sum(test * test, axis=1)
    distance_train = np.maximum(squared_train[:, None] + squared_train[None, :]
                                - 2. * linear, 0.)
    distance_test = np.maximum(squared_test[:, None] + squared_train[None, :]
                               - 2. * cross, 0.)
    # Regularized additive, second-order epistasis, and smooth higher-order models.
    kernels = (
        (linear, cross, .5),
        (linear + .2 * linear ** 2 / 8., cross + .2 * cross ** 2 / 8., 1.),
        (np.exp(-.03 * distance_train), np.exp(-.03 * distance_test), .005),
    )
    predictions = []
    for k, kt, penalty in kernels:
        coefficients = solve(k + penalty * np.eye(n), residual, assume_a='pos')
        predictions.append(kt @ coefficients + baseline)
    # A different shrinkage geometry favors a limited set of informative effects.
    sparse = ElasticNet(alpha=.004, l1_ratio=.2, fit_intercept=True,
                        selection='cyclic', max_iter=10000, tol=1e-7,
                        random_state=SEED)
    sparse.fit(train, target)
    predictions.append(sparse.predict(test))
    # Reduced-feedback validation slightly favored additive shrinkage. This
    # count-only rule changes no learned effects and never inspects query labels.
    feedback_count = int(np.sum(counts[:n] > 1))
    additive_weight = .25 + .30 * np.clip((64. - feedback_count) / 32., 0., 1.)
    weights = np.array([additive_weight] + [(1. - additive_weight) / 3.] * 3)
    return weights @ np.asarray(predictions)


def propose(case, assignment):
    """Fit only supplied supports and score every supplied query; request no data."""
    support = list(case.get('support', []))
    query = list(case.get('query', []))
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not support:
        return {'scores': {r['id']: 0. for r in query}, 'selected_ids': []}
    if not all(np.isfinite(float(r['fitness'])) for r in support):
        raise ValueError('Support fitness must be finite')
    with threadpool_limits(limits=1):
        prediction = _fit_predict(support, query, case['reference_sequence'])
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite model prediction')
    # Arbitrary ordinal scores in the requested interval. Equal predictions tie.
    score = 2. * (rankdata(prediction, method='average') - .5) / len(query) - 1.
    return {'scores': {r['id']: float(s) for r, s in zip(query, score)},
            'selected_ids': []}
