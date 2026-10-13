#!/usr/bin/env python3
"""Deterministic support-only ordinal predictor for the PAB1 growth assay.

All coefficients and target transforms are fitted anew from the supplied supports.
No external data or stored fitted state is used.
"""
import re
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata

# Ordinary chemical descriptors, not pretrained predictor scores. Columns are
# Kyte-Doolittle hydropathy, residue volume, Grantham polarity, charge, and
# indicators of aromaticity, glycine, proline, amide/hydroxyl, branching, cysteine.
AA = 'ACDEFGHIKLMNPQRSTVWY'
_HYD = [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]
_VOL = [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]
_POL = [8.1,5.5,13,12.3,5.2,9,10.4,5.2,11.3,4.9,5.7,11.6,8,10.5,10.5,9.2,8.6,5.9,5.4,6.2]
PROPERTIES = np.array([
    _HYD, _VOL, _POL,
    [1 if a in 'KR' else -1 if a in 'DE' else .25 if a == 'H' else 0 for a in AA],
    [float(a in 'FWY') for a in AA], [float(a == 'G') for a in AA],
    [float(a == 'P') for a in AA], [float(a in 'NQST') for a in AA],
    [float(a in 'ILV') for a in AA], [float(a == 'C') for a in AA],
], dtype=float).T
PROPERTIES = (PROPERTIES - PROPERTIES.mean(0)) / PROPERTIES.std(0)


def mutations(record, reference):
    text = record.get('mutation', '')
    found = re.findall(r'([A-Z])(\d+)([A-Z])', text)
    if found:
        return [(int(p), a, b) for a, p, b in found if a != b]
    return [(i + 1, a, b) for i, (a, b) in enumerate(zip(reference, record['sequence'])) if a != b]


def basic_features(records, reference):
    # Fixed coordinates derived only from the supplied reference / variant strings.
    ms = [mutations(r, reference) for r in records]
    positions = sorted(set(p for m in ms for p, a, b in m))
    tokens = sorted(set((p, b) for m in ms for p, a, b in m))
    pi = {p: i for i, p in enumerate(positions)}
    ti = {t: i for i, t in enumerate(tokens)}
    p = np.zeros((len(ms), len(positions)))
    t = np.zeros((len(ms), len(tokens)))
    for i, m in enumerate(ms):
        for pos, a, b in m:
            p[i, pi[pos]] += 1
            t[i, ti[pos, b]] += 1
    return p, t, np.asarray(positions), ms


def ridge_predict(kernel, y, n, regularization=0.25, mode='sqrt'):
    train = kernel[:n, :n]
    mean = train.mean(axis=0)
    grand = mean.mean()
    k = train - mean[:, None] - mean[None, :] + grand
    # Monotone transforms are alternative latent combination scales; no WT
    # calibration or probability interpretation is assumed. Floors only define
    # these transforms outside the observed nonnegative assay range.
    if mode == 'sqrt':
        target = np.sqrt(np.maximum(y, 0) + .01)
    else:
        target = np.log(np.maximum(y, 0) + float(mode[3:]))
    scale = max(float(target.std()), 1e-12)
    alpha = solve(k + regularization * np.eye(n),
                  (target - target.mean()) / scale, assume_a='pos', check_finite=False)
    cross = kernel[n:, :n]
    cross = cross - cross.mean(axis=1)[:, None] - mean[None, :] + grand
    return cross @ alpha


def model_kernels(records, reference):
    p, _, positions, ms = basic_features(records, reference)
    pi = {pos: i for i, pos in enumerate(positions)}
    z = np.zeros((len(records), len(positions), 30))
    for i, m in enumerate(ms):
        for pos, a, b in m:
            source, target = PROPERTIES[AA.index(a)], PROPERTIES[AA.index(b)]
            delta = target - source
            z[i, pi[pos]] += np.r_[delta, np.abs(delta), target]
    flat = z.reshape(len(records), -1) / np.sqrt(30)
    global_features = z.sum(1) / np.sqrt(30)
    site = p @ p.T
    smooth = p @ np.exp(-.5 * ((positions[:, None] - positions[None, :]) / 1.5) ** 2) @ p.T
    chemistry = flat @ flat.T
    general = global_features @ global_features.T
    base = site + .3 * smooth + 3 * chemistry + .1 * general
    distance = np.maximum(np.diag(base)[:, None] + np.diag(base)[None, :] - 2 * base, 0)
    # RBF residual kernels allow combination-dependent effects beyond additive
    # mutation effects. They imply similarity, not experimentally known contacts.
    return [base, base, base + 3 * np.exp(-distance / 6),
            base + 3 * np.exp(-distance / 20),
            site + smooth + 10 * chemistry + .1 * general]


def propose(case, assignment):
    support, query = case['support'], case['query']
    if not query:
        return {'scores': {}, 'selected_ids': []}
    if not support:
        return {'scores': {r['id']: 0.0 for r in query}, 'selected_ids': []}
    # Vocabulary includes queries only to represent their observed mutations.
    # Every fitted coefficient, centering mean, and target scale uses supports.
    # No persistent random generator or model state: computation is deterministic.
    y = np.asarray([r['fitness'] for r in support], dtype=float)
    kernels = model_kernels(support + query, case['reference_sequence'])
    # Broadly competitive configurations from repeated held-out Spearman tests.
    # Equal averaging avoids choosing one narrow optimum from just 96 outcomes.
    settings = [(.25, 'sqrt'), (1, 'log0.1'), (.05, 'sqrt'),
                (.25, 'log0.3'), (.5, 'sqrt')]
    pred = np.mean([ridge_predict(k, y, len(support), a, mode)
                    for k, (a, mode) in zip(kernels, settings)], axis=0)
    scores = 2 * (rankdata(pred) - 0.5) / len(query) - 1
    return {'scores': {r['id']: float(s) for r, s in zip(query, scores)}, 'selected_ids': []}
