"""Unsupervised, sequence-only PAB1 RRM2 variant prioritization.

No assay data or outside files are used. Scores are relative heuristics.
Run with --checkpoint for the initial conservative-substitution model.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

AA = 'ARNDCQEGHILKMFPSTWYV'
# Standard BLOSUM62 log-odds substitution matrix (not an assay fit).
B62 = np.array([
[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
], dtype=float)
AI = {a: i for i, a in enumerate(AA)}


def load_case():
    case = json.loads(Path('case.json').read_text())
    manifest = json.loads(Path('evidence_manifest.json').read_text())
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better'] is True
    wt = case['reference_sequence']
    candidates = case['query']
    assert len(candidates) == 250
    assert {q['id'] for q in candidates} == {f'M{i:03d}' for i in range(1, 251)}
    mutations = {}
    for q in candidates:
        muts = []
        reconstructed = list(wt)
        for token in q['mutation'].split(':'):
            m = re.fullmatch(r'([A-Z])(\d+)([A-Z])', token)
            assert m, token
            a, p, b = m.group(1), int(m.group(2)), m.group(3)
            assert a == wt[p-1] and a in AA and b in AA and a != b
            reconstructed[p-1] = b
            muts.append((p, a, b))
        assert len(muts) == q['mutation_order']
        assert len({p for p, a, b in muts}) == len(muts)
        assert ''.join(reconstructed) == q['sequence']
        mutations[q['id']] = muts
    return case, manifest, mutations


def substitution_distance(a, b):
    return (0.5 * (B62[AI[a], AI[a]] + B62[AI[b], AI[b]])
            - B62[AI[a], AI[b]]) / 5.0


def checkpoint_cost(p, a, b):
    importance = 1.0
    if p in {128, 130, 133, 142, 146, 149, 152, 153, 183, 186, 189, 191, 196, 198}:
        importance = 1.25
    if p in {129, 131, 132, 166, 167, 168, 169, 170, 171, 172, 173}:
        importance = 1.55
    if p in range(157, 166) or p in {174, 175, 176, 177, 190, 192, 193, 194, 195, 199}:
        importance = 0.65
    cost = importance * substitution_distance(a, b)
    if b == 'P' and p in set(range(128,134)) | set(range(140,150)) | set(range(152,157)) | set(range(166,174)) | set(range(179,188)) | set(range(194,200)):
        cost += 2.0
    if p in {129,168,170} and b not in 'FYW':
        cost += 2.2
    return cost


def score_from_cost(cost):
    # A single smooth global scale. No measured WT value is assumed.
    return float(2.0 * math.exp(-cost / 3.5) - 1.0)


def write_predictions(path, candidates, costs, summary):
    out = {'predictions': [{'id': q['id'], 'score': score_from_cost(costs[q['id']])}
                           for q in candidates], 'summary': summary}
    assert len(out['predictions']) == 250
    assert len({p['id'] for p in out['predictions']}) == 250
    assert all(math.isfinite(p['score']) and -1 <= p['score'] <= 1
               for p in out['predictions'])
    Path(path).write_text(json.dumps(out, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    case, manifest, mutations = load_case()
    costs = {q['id']: sum(checkpoint_cost(*m) for m in mutations[q['id']])
             for q in case['query']}
    summary = ('Sequence-only unsupervised ranking using BLOSUM62 substitution '
               'severity, RRM RNA-binding motif importance, and backbone-disrupting '
               'proline penalties; mutation costs are additive on one global scale. '
               'No assay outcomes, MSA, or coordinates are supplied. Mutation identities '
               'and all 250 IDs were checked. Conservation/folding are imperfect growth '
               'and RNA-binding proxies; double-mutant epistasis is unvalidated.')
    path = 'checkpoint_predictions.json' if args.checkpoint else 'predictions.json'
    write_predictions(path, case['query'], costs, summary)
    print(f'Wrote {path}: 250 finite scores, higher = predicted better.')
    audit = {'case_sha256': hashlib.sha256(Path('case.json').read_bytes()).hexdigest(),
             'manifest_sha256': hashlib.sha256(Path('evidence_manifest.json').read_bytes()).hexdigest(),
             'reference_length': len(case['reference_sequence']),
             'verified_candidates': 250, 'msa_coverage': 0, 'structure_coverage': 0,
             'fixed_partner_sequences_supplied': False,
             'outside_mutated_domain_preserved': True}
    Path('input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
