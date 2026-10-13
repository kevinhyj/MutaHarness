"""Align four RRM repeats within the supplied PAB1 reference, without external data.

The two RNP motifs anchor the mapping. Inter-motif and post-motif regions use
affine-gap global alignment. These are paralogous domains, not an external MSA.
"""
import json
from pathlib import Path

import numpy as np

from predict import AA, AI, B62


def affine_align(a, b, gap_open=9.0, gap_extend=1.0):
    n, m = len(a), len(b)
    dp = np.full((3, n+1, m+1), -np.inf)
    prev = np.zeros((3, n+1, m+1), dtype=np.int8)
    dp[0, 0, 0] = 0.0
    for i in range(1, n+1):
        dp[1, i, 0] = -gap_open - (i-1)*gap_extend
        prev[1, i, 0] = 0 if i == 1 else 1
    for j in range(1, m+1):
        dp[2, 0, j] = -gap_open - (j-1)*gap_extend
        prev[2, 0, j] = 0 if j == 1 else 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            v = dp[:, i-1, j-1]
            k = int(np.argmax(v))
            dp[0, i, j] = v[k] + B62[AI[a[i-1]], AI[b[j-1]]]
            prev[0, i, j] = k
            v = dp[:, i-1, j] - [gap_open, gap_extend, gap_open]
            k = int(np.argmax(v)); dp[1, i, j] = v[k]; prev[1, i, j] = k
            v = dp[:, i, j-1] - [gap_open, gap_open, gap_extend]
            k = int(np.argmax(v)); dp[2, i, j] = v[k]; prev[2, i, j] = k
    state = int(np.argmax(dp[:, n, m])); x, y = [], []
    i, j = n, m
    while i or j:
        old = state; state = int(prev[old, i, j])
        if old == 0:
            x.append(a[i-1]); y.append(b[j-1]); i -= 1; j -= 1
        elif old == 1:
            x.append(a[i-1]); y.append('-'); i -= 1
        else:
            x.append('-'); y.append(b[j-1]); j -= 1
    return ''.join(reversed(x)), ''.join(reversed(y))


def repeat_map(sequence):
    # RNP2 six-residue and RNP1 eight-residue motifs, 1-based full-length indices.
    motifs = {'RRM1': (40,79,112), 'RRM2': (128,166,199),
              'RRM3': (221,259,294), 'RRM4': (324,362,395)}
    target_start, target_rnp1, target_end = motifs['RRM2']
    mapping = {p: [] for p in range(126,200)}
    displays = []
    for name, (start, rnp1, end) in motifs.items():
        if name == 'RRM2':
            continue
        segments = [
            # Include two immediately preceding residues at lower confidence;
            # the six-residue RNP2 motif still supplies the alignment anchor.
            (sequence[target_start-3:target_start+5], sequence[start-3:start+5]),
            affine_align(sequence[target_start+5:target_rnp1-1], sequence[start+5:rnp1-1]),
            (sequence[target_rnp1-1:target_rnp1+7], sequence[rnp1-1:rnp1+7]),
            affine_align(sequence[target_rnp1+7:target_end], sequence[rnp1+7:end]),
        ]
        a = ''.join(x for x,y in segments); b = ''.join(y for x,y in segments)
        p, q = target_start-3, start-3
        for x, y in zip(a,b):
            if x != '-': p += 1
            if y != '-': q += 1
            if x != '-' and y != '-':
                assert sequence[p-1] == x and sequence[q-1] == y
                mapping[p].append({'domain': name, 'position': q, 'aa': y})
        displays.append({'domain': name, 'target': a, 'repeat': b})
    return mapping, displays


if __name__ == '__main__':
    sequence = json.loads(Path('case.json').read_text())['reference_sequence']
    mapping, displays = repeat_map(sequence)
    Path('internal_repeat_mapping.json').write_text(json.dumps(
        {'source': 'case.json reference_sequence only',
         'interpretation': 'weak intraprotein paralog prior; not public homolog MSA',
         'mapping': mapping, 'alignments': displays}, indent=2)+'\n')
    for d in displays:
        print(d['domain']); print(d['target']); print(d['repeat'])
    for p in range(128,200):
        print(p, sequence[p-1], ''.join(x['aa'] for x in mapping[p]),
              ','.join(str(x['position']) for x in mapping[p]))
