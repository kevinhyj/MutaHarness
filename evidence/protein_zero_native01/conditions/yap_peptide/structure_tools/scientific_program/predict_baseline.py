"""Initial zero-shot WW-domain checkpoint; no measured outcomes used."""
import json
import math
import re
from pathlib import Path

CASE = json.loads(Path('case.json').read_text())
PATTERN = re.compile(r'([A-Za-z0-9]+):([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])')

def verified_changes(q):
    chains = CASE['reference_chains']
    reconstructed = {c: list(s) for c, s in chains.items()}
    changes = []
    seen = set()
    for text in q['mutation'].split(';'):
        m = PATTERN.fullmatch(text)
        assert m, text
        c, w, p, a = m.groups()
        p = int(p)
        assert (c, p) not in seen and chains[c][p-1] == w and w != a
        seen.add((c, p))
        reconstructed[c][p-1] = a
        changes.append((c, p, w, a))
    assert len(changes) == q['mutation_order']
    reconstructed = {c: ''.join(s) for c, s in reconstructed.items()}
    assert reconstructed == q['mutated_chains']
    assert q['sequence'] == reconstructed['A']
    for c, s in CASE['context']['fixed_partner_chains'].items():
        assert reconstructed[c] == s
    return changes

def single_effect(p, w, a):
    hydrophobic = set('AVILMFWY')
    polar = set('STNQ')
    charged = set('DEKR')
    same_class = any(w in group and a in group for group in [hydrophobic, polar, set('DE'), set('KR')])
    effect = -0.12 if same_class else -0.35
    if p in (13, 24, 25, 35):
        effect -= 0.5 if a in 'FWY' else (1.8 if a in hydrophobic else 3.0)
    if p == 27:
        effect -= 0.2 if a in 'NQST' else 1.2
    if p in (9, 15, 26, 29) and w in hydrophobic and a not in hydrophobic:
        effect -= 0.7
    if a == 'P' and p in list(range(13,19))+list(range(23,29))+list(range(33,37)):
        effect -= 1.8
    if w == 'G' and a not in 'GAS':
        effect -= 0.5
    if p == 6:
        effect = {'K':0.8, 'R':0.55, 'H':0.3, 'N':0.3, 'A':0.2, 'G':0.15, 'E':0.0}.get(a, -0.05)
    if p == 30:
        effect = {'N':0.85, 'A':0.3, 'G':0.15, 'S':0.2, 'T':0.15, 'E':-0.05}.get(a, -0.25)
    if p == 19 and a == 'R':
        effect = 0.3
    return effect

def save_predictions(predictions, path, summary):
    expected = {q['id'] for q in CASE['query']}
    ids = [q['id'] for q in predictions]
    assert len(ids) == 250 and len(set(ids)) == 250 and set(ids) == expected
    assert all(math.isfinite(q['score']) and -1 <= q['score'] <= 1 for q in predictions)
    Path(path).write_text(json.dumps({'predictions': predictions, 'summary': summary}, indent=2)+'\n')

if __name__ == '__main__':
    assert CASE['context']['higher_is_better'] is True
    predictions = []
    for q in CASE['query']:
        effect = sum(single_effect(p, w, a) for c, p, w, a in verified_changes(q))
        predictions.append({'id':q['id'], 'score':round(math.tanh(effect/3), 6)})
    save_predictions(predictions, 'checkpoint_predictions.json',
        'Initial unsupervised WW-domain chemistry prior: aromatic core and peptide groove preservation, backbone compatibility, and plausible surface/turn stabilization. Fixed peptide P retained. One global monotone scale; no measured supports or MSA, so relative rankings and multi-mutant effects are uncertain.')
    print('Checkpoint saved and validated: 250 unique candidates, unchanged partner P, finite scores in [-1,1].')
