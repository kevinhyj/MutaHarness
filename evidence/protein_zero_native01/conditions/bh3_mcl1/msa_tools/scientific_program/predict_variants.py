"""Label-free BH3/Mcl-1 ranking; all parameters are scientific priors, not fits."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_and_validate():
    case = json.loads((ROOT / 'case.json').read_text())
    manifest = json.loads((ROOT / 'evidence_manifest.json').read_text())
    msa = json.loads((ROOT / 'msa_evidence.json').read_text())
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better'] is True
    refs = case['reference_chains']
    for ch, seq in refs.items():
        assert hashlib.sha256(seq.encode()).hexdigest() == msa['reference_sequences_sha256'][ch]
    for name, meta in manifest['files'].items():
        if isinstance(meta, dict) and 'sha256' in meta:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == meta['sha256']
    queries = case['query']
    assert len(queries) == 250
    assert {q['id'] for q in queries} == {f'M{i:03d}' for i in range(1, 251)}
    assert len({q['id'] for q in queries}) == 250
    assert msa['candidate_order'] == [q['id'] for q in queries]
    evidence = {x['id']: x for x in msa['candidates']}
    for q in queries:
        changes = []
        for ch, wtseq in refs.items():
            altseq = q['mutated_chains'][ch]
            assert len(wtseq) == len(altseq)
            changes += [f'{ch}:{w}{p}{a}' for p, (w, a) in enumerate(zip(wtseq, altseq), 1) if w != a]
        assert set(changes) == set(q['mutation'].split(';'))
        assert len(changes) == q['mutation_order']
        assert q['sequence'] == q['mutated_chains']['C']
        for ch, seq in case['context']['fixed_partner_chains'].items():
            assert q['mutated_chains'][ch] == seq == refs[ch]
        assert evidence[q['id']]['mutation'] == q['mutation']
        for token in changes:
            ch, w, p, a = re.fullmatch(r'([^:]+):([A-Z])(\d+)([A-Z])', token).groups()
            assert w != 'X' and a in 'ACDEFGHIKLMNPQRSTVWY'
    return case, msa, evidence


def save_predictions(predictions, summary, filename):
    assert len(predictions) == 250 and len({x['id'] for x in predictions}) == 250
    assert all(math.isfinite(x['score']) and -1 <= x['score'] <= 1 for x in predictions)
    result = {'predictions': predictions, 'summary': summary}
    (ROOT / filename).write_text(json.dumps(result, indent=2) + '\n')
    return result


def checkpoint(case, evidence):
    preds = []
    for q in case['query']:
        score = evidence[q['id']]['scores']['sum_log_frequency_ratio']
        assert score['complete'] and score['coverage'] == 1 and score['score'] is not None
        preds.append({'id': q['id'], 'score': round(math.tanh((score['score'] + 8) / 10), 7)})
    return save_predictions(preds,
        'Checkpoint: additive supplied natural-homolog log-frequency ratios, with a single global monotonic scale. All candidate substitutions have observed homolog coverage; zero-depth flanking sites are excluded. Hashes, mutation identities and unchanged Mcl-1 partner verified. Conservation is an uncertain binding prior; no assay labels, mutant structures or validated epistasis.',
        'checkpoint_predictions.json')


# Fauchere-Pliska side-chain hydrophobicity; Zamyatnin residue volumes;
# Pace-Scholtz relative alpha-helix free energies. These descriptors are priors,
# not measured affinities, and the resulting burden has arbitrary relative units.
HYDRO = dict(zip('ARNDCQEGHILKMFPSTWYV',
                [.31, -1.01, -.60, -.77, 1.54, -.22, -.64, 0, .13,
                 1.80, 1.70, -.99, 1.23, 1.79, .72, -.04, .26, 2.25, .96, 1.22]))
VOLUME = dict(zip('ARNDCQEGHILKMFPSTWYV',
                 [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1,
                  153.2, 166.7, 166.7, 168.6, 162.9, 189.9, 112.7,
                  89.0, 116.1, 227.8, 193.6, 140.0]))
HELIX = dict(zip('ARNDCQEGHILKMFPSTWYV',
                [0, .21, .65, .69, .68, .39, .40, 1.00, .61, .41,
                 .21, .26, .24, .54, 3.16, .50, .66, .49, .53, .61]))
# Four BH3 hydrophobic positions; h2 Leu is the most stringent anchor.
# h3/h4 are more permissive in Mcl-1 than in a generic hydrophobic core.
# No actual mutant pocket sizes or contacts are asserted.
ANCHORS = {
    6: (.60, .004, .003, .60, .14),
    10: (.85, .008, .0075, .90, .22),
    13: (.45, .002, .003, .40, .14),
    17: (.45, .003, .004, .35, .14),
}


def binding_burden(term):
    p, w, a = term['position'], term['WT'], term['ALT']
    helix = .60 * (HELIX[a] - HELIX[w])
    # Proline in the bound BH3 helix disrupts backbone H-bonding even when
    # natural homolog frequencies occasionally admit it in other contexts.
    if a == 'P':
        helix = 3.5
    chemistry = 0.0
    evo_weight = .09
    if p in ANCHORS:
        hweight, cavity, bulk, charge, evo_weight = ANCHORS[p]
        loss = hweight * max(0.0, HYDRO[w] - HYDRO[a])
        if p in (13, 17) and a in 'NQ':
            loss *= .65
        chemistry = loss + cavity * max(0, VOLUME[w] - VOLUME[a])
        chemistry += bulk * max(0, VOLUME[a] - VOLUME[w])
        if a in 'DEKR':
            chemistry += charge
        if a == 'G':
            chemistry += .5
    elif p == 15:
        # BH3 Asp engages the conserved basic residue of the Bcl-2-family
        # groove. Glu preserves charge but not exact carboxylate geometry.
        chemistry = 1.0 if a == 'E' else (3.8 if a in 'KR' else 2.8)
        evo_weight = .18
    elif p in (7, 14):
        # Small motif residues: conservative Ala/Ser at Gly14 are plausible;
        # bulky replacements are disfavored, without inventing WT distances.
        allowance = VOLUME['A'] if p == 14 else VOLUME[w]
        chemistry = .010 * max(0, VOLUME[a] - allowance)
        if a in 'DEKR':
            chemistry += .3
        evo_weight = .12
    else:
        # Other assayed sites are outside the canonical four-anchor/Asp
        # pharmacophore; exact salt bridges are unknown without coordinates.
        chemistry = .15 if a in 'ILVFWY' else 0.0
    lr = term['sum_log_frequency_ratio']
    observed = term['non_gap_depth'] is not None and term['non_gap_depth'] > 0
    evolutionary = -evo_weight * max(-6.0, min(6.0, lr)) if observed and lr is not None else 0.0
    return {'chemistry': chemistry, 'helix': helix, 'evolutionary': evolutionary,
            'evolution_observed': bool(observed and lr is not None)}


def refined_predictions(case, evidence):
    predictions, diagnostics = [], []
    for q in case['query']:
        terms = evidence[q['id']]['substitution_terms']
        features = [binding_burden(t) for t in terms]
        burden = math.fsum(f[k] for f in features for k in ('chemistry', 'helix', 'evolutionary'))
        # One fixed strictly decreasing transform for every candidate, with
        # WT burden zero. No block normalization or mutation-count penalty.
        score = math.tanh((2.5 - burden) / 4.5)
        predictions.append({'id': q['id'], 'score': round(score, 7)})
        diagnostics.append({'id': q['id'], 'mutation': q['mutation'],
                            'binding_burden': burden, 'terms': features,
                            'msa_complete': evidence[q['id']]['scores']['sum_log_frequency_ratio']['complete']})
    (ROOT / 'prediction_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    return save_predictions(predictions,
        'Label-free BH3–Mcl-1 binding prior: four hydrophobic anchors, conserved Asp15, side-chain size/charge and helix propensity, plus downweighted, capped natural-homolog log-frequency ratios. Chain A stays fixed. Source hashes and residue mapping verified; all queried substitutions have homolog coverage (2048 unique unweighted rows, unknown Neff); zero-depth flanks contribute no evidence. One global monotonic scale, higher is better. No assay calibration, structure or validated epistasis; partner-specific tolerance and combination effects remain uncertain.',
        'final_predictions.json')


if __name__ == '__main__':
    case, msa, evidence = load_and_validate()
    if not (ROOT / 'checkpoint_predictions.json').exists():
        checkpoint(case, evidence)
    refined_predictions(case, evidence)
    print('Saved validated predictions and diagnostics for 250 candidates.')
