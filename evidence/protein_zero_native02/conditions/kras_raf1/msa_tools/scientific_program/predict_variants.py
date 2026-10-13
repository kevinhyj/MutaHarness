"""Label-free KRAS/RAF1 ranking; no assay outcomes are used."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Standard BLOSUM62: a general conservative-substitution prior, not an
# experimentally measured KRAS phenotype or a smoothed MSA frequency.
AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'
BLOSUM_ROWS = '''
4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4
'''
BLOSUM = {(a, b): int(v) for a, line in zip(AA_ORDER, BLOSUM_ROWS.strip().splitlines())
          for b, v in zip(AA_ORDER, line.split())}


def mutation_prior(term):
    """Return basal compatibility and an uncertain active-state propensity.

    Values are fixed a priori on an arbitrary log-performance scale. Source
    positions are canonical KRAS positions +1. Mechanistic overrides replace
    the generic prior to avoid counting catalytic conservation as folding loss.
    """
    wt, alt = term['WT'], term['ALT']
    pos = term['position'] - 1
    lr = term['sum_log_frequency_ratio']
    observed = (lr is not None and term['non_gap_depth'] is not None
                and term['non_gap_depth'] > 0)
    evolutionary = (-0.14 * min(max(-lr, 0.0), 10.7)
                    + 0.04 * min(max(lr, 0.0), 3.0)) if observed else -0.42
    chemistry = -0.60 * max(0.0, (3.0 - BLOSUM[wt, alt]) / 6.0)
    basal = evolutionary + chemistry
    activation = 0.0
    reasons = ['fixed_msa_plus_conservative_substitution']

    # Sequence-derived, established Ras secondary-structure ranges; no
    # supplied coordinates, burial or RSA is asserted.
    helices = [(16, 25), (67, 74), (87, 104), (126, 138), (152, 166)]
    strands = [(5, 9), (38, 46), (50, 58), (77, 81), (116, 119), (145, 147)]
    if alt == 'P' and any(lo <= pos <= hi for lo, hi in helices):
        basal -= 0.8
        reasons.append('proline_in_expected_helix')
    elif alt == 'P' and any(lo <= pos <= hi for lo, hi in strands):
        basal -= 0.6
        reasons.append('proline_in_expected_strand')
    if alt == 'G' and wt != 'G' and any(lo <= pos <= hi for lo, hi in helices):
        basal -= 0.25
        reasons.append('glycine_in_expected_helix')

    # Canonical G12 and Q61 mutations can preserve the fold while favoring
    # GTP-bound Ras. Their natural rarity is not an assay loss measurement.
    if pos == 12 and alt in 'CDVM':
        basal, activation = {
            'C': (-0.08, 0.35), 'D': (-0.12, 0.55),
            'V': (-0.12, 0.60), 'M': (-0.25, 0.35)}[alt]
        reasons.append('G12_active_state_prior')
    elif pos == 61 and alt in 'LIWNE':
        basal, activation = {
            'L': (-0.10, 0.55), 'I': (-0.15, 0.45),
            'W': (-0.30, 0.40), 'N': (-0.25, 0.25),
            'E': (-0.35, 0.25)}[alt]
        reasons.append('Q61_active_state_prior')
    elif pos == 28 and alt in 'LVI':
        basal, activation = -0.25, 0.35
        reasons.append('F28_fast_exchange_prior')

    # RAF1 specificity is distinct from generic GTPase conservation.
    if pos == 37 and alt not in 'DE':
        basal = min(basal, -2.40)
        reasons.append('RAF_effector_interface_E37')
    if pos == 38:
        basal = -0.50 if alt == 'E' else min(basal, -2.60)
        reasons.append('RAF_effector_interface_D38')
    if pos == 40:
        basal = -0.55 if alt in 'FW' else min(basal, -2.50)
        reasons.append('RAF_effector_interface_Y40')
    if pos == 36 and alt in 'DEKR':
        basal = min(basal, -2.20)
        reasons.append('switch_I_charged_disruption')
    if pos == 35 and alt == 'S':
        basal = -0.40
        reasons.append('T35S_retains_RAF_preference_but_state_uncertain')
    if pos == 32 and alt == 'F':
        basal = -0.40
        reasons.append('conservative_switch_I_aromatic')

    # Mg/phosphate/guanine binding defects can reduce active occupancy.
    if ((pos == 15 and alt in 'PT') or (pos == 16 and alt in 'WY')
            or (pos == 17 and alt in 'NW') or (pos == 57 and alt in 'LW')
            or (pos == 60 and alt in 'FW') or (pos == 119 and alt == 'M')):
        basal = min(basal, -2.80)
        reasons.append('nucleotide_binding_defect_prior')

    # K117/A146 are exchange hotspots, but the supplied K117T/A146F
    # substitutions are not assumed to share validated activating phenotypes.
    if pos == 117 and alt == 'T':
        basal = -1.00
        reasons.append('exchange_hotspot_uncertain_K117T')
    if pos == 146 and alt == 'F':
        basal = -1.20
        reasons.append('exchange_hotspot_uncertain_A146F')

    # The source includes one artificial N-terminal residue and a short
    # extension past the usual catalytic domain; do not equate conservation
    # of the linker with a direct RAF1 contact.
    if pos == 2 and alt in 'KH':
        basal = -0.08 if alt == 'K' else -0.15
        reasons.append('N_terminal_tolerant_prior')
    if pos == 3 and alt in 'CM':
        basal = -0.25 if alt == 'C' else -0.15
        reasons.append('N_terminal_surface_prior')
    if pos >= 167:
        basal *= 0.30
        reasons.append('post_catalytic_domain_extension')
    return {'basal': basal, 'activation': activation, 'msa_observed': observed,
            'canonical_position': pos, 'reasons': reasons}


def load_and_validate():
    case = json.loads((ROOT / 'case.json').read_text())
    manifest = json.loads((ROOT / 'evidence_manifest.json').read_text())
    evidence = json.loads((ROOT / 'msa_evidence.json').read_text())
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better'] is True
    refs = case['reference_chains']
    assert evidence['reference_sequences_sha256'] == {
        c: hashlib.sha256(s.encode()).hexdigest() for c, s in refs.items()}
    for name, info in manifest['files'].items():
        if isinstance(info, dict) and 'sha256' in info:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == info['sha256']
    assert len(case['query']) == 250
    assert {q['id'] for q in case['query']} == {f'M{i:03d}' for i in range(1, 251)}
    index = {q['id']: q for q in evidence['candidates']}
    assert len(index) == 250
    for q in case['query']:
        changes = []
        for chain, ref in refs.items():
            seq = q['mutated_chains'][chain]
            assert len(seq) == len(ref)
            changes.extend((chain, i, a, b) for i, (a, b) in enumerate(zip(ref, seq), 1) if a != b)
        annotated = []
        for token in q['mutation'].split(';'):
            match = re.fullmatch(r'([A-Z]):([A-Z])(\d+)([A-Z])', token)
            assert match
            chain, wt, pos, alt = match.groups()
            annotated.append((chain, int(pos), wt, alt))
        assert sorted(changes) == sorted(annotated)
        assert len(changes) == q['mutation_order']
        assert q['sequence'] == q['mutated_chains']['A']
        for chain, seq in case['context']['fixed_partner_chains'].items():
            assert q['mutated_chains'][chain] == seq == refs[chain]
        terms = index[q['id']]['substitution_terms']
        assert sorted(changes) == sorted((t['chain'], t['position'], t['WT'], t['ALT']) for t in terms)
    return case, evidence, index


def write_predictions(predictions, summary, name):
    assert len(predictions) == 250
    assert {p['id'] for p in predictions} == {f'M{i:03d}' for i in range(1, 251)}
    assert all(math.isfinite(p['score']) and -1 <= p['score'] <= 1 for p in predictions)
    result = {'predictions': predictions, 'summary': summary}
    (ROOT / name).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    case, evidence, index = load_and_validate()
    predictions = []
    for q in case['query']:
        row = index[q['id']]
        agg = row['scores']['sum_log_frequency_ratio']
        if agg['complete'] and agg['score'] is not None:
            raw = agg['score']
        else:
            # Missing terms are assigned an explicitly unsupervised penalty;
            # observed partial sums are never called complete MSA scores.
            raw = sum(t['sum_log_frequency_ratio'] if t['sum_log_frequency_ratio'] is not None
                      else -3.0 for t in row['substitution_terms'])
        predictions.append({'id': q['id'], 'score': round(math.tanh(raw / 15.0), 8)})
    # Preserve the usable conservation-only checkpoint across later runs.
    if not (ROOT / 'checkpoint_predictions.json').exists():
        write_predictions(predictions,
        'Initial checkpoint: fixed natural-homolog log-frequency ratios on one global tanh scale; '
        'sequence identities, source hashes and unchanged RAF1 partner checked. '
        'No measured outcomes; conservation is an uncertain prior for KRAS–RAF1 binding.',
        'checkpoint_predictions.json')

    final, diagnostics = [], []
    for q in case['query']:
        priors = [mutation_prior(t) for t in index[q['id']]['substitution_terms']]
        basal_sum = sum(t['basal'] for t in priors)
        active_sum = sum(t['activation'] for t in priors)
        # A bounded occupancy bonus prevents double counting two mutations
        # acting through the same activation pathway. This is a mechanistic
        # assumption, not fitted or experimentally validated epistasis.
        active_bonus = 0.75 * (-math.expm1(-active_sum / 0.75))
        raw = basal_sum + active_bonus
        score = round(math.tanh(raw / 3.0), 8)
        final.append({'id': q['id'], 'score': score})
        diagnostics.append({'id': q['id'], 'mutation': q['mutation'],
                            'priors': priors, 'raw': raw, 'score': score})
    summary = (
        'Unsupervised KRAS–RAF1 ranking blends unchanged MSA log-frequency ratios, '
        'conservative-substitution priors and established RAF-interface/nucleotide-state biology. '
        'Canonical KRAS positions equal source positions minus 1; RAF1 is fixed. Source/sequence hashes verified; '
        'all 250 MSA scores are complete, using unweighted homologs with unknown Neff. '
        'One global tanh scale; higher is better. No structure, outcomes or calibrated mutant/epistasis model; '
        'activation effects and double-mutant ordering remain uncertain.')
    write_predictions(final, summary, 'final_predictions.json')
    (ROOT / 'prediction_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')


if __name__ == '__main__':
    main()
