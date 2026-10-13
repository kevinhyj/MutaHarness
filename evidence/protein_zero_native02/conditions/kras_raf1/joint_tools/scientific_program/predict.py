"""Label-free ranking of KRAS variants against fixed RAF1-RBD.

No assay outcomes are used. The checkpoint uses complete supplied MSA ratios.
The final predictor combines tempered conservation, WT packing/backbone and
fixed-partner contact priors, plus bounded KRAS nucleotide-state knowledge.
All coefficients are fixed expert priors, never fitted or searched against
outcomes. Refinements and their rationale are recorded in model_notes.md.
"""
import hashlib
import json
import math
import re
from pathlib import Path


def load_and_validate():
    case = json.loads(Path('case.json').read_text())
    manifest = json.loads(Path('evidence_manifest.json').read_text())
    msa = json.loads(Path('msa_evidence.json').read_text())
    structure = json.loads(Path('structure_evidence.json').read_text())
    assert manifest['support_count'] == 0 and not manifest['measurements_present']
    assert case['context']['higher_is_better'] is True
    refs = case['reference_chains']
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in refs.items()}
    assert hashes == msa['reference_sequences_sha256']
    assert structure['reference_sequence_sha256'] == hashes['A']
    raw = structure['raw_coordinate_evidence']
    assert raw['reference_sequence_sha256'] == hashes['A']
    for name, entry in manifest['files'].items():
        if isinstance(entry, dict) and 'sha256' in entry:
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == entry['sha256']
    assert raw['raw_source']['sha256'] == manifest['files']['raw_evidence/6VJJ.pdb']['sha256']
    query = case['query']
    ids = [q['id'] for q in query]
    assert len(ids) == len(set(ids)) == 250
    assert set(ids) == {f'M{i:03d}' for i in range(1, 251)}
    assert msa['candidate_order'] == ids
    for q in query:
        seqs = {ch: list(seq) for ch, seq in refs.items()}
        seen = set()
        for token in q['mutation'].split(';'):
            match = re.fullmatch(r'([A-Z]):([A-Z])(\d+)([A-Z])', token)
            assert match is not None
            ch, wt, pos, alt = match.groups()
            pos = int(pos)
            assert (ch, pos) not in seen and seqs[ch][pos - 1] == wt and wt != alt
            seen.add((ch, pos))
            seqs[ch][pos - 1] = alt
        expected = {ch: ''.join(seq) for ch, seq in seqs.items()}
        assert expected == q['mutated_chains']
        assert q['sequence'] == expected['A']
        assert len(seen) == q['mutation_order']
        for ch, seq in case['context']['fixed_partner_chains'].items():
            assert expected[ch] == seq == refs[ch]
    for packet in [msa, structure]:
        assert [row['id'] for row in packet['candidates']] == ids
        for q, row in zip(query, packet['candidates']):
            assert q['mutation'] == row['mutation']
    return case, manifest, msa, structure


def checkpoint(case, msa):
    predictions = []
    for q, row in zip(case['query'], msa['candidates']):
        complete = row['scores']['sum_log_frequency_ratio']
        if complete['complete']:
            value = complete['score']
            assert value is not None
        else:
            # Missing terms receive a broad unfavorable prior, rather than
            # interpreting a partial sum as a complete evolutionary score.
            value = sum(t['sum_log_frequency_ratio'] if t['sum_log_frequency_ratio'] is not None
                        and (t.get('non_gap_depth') or 0) > 0 else -4.0
                        for t in row['substitution_terms'])
        predictions.append({'id': q['id'], 'score': round(math.tanh(value / 18.0), 7)})
    output = {'predictions': predictions, 'summary':
              'Initial zero-shot checkpoint: additive supplied natural-homolog log-frequency ratios, '
              'one global tanh scale with higher scores better. All mutation identities, fixed RAF1-RBD '
              'sequences and source hashes verified. No assay outcomes; conservation is an uncertain '
              'binding prior and does not model activation, mutant structure or epistasis.'}
    validate_output(output, case)
    Path('checkpoint_predictions.json').write_text(json.dumps(output, indent=2) + '\n')


def validate_output(output, case):
    rows = output['predictions']
    assert len(rows) == 250
    assert [r['id'] for r in rows] == [q['id'] for q in case['query']]
    assert len({r['id'] for r in rows}) == 250
    assert all(math.isfinite(r['score']) and -1 <= r['score'] <= 1 for r in rows)


# Approximate residue volumes and nonpolar character are biochemical priors,
# not an energy force field. No mutant coordinates or rotamers are generated.
VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                  [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,
                   162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
NONPOLAR = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                    [1,.8,0,0,1,.5,.35,1,.05,1,1,0,.8,0,0,.1,.3,1,.9,.8]))
CHARGE = {aa: (-1 if aa in 'DE' else 1 if aa in 'KR' else 0) for aa in VOLUME}
CHARGED_ATOMS = {'D': ['OD1','OD2'], 'E': ['OE1','OE2'],
                 'R': ['NE','NH1','NH2'], 'K': ['NZ']}


def chemical_distance(wt, alt):
    """Bounded, conservative-residue-aware side-chain disruption descriptor."""
    volume = min(1.5, abs(VOLUME[alt] - VOLUME[wt]) / 100)
    polarity = abs(NONPOLAR[alt] - NONPOLAR[wt])
    charge = abs(CHARGE[alt] - CHARGE[wt]) / 2
    aromatic = float((wt in 'FYW') != (alt in 'FYW'))
    return .35*volume + .35*polarity + .45*charge + .3*aromatic


def electrostatic_delta(index, chain, position, wt, alt):
    """WT terminal-charge contacts give a bounded electrostatic prior.

    Distances refer ONLY to existing WT atoms. Assigning mutant residue charge
    does not imply its charge has the same coordinates; the estimate is weak.
    """
    import numpy as np
    if wt not in CHARGED_ATOMS or CHARGE[alt] == CHARGE[wt]:
        return 0.0
    residue = index[(chain, position)]
    result = 0.0
    for (pch, pp), partner in index.items():
        if pch != 'B' or partner['WT'] not in CHARGED_ATOMS:
            continue
        distances = [np.linalg.norm(np.array(residue['atoms'][a]['coord']) -
                                     partner['atoms'][b]['coord'])
                     for a in CHARGED_ATOMS[wt] for b in CHARGED_ATOMS[partner['WT']]]
        distance = float(min(distances))
        strength = max(0.0, min(1.0, (4.3 - distance) / 1.3))
        result -= strength * (CHARGE[alt] - CHARGE[wt]) * CHARGE[partner['WT']]
    # Relief of WT like-charge proximity is particularly uncertain.
    return max(-2.8, min(.5, result))


def activation_prior(canonical_pos, alt):
    """State-occupancy priors, uncertain because nucleotide state is unspecified.

    These are general KRAS mechanisms, not measured examples for this assay.
    Inhibiting GTP hydrolysis or increasing exchange can favor RAF engagement;
    direct interface/stability defects remain separately unfavorable.
    """
    if canonical_pos == 12:
        return {'C': 2.5, 'D': 3.0, 'V': 3.6, 'M': 2.8}.get(alt, 0.0)
    if canonical_pos == 61:
        return {'L': 3.5, 'I': 3.2, 'W': 2.8, 'N': 2.4, 'E': 2.2}.get(alt, 0.0)
    if canonical_pos == 28:
        return {'L': 1.6, 'I': 1.4, 'V': 1.4}.get(alt, 0.0)
    if canonical_pos == 117 and alt == 'T':
        return .8
    if canonical_pos == 146 and alt == 'F':
        # A146 substitutions may increase exchange, but bulky F is much less
        # secure than the established small activating substitutions.
        return .6
    return 0.0


def nucleotide_penalty(canonical_pos, alt):
    penalties = {
        10: {'L': 1.0, 'R': 1.0, 'S': .6},
        11: {'P': 1.3},
        13: {'W': 1.6},
        14: {'S': 1.2},
        15: {'P': 1.4, 'T': 1.4},
        16: {'W': 2.5, 'Y': 2.5},
        17: {'N': 2.8, 'W': 3.0, 'C': 1.5},
        28: {'K': 2.0, 'T': 1.6},
        35: {'S': .5},
        57: {'L': 2.2, 'W': 2.2},
        59: {'F': 1.0, 'H': .8, 'N': .3},
        60: {'F': 1.5, 'W': 1.5},
        117: {'T': 1.3},
        119: {'M': 3.0},
        146: {'F': 1.3},
    }
    return penalties.get(canonical_pos, {}).get(alt, 0.0)


def mutation_terms(term, raw_site, coordinate_index):
    from inspect_evidence import torsions
    chain, p, wt, alt = term['chain'], term['position'], term['WT'], term['ALT']
    assert chain == 'A' and raw_site['WT'] == wt
    canonical = p - 1  # Verified exact sequence mapping: initial construct Gly.
    rsa = raw_site.get('isolated_chain_wt_RSA')
    has_geometry = raw_site.get('coordinate_present') and rsa is not None
    # Unknown geometry never means exposed. This fallback is unused here.
    burial = math.exp(-max(0.0, rsa) / .18) if has_geometry else .35
    bsa = raw_site.get('observed_atom_SASA_buried_by_context_A2')
    interface = 1 - math.exp(-max(0.0, bsa) / 45) if bsa is not None else 0.0
    nucleotide_motif = (10 <= canonical <= 17 or canonical in
                        [28,35,57,59,60,61,116,117,119,145,146,147])
    lr = term.get('sum_log_frequency_ratio')
    valid_msa = lr is not None and (term.get('non_gap_depth') or 0) > 0
    # Preserve supplied ratios exactly in diagnostics. Temper the evolutionary
    # prior for phenotype relevance; floor magnitudes reflect finite sampling.
    evolutionary_weight = .13 + .1*burial + .08*interface + .07*nucleotide_motif
    if canonical >= 165:
        evolutionary_weight *= .5  # terminal conservation is weak RAF evidence
    evolution = evolutionary_weight * max(-8.0, min(.8, lr if valid_msa else -4.0))

    delta_volume = VOLUME[alt] - VOLUME[wt]
    polarity_loss = max(0.0, NONPOLAR[wt] - NONPOLAR[alt])
    core_damage = 2.0*polarity_loss
    if CHARGE[alt] != 0 and CHARGE[wt] == 0:
        core_damage += 1.5
    if wt in 'DNQEHKR' and alt in 'AVILMFWY':
        core_damage += 1.2  # possible loss of buried polar network
    core_damage += (1.1*max(0.0,-delta_volume) + 1.45*max(0.0,delta_volume))/100
    core = -burial * core_damage

    phi, psi = torsions(coordinate_index, chain, p)
    backbone_damage = 0.0
    if wt == 'G' and phi is not None and phi > 0:
        backbone_damage += (.8 if alt in 'AS' else 2.2) * (.45 + .55*burial)
    if alt == 'P':
        incompatible_phi = phi is not None and not -100 <= phi <= -40
        helical = phi is not None and psi is not None and -100 < phi < -30 and -80 < psi < 0
        beta = phi is not None and psi is not None and phi < -90 and psi > 70
        backbone_damage += (2.2 if incompatible_phi else 0.0)
        backbone_damage += (2.4 if helical else 1.0 if beta else .35) * (.55 + .45*burial)
    if alt == 'G' and wt != 'G' and phi is not None and psi is not None and -90 < phi < -35 and -75 < psi < 0:
        backbone_damage += .7 * (.5 + .5*burial)
    if wt == 'P' and alt != 'P':
        backbone_damage += .25 + .35*burial
        if canonical == 34:
            backbone_damage += .55  # switch-I backbone organization
    backbone = -backbone_damage

    distance = chemical_distance(wt, alt)
    interface_loss = interface * (.9*distance + .6*polarity_loss)
    if wt in 'FYW' and alt not in 'FYW':
        interface_loss += .65*interface
    # Explicit fixed RAF1 chain B charge contacts; no generic fold-to-affinity
    # substitution is made here.
    electrostatic = electrostatic_delta(coordinate_index, chain, p, wt, alt)
    binding = -interface_loss + electrostatic
    function = -nucleotide_penalty(canonical, alt)
    activation = activation_prior(canonical, alt)
    return {'chain': chain, 'position': p, 'canonical_KRAS_position': canonical,
            'WT': wt, 'ALT': alt, 'complete_msa_term': valid_msa,
            'supplied_log_frequency_ratio': lr, 'msa_non_gap_depth': term.get('non_gap_depth'),
            'msa_gap_fraction': term.get('gap_fraction'),
            'raw_coordinate_present': raw_site.get('coordinate_present'),
            'isolated_WT_RSA': rsa, 'WT_RAF_burial_A2': bsa,
            'WT_phi': phi, 'WT_psi': psi, 'evolution': evolution,
            'core': core, 'backbone': backbone, 'binding': binding,
            'nucleotide_function': function, 'activation': activation}


def final_predictions(case, msa, structure):
    from inspect_evidence import coordinates
    coordinate_index, mappings = coordinates(case)
    assert mappings == structure['raw_coordinate_evidence']['mapping']
    raw_index = {(r['chain'], r['position']): r for r in structure['raw_coordinate_evidence']['sites']}
    predictions, diagnostics = [], []
    for q, candidate in zip(case['query'], msa['candidates']):
        terms = [mutation_terms(t, raw_index[(t['chain'],t['position'])], coordinate_index)
                 for t in candidate['substitution_terms']]
        damage = math.fsum(t[k] for t in terms for k in
                           ['evolution','core','backbone','binding','nucleotide_function'])
        # At known G12/Q61 activators, natural negative conservation chiefly
        # concerns regulated hydrolysis, whereas RAF occupancy favors GTP.
        # Aggregate their net state effect once, keeping EVERY core, backbone,
        # interface and nucleotide-competence term. This avoids summing two
        # conservation losses for the same hydrolysis route in double activators.
        state_terms = [t for t in terms if t['canonical_KRAS_position'] in [12,61]
                       and t['activation'] > 0]
        if state_terms:
            damage -= math.fsum(t['evolution'] for t in state_terms)
            net_state = max(t['activation'] + t['evolution'] for t in state_terms)
        else:
            net_state = max((t['activation'] for t in terms), default=0.0)
        # Hydrolysis inhibition requires nucleotide occupancy. Conservatively
        # attenuate its benefit in the presence of an independent pocket defect.
        # This smooth gate is an expert assumption, not measured epistasis.
        competence = math.exp(math.fsum(t['nucleotide_function'] for t in terms) / 2.0)
        activation = net_state * competence
        latent = damage + activation
        score = round(math.tanh(latent / 7.0), 7)
        predictions.append({'id': q['id'], 'score': score})
        diagnostics.append({'id': q['id'], 'mutation': q['mutation'], 'terms': terms,
                            'damage_prior_sum': damage, 'activation_prior': activation,
                            'net_state_prior_before_competence': net_state,
                            'nucleotide_competence_gate': competence,
                            'latent_relative_performance': latent, 'score': score})
    output = {'predictions': predictions, 'summary':
              'Zero-shot KRAS–fixed RAF1-RBD ranking combines tempered supplied MSA log-ratios, '
              'WT burial/backbone and RAF1 contact priors, and bounded KRAS activation/nucleotide-binding '
              'knowledge. Source hashes, sequence-to-coordinate mapping, mutation identities and unchanged '
              'RAF1 were verified; all candidate MSA terms and WT sites are covered. One global tanh scale, '
              'higher is better. No assay outcomes: scores are uncalibrated, with uncertain nucleotide-state '
              'effects, modeled WT geometry, mutant rearrangements and epistasis.'}
    validate_output(output, case)
    Path('final_predictions.json').write_text(json.dumps(output, indent=2) + '\n')
    Path('prediction_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    return output, diagnostics


if __name__ == '__main__':
    case, manifest, msa, structure = load_and_validate()
    checkpoint(case, msa)
    print('Saved and validated checkpoint_predictions.json (250 candidates).')
    output, diagnostics = final_predictions(case, msa, structure)
    print('Saved and validated final_predictions.json (250 candidates).')
    print('Score range:', min(r['score'] for r in output['predictions']),
          max(r['score'] for r in output['predictions']))
    for group, rows in [('Highest', sorted(diagnostics,key=lambda r:-r['score'])[:12]),
                        ('Lowest', sorted(diagnostics,key=lambda r:r['score'])[:12])]:
        print(group)
        for row in rows:
            print(row['id'], row['mutation'], row['score'])
