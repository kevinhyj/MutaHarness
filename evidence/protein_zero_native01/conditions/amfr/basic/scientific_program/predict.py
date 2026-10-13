"""Unsupervised, sequence-only stability ranking for the supplied AMFR case.

No assay observations, homologs, coordinates, or external data are used.
The checkpoint model uses generic helix and hydrophobic compatibility.
The final model averages a small predetermined set of helix-boundary and
packing assumptions; parameters are physical priors, never outcome-fitted.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Kyte-Doolittle hydropathy and Pace-Scholtz helix free-energy propensities.
HYDRO = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                 [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
HELIX = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                 [0,.68,.69,.40,.54,1,.61,.41,.26,.21,.24,.65,3.16,.39,.21,.50,.66,.61,.49,.53]))
VOLUME = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                  [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89,116.1,140,227.8,193.6]))
CHARGE = {a: (1 if a in 'KR' else -1 if a in 'DE' else .15 if a == 'H' else 0) for a in HYDRO}
# Side-chain octanol/water partition hydrophobicities (Fauchere-Pliska).
# Unlike max(Kyte-Doolittle, 0), these represent aromatic core compatibility.
TRANSFER = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                   [.31,1.54,-.77,-.64,1.79,0,.13,1.8,-.99,1.7,1.23,-.6,.72,-.22,-1.01,-.04,.26,1.22,2.25,.96]))
POLAR = set('DENQSTHKR')

# Each is an approximate three-helix CUE fold, not a coordinate observation.
HELIX_SCENARIOS = [((4,17),(22,30),(32,46)),
                   ((5,17),(23,31),(34,45)),
                   ((5,16),(22,32),(35,44)),
                   ((6,17),(23,29),(31,46))]


def load_and_validate():
    case = json.loads(Path('case.json').read_text())
    manifest = json.loads(Path('evidence_manifest.json').read_text())
    wt = case['reference_sequence']
    assert case['context']['higher_is_better'] is True
    assert not manifest['measurements_present'] and manifest['support_count'] == 0
    assert manifest['allowed_modalities'] == {'msa': False, 'structure': False}
    ids = [q['id'] for q in case['query']]
    assert len(ids) == len(set(ids)) == 250
    assert set(ids) == {f'M{i:03d}' for i in range(1, 251)}
    for q in case['query']:
        mutant = list(wt)
        mutations = q['mutation'].split(':')
        assert len(mutations) == q['mutation_order']
        positions = set()
        for mutation in mutations:
            a, p, b = mutation[0], int(mutation[1:-1]), mutation[-1]
            assert p not in positions and wt[p-1] == a and a != b and b in HYDRO
            positions.add(p)
            mutant[p-1] = b
        assert ''.join(mutant) == q['sequence']
    return case, manifest


def helix_burial(wt, helices):
    """Amphipathic face inferred from WT sequence; not solvent accessibility."""
    burial = {p: .15 for p in range(1, len(wt)+1)}
    helical = set()
    for start, end in helices:
        vector = 0j
        for p in range(start, end+1):
            angle = math.radians(100*(p-start))
            vector += max(HYDRO[wt[p-1]], 0)*np.exp(1j*angle)
        phase = np.angle(vector)
        for p in range(start, end+1):
            angle = math.radians(100*(p-start))
            burial[p] = .15+.65*(1+math.cos(angle-phase))/2
            helical.add(p)
    return burial, helical


def checkpoint_cost(wt, p, mutant, burial, helical):
    native = wt[p-1]
    b = burial[p]
    hydrophobic_loss = (max(HYDRO[native],0)-max(HYDRO[mutant],0))/4
    cost = .8*b*hydrophobic_loss
    cost += .35*b*abs(VOLUME[mutant]-VOLUME[native])/70
    cost += .5*b*max(0, abs(CHARGE[mutant])-abs(CHARGE[native]))
    if p in helical:
        cost += .45*(HELIX[mutant]-HELIX[native])
        if mutant == 'P':
            cost += 1
    elif native in 'GP' and mutant != native:
        cost += .4
    cost += .12
    return float(cost)


def predict_checkpoint(case):
    wt = case['reference_sequence']
    burial, helical = helix_burial(wt, [(5,17),(22,32),(35,44)])
    output = []
    for q in case['query']:
        cost = sum(checkpoint_cost(wt, int(m[1:-1]), m[-1], burial, helical)
                   for m in q['mutation'].split(':'))
        output.append({'id': q['id'], 'score': float(np.tanh(-cost/2))})
    return {'predictions': output, 'summary':
            'Initial sequence-only stability prior: additive hydrophobic packing, side-chain volume, '
            'charge and helix propensity, using approximate AMFR CUE helices. Higher scores predict '
            'greater stability; one global tanh scale. No measured supports, WT baseline, MSA, '
            'coordinates or partners are supplied. Helical burial is inferred, not observed; '
            'assay effects and epistasis are highly uncertain.'}


def verify(result, case):
    assert {q['id'] for q in result['predictions']} == {q['id'] for q in case['query']}
    assert len(result['predictions']) == 250
    assert all(math.isfinite(q['score']) and -1 <= q['score'] <= 1 for q in result['predictions'])


def core_prior(wt, helices):
    """Sequence-inferred packing in a compact bundle with multiple interfaces.

    A single hydrophobic moment misses a helix's second packing face. Same-face
    i+/-3,4 native hydrophobic neighbors and the native side chain are therefore
    also used. All values represent uncertain burial, not measured exposure.
    """
    face, helical = helix_burial(wt, helices)
    burial = {p: .12 for p in range(1, len(wt)+1)}
    for start, end in helices:
        for p in range(start, end+1):
            neighbors = [q for q in (p-4,p-3,p+3,p+4) if start <= q <= end]
            support = np.mean([min(1,max(0,TRANSFER[wt[q-1]])/1.3)
                               for q in neighbors]) if neighbors else 0
            own = min(1,max(0,TRANSFER[wt[p-1]])/1.3)
            local = .18+.47*support+.22*own
            burial[p] = float(.45*face[p]+.55*local)
            # Hydrophobes may belong to either of the bundle's packing faces.
            burial[p] = max(burial[p], .18+.50*own)
    # Aromatic/hydrophobic termini may cap the bundle; their packing is less
    # certain than internal helices, so keep this effect modest.
    for p in range(1, len(wt)+1):
        if p not in helical:
            burial[p] = .12+.20*min(1,max(0,TRANSFER[wt[p-1]])/1.3)
    return burial, helical


def hydrogen_bond_retention(a, b):
    if a == b:
        return 1.
    if a in 'DE':
        return .90 if b in 'DE' else .65 if b in 'NQ' else .4 if b in 'STH' else .05 if b in 'WY' else 0.
    if a in 'NQ':
        return .8 if b in 'NQ' else .65 if b in 'HDE' else .45 if b in 'ST' else .25 if b in 'WYKR' else 0.
    if a in 'ST':
        return .9 if b in 'ST' else .55 if b in 'NQH' else .4 if b in 'DEY' else .2 if b in 'CWKR' else 0.
    if a == 'H':
        return .6 if b in 'NQ' else .4 if b in 'DEKRSTWY' else 0.
    if a in 'KR':
        return .9 if b in 'KR' else .45 if b in 'HNQ' else .25 if b in 'STWY' else 0.
    return 1.


def physical_cost(wt, p, b, burial, helical, polar_weight):
    a = wt[p-1]
    core = burial[p]
    delta_volume = VOLUME[b]-VOLUME[a]
    # Cavities and clashes are asymmetric; allow modest local repacking by
    # damping rather than assuming that the WT backbone remains perfectly rigid.
    cavity = max(0,-delta_volume)/60
    clash = max(0,delta_volume-20)/90
    packing = core*(.95*cavity+1.05*clash**1.3)
    hydrophobic = -.65*core*(TRANSFER[b]-TRANSFER[a])
    # Weak solvent cost for creating an exposed hydrophobic patch in a small
    # domain. This is not a predicted binding or aggregation measurement.
    surface = .12*(1-core)*max(0,TRANSFER[b]-max(TRANSFER[a],.3))
    polar = 0.
    if a in POLAR:
        # The favorable desolvation term above is discounted when the WT polar
        # residue may satisfy a structural hydrogen bond. Compensated vs mildly
        # frustrated WT sites are both included in the fixed prior ensemble.
        polar = polar_weight*core*(1-hydrogen_bond_retention(a,b))
        if (a in 'DE' and b not in 'DE') or (a in 'KR' and b not in 'KR'):
            polar += .25*core
    charge = 1.55*core*max(0, abs(CHARGE[b])-abs(CHARGE[a]))
    if CHARGE[a]*CHARGE[b] < 0:
        charge += .55*core
    # Natural core hydrophobes replaced by uncharged polar side chains have
    # uncompensated polar atoms in addition to loss of hydrophobic transfer.
    if TRANSFER[a] > .9 and b in POLAR and abs(CHARGE[b]) < .5:
        polar += .35*core
    backbone = 0.
    if p in helical:
        backbone = .75*(HELIX[b]-HELIX[a])
        if b == 'P':
            backbone += 1.4
        if b == 'G':
            backbone += .35
    elif a in 'GP':
        # Loss of a turn-compatible residue; boundaries are uncertain.
        backbone = .40 if a == 'G' else .45
        if b in 'GPAST':
            backbone *= .5
    elif b == 'P':
        backbone = .25
    # Reduced side-chain entropy can stabilize Ala; cavity formation remains
    # explicit. Avoid declaring all tiny substitutions stabilizing.
    entropy = -.10 if b == 'A' and a not in 'GP' else 0.
    return float(packing+hydrophobic+surface+polar+charge+backbone+entropy)


def local_pair_cost(wt, mutant, mutations, burial, helices):
    """Strongly shrunk local charge effects, with no interhelix contact claims."""
    total = 0.
    positions = sorted(mutations)
    for i, p in enumerate(positions):
        for q in positions[i+1:]:
            if q-p not in (3,4):
                continue
            if not any(start <= p < q <= end for start,end in helices):
                continue
            native = CHARGE[wt[p-1]]*CHARGE[wt[q-1]]
            changed = CHARGE[mutant[p-1]]*CHARGE[mutant[q-1]]
            # Weak same-face electrostatic prior; unknown side-chain geometry
            # prevents claiming an actual salt bridge or validated epistasis.
            total += .25*math.sqrt(burial[p]*burial[q])*(changed-native)
    return total


def predict_final(case):
    wt = case['reference_sequence']
    ensemble = []
    for helices in HELIX_SCENARIOS:
        burial, helical = core_prior(wt, helices)
        for polar_weight in (1.0, 1.8, 2.6):
            ensemble.append((helices, burial, helical, polar_weight))
    predictions, diagnostics = [], []
    for q in case['query']:
        mutations = {int(m[1:-1]):m[-1] for m in q['mutation'].split(':')}
        costs = []
        for helices, burial, helical, polar_weight in ensemble:
            cost = sum(physical_cost(wt,p,b,burial,helical,polar_weight)
                       for p,b in mutations.items())
            cost += local_pair_cost(wt,q['sequence'],mutations,burial,helices)
            costs.append(cost)
        cost = float(np.mean(costs))
        score = float(np.tanh(-cost/3.))
        predictions.append({'id':q['id'],'score':round(score,8)})
        diagnostics.append({'id':q['id'],'mutation':q['mutation'],
                            'mean_stability_cost':cost,'assumption_sd':float(np.std(costs)),
                            'scenario_costs':costs})
    result = {'predictions': predictions, 'summary':
              'Sequence-only AMFR stability prior combining hydrophobic transfer, packing/cavity '
              'costs, polar/charge compatibility and helix propensity, averaged over plausible '
              'CUE helix boundaries and hydrogen-bond assumptions. Higher scores predict greater '
              'stability on one global tanh scale. All 250 mutation sequences were verified; input '
              'SHA-256 hashes and diagnostics are saved. No assay outcomes, MSA, coordinates or '
              'fixed partners are supplied. Burial and local interactions are uncertain priors; '
              'epistasis and proteolysis-specific effects remain unvalidated.'}
    return result, diagnostics


if __name__ == '__main__':
    case, manifest = load_and_validate()
    result = predict_checkpoint(case)
    verify(result, case)
    if not Path('checkpoint_predictions.json').exists():
        Path('checkpoint_predictions.json').write_text(json.dumps(result, indent=2)+'\n')
    audit = {'inputs': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in ['case.json','evidence_manifest.json']},
             'evidence': {'assay_supports': 0, 'msa': 'absent', 'structure': 'absent',
                          'fixed_partners': 'none supplied', 'WT_residues': len(case['reference_sequence'])},
             'validation': 'All mutation identities, sequences, orders, IDs, direction and finite range pass.',
             'rationale': ['Initial checkpoint uses additive sequence-based stability priors.',
                           'Final model corrects clipped hydropathy, which mistreats aromatic packing.',
                           'Include hydrogen-bond losses and buried-charge costs rather than rewarding all polar-to-hydrophobic changes.',
                           'A three-helix bundle can have multiple packing faces; add local i+/-3,4 packing support instead of relying on a single hydrophobic moment.',
                           'Average four fixed helix-boundary scenarios and three polar-compensation strengths because geometry is absent; no coefficients are fitted.',
                           'Code review found native Lys/Arg omitted from polar-contact losses; include them consistently with acidic native residues. No outcome-based accuracy claims follow from validation.']}
    final, diagnostics = predict_final(case)
    verify(final, case)
    Path('final_predictions.json').write_text(json.dumps(final, indent=2)+'\n')
    Path('prediction_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    Path('prediction_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('Validated checkpoint and final predictions for all 250 candidates.')
