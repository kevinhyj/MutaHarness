"""Fixed, unsupervised biochemical model; no parameters are fit to outcomes.

Uses only case.json and evidence_manifest.json. RRM topology annotations below
are sequence/fold priors, not observed solvent accessibility or coordinates.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from predict import load_case, substitution_distance, write_predictions
from repeat_alignment import repeat_map


# Approximate buried-sidechain importance from canonical RRM topology and the
# reference's intraprotein repeats. Low values represent edge/ambiguous sites.
CORE = {
    128:.85, 129:.10, 130:.95, 133:.90, 137:.40,
    142:1.0, 146:1.0, 148:.45, 149:.80,
    152:.85, 153:.40, 154:.15, 155:.10, 157:.45,
    168:.10, 170:.10, 171:.80, 173:1.0,
    179:.25, 182:.85, 183:1.0, 185:.15, 186:.65,
    189:.15, 190:.10, 191:.60, 194:.25, 196:.75, 198:.65, 199:.25,
}
VOLUME = dict(zip('ARNDCQEGHILKMFPSTWYV',
                 [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,
                  166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
CORE_COMPAT = {'A':.65, 'V':1., 'I':1., 'L':1., 'M':.9,
               'F':1., 'W':.85, 'Y':.75, 'C':.75,
               'T':.35, 'S':.2, 'H':.25, 'N':.05, 'Q':.05,
               'D':0., 'E':0., 'K':0., 'R':0., 'G':.1, 'P':.3}
HELICES = set(range(141,148)) | set(range(179,188))
STRANDS = set(range(128,134)) | set(range(151,157)) | set(range(166,174)) | set(range(194,200))
CHARGED = set('DEKR')


def repeat_penalty(p, a, b, mapping):
    """Small smoothed compatibility penalty, not a complete evolutionary score.

    The WT anchors the distribution. Each paralog is only 0.25 effective
    observations, reduced to 0.12 in indel-prone loops. All coverage is recorded.
    """
    donors = mapping.get(p, [])
    if not donors:
        return 0.0
    weight = .25 if p in HELICES | STRANDS | {150,193} else .12
    def kernel(x,y):
        return np.exp(-substitution_distance(x,y)/.65)
    pa = .12 + 1.0 + weight * sum(kernel(x['aa'],a) for x in donors)
    pb = .12 + kernel(a,b) + weight * sum(kernel(x['aa'],b) for x in donors)
    return float(.40 * max(0.0, np.log(pa/pb)))


def core_penalty(p,a,b):
    importance = CORE.get(p,0.0)
    if not importance:
        return 0.0
    chemical = 1.65 * max(0., CORE_COMPAT[a]-CORE_COMPAT[b])
    if b in CHARGED and a not in CHARGED:
        chemical += .75
    # Both cavities and bulky substitutions can disrupt packing. Larger
    # penalties for overpacking, but no attempt to infer mutant relaxation.
    dv = VOLUME[b]-VOLUME[a]
    packing = .008*max(0.,-dv-20.) + .014*max(0.,dv-25.)
    if a in 'FY' and b not in 'FYW':
        packing += .18
    return importance*(chemical+packing)


def backbone_penalty(p,a,b):
    c = 0.0
    if b == 'P':
        if p in HELICES: c += 2.65
        elif p in STRANDS: c += 2.8
        elif p == 134: c += .45  # adjacent to WT P135
        else: c += .15
    if b == 'G' and a != 'G' and p in HELICES:
        c += .55
    if a == 'G':
        if p in {150,167,169,193}:
            importance = {150:1.25,167:1.65,169:1.45,193:1.10}[p]
            severity = .25 if b == 'A' else .55 if b == 'S' else .80 if b == 'C' else 1.10 if b == 'T' else 1.65
            c += importance*severity
        elif p == 163:
            c += .12 if b in 'AST' else .45
        # G177 and G188 are not invariant in the other RRMs. No motif-gly
        # penalty is assigned to them merely because the WT residue is G.
    if p == 200 and a == 'P':
        # Boundary turn can control interdomain orientation. Weak prior only:
        # no conserved-P claim, because the other repeats differ here.
        c += .85 if b in 'ASGT' else 1.35
    return c


def polar_constraint(p,b):
    if p == 127:
        # N/S in the other three RRM motif flanks; weak polar/turn prior,
        # less stringent than the RNP2 N132 sidechain.
        return .10 if b in 'STQD' else .25 if b in 'EHAG' else .45 if b in 'KR' else .85
    if p == 132:
        return {'Q':.15,'H':.30,'S':.40,'T':.45,'D':.55,'E':.65,
                'K':.85,'R':.90}.get(b,1.35)
    if p == 144:
        return {'E':.03,'N':.25,'Q':.35,'S':.5,'T':.55,'H':.75,
                'A':.85,'G':1.05}.get(b,1.3)
    if p == 154:
        return {'T':.03,'A':.20,'C':.25,'N':.35,'G':.40,'Q':.45}.get(b,1.05)
    if p == 187:
        return {'Q':.15,'S':.35,'T':.35,'D':.4,'E':.45,'H':.55,
                'K':.6,'R':.65,'A':.70,'G':.75}.get(b,1.0)
    if p == 160:
        return .0 if b == 'E' else .15 if b in 'NSTQ' else .35 if b in 'AGH' else .65
    return 0.0


def rna_penalty(p,a,b):
    # RNP2 aromatic and RNP1 positions 3/5 form the canonical RNA-stacking
    # face. RNP1 position 8 (F173) is mainly a structural hydrophobic prior,
    # so F173L is not equated to loss of F168/F170 stacking.
    if p in {129,168,170}:
        if b in 'FY': return .0
        if b == 'W': return .35
        if b in 'ILMV': return 1.75
        if b in 'H': return 1.1
        return 2.55
    if p == 172:
        return .15 if b in 'NQSC' else .35 if b in 'YF' else .65 if b in 'KR' else 1.25 if b in 'DE' else .9
    if p in {131,164,166}:
        importance = {131:.65,164:.45,166:.85}[p]
        return .0 if b in 'KR' else 1.25*importance if b in 'DE' else .55*importance
    if p == 156:
        return .0 if b in 'KR' else .55 if b in 'DE' else .20
    if p == 197 and b not in 'FYW':
        return .35  # less certain beta4 contribution, not a main RNP motif
    return .0


def mutation_components(p,a,b,mapping):
    return {
        'generic_substitution': .24*substitution_distance(a,b),
        'weak_internal_repeat': repeat_penalty(p,a,b,mapping),
        'putative_core': core_penalty(p,a,b),
        'backbone': backbone_penalty(p,a,b),
        'polar_site': polar_constraint(p,b),
        'rna_function': rna_penalty(p,a,b),
    }


def main():
    case, manifest, mutations = load_case()
    mapping, alignments = repeat_map(case['reference_sequence'])
    components = {}
    costs = {}
    for q in case['query']:
        terms = [mutation_components(p,a,b,mapping) for p,a,b in mutations[q['id']]]
        costs[q['id']] = sum(sum(t.values()) for t in terms)
        components[q['id']] = {'mutation': q['mutation'], 'total_cost': costs[q['id']],
                               'per_mutation': terms,
                               'repeat_observations': [len(mapping.get(p,[])) for p,a,b in mutations[q['id']]]}
    summary = ('Unsupervised sequence-only ranking uses substitution chemistry, '
               'canonical RRM folding and RNA-binding motifs, and weak intraprotein '
               'repeat comparisons. Mutation identities and all 250 IDs were checked; '
               'additive costs share one global scale. No measured outcomes, external '
               'MSA, coordinates or explicit RNA partners were supplied. Scores are '
               'relative priorities, with substantial uncertainty about growth, '
               'packing and double-mutant epistasis.')
    out = write_predictions('predictions.json', case['query'], costs, summary)
    for prediction in out['predictions']:
        prediction['score'] = round(prediction['score'], 6)
    Path('predictions.json').write_text(json.dumps(out,indent=2)+'\n')
    Path('model_components.json').write_text(json.dumps(components,indent=2)+'\n')
    Path('internal_repeat_mapping.json').write_text(json.dumps({
        'source': 'case.json reference_sequence only',
        'interpretation': 'weak, correlated intraprotein paralogs; not an external homolog MSA',
        'mapping': mapping, 'alignments': alignments},indent=2)+'\n')
    audit = {
        'case_sha256': hashlib.sha256(Path('case.json').read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(Path('evidence_manifest.json').read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(case['reference_sequence'].encode()).hexdigest(),
        'reference_length': len(case['reference_sequence']),
        'verified_candidates': 250, 'full_sequence_mutation_checks': True,
        'direction': 'higher_is_better', 'global_transform': '2*exp(-total_cost/3.5)-1',
        'measured_supports': 0, 'msa_evidence_present': False,
        'external_homolog_depth': 0, 'msa_coverage': 0,
        'structure_evidence_present': False, 'coordinate_coverage': 0,
        'residue_mapping': '1-based full-length reference positions; WT identities checked',
        'internal_repeat_observations_per_column': {p:len(v) for p,v in mapping.items()},
        'internal_repeat_missing_positions': [200],
        'internal_repeat_confidence': 'RNP anchors high; loops lower; three correlated domain copies',
        'fixed_partners': 'No explicit partner sequences or coordinates supplied; RNA target inferred from assay context only',
        'nonmutated_residues_unchanged': True,
        'parameters_fit_to_assay_outcomes': False,
        'validated_epistasis': False,
        'scores_finite_in_range': True,
    }
    Path('input_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    scores = [x['score'] for x in out['predictions']]
    print('Verified:',len(scores),'candidates; score range',min(scores),max(scores))
    for label, reverse in [('Highest priorities',False),('Lowest priorities',True)]:
        print(label)
        for q in sorted(case['query'],key=lambda q:costs[q['id']],reverse=reverse)[:12]:
            print(q['id'],q['mutation'],round(costs[q['id']],3),
                  round(next(x['score'] for x in out['predictions'] if x['id']==q['id']),4))


if __name__ == '__main__':
    main()
