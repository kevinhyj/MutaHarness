Initial checkpoint: BLOSUM62 substitution distances, generic RRM site weighting,
RNA-motif aromatic penalties, and secondary-structure proline penalties.

Refinement rationale (no outcome-supervised model selection):

1. The reference contains four identifiable RRM repeats. An anchored intraprotein
   comparison distinguishes conserved glycines G150/G167/G193 from divergent
   G177/G188, and identifies variable helix/loop sidechains. This is only a weak
   prior: three correlated paralogs, incomplete terminal coverage, and different
   domain-specific roles. Indel-prone loop columns are downweighted. No external
   MSA, homolog depth, or complete evolutionary fitness score is inferred.
2. Replace coarse positional BLOSUM weights with explicit, approximate core,
   backbone and polar-site penalties. Exposed charge changes and the variable
   A185/G188 region should not be penalized like buried hydrophobic replacements.
   The inferred topology is a known RRM-fold prior, not observed WT coordinates
   or mutant relaxation. All settings are fixed biochemical heuristics.
3. RNP1 F168/F170 and RNP2 F129 are canonical RNA-stacking aromatics; F173 is
   primarily a hydrophobic structural position. A hydrophobic F173 replacement
   therefore gets less penalty than loss of F168/F170 stacking. Growth integrates
   RNA binding and folding with other fixed RRM domains; neither proxy alone
   establishes assay performance.

Mutation effects are additive costs (equivalently independent multiplicative
compatibility on a log scale), with no claimed experimentally validated epistasis.
One monotonic transform is used across all 250 IDs. Input hashes, residue identity
checks, and explicit absence/coverage fields are saved in input_audit.json.
Execution validates implementation and completeness, not ranking accuracy.

Final mapping audit: include the two residues immediately before the anchored
RNP2 motif at lower confidence. Their sequence context is supplied; the initial
repeat alignment simply trimmed them. N127 corresponds to N/S across the other
RRMs, supporting a weak polar-flank penalty rather than completely unconstrained
terminal treatment. P200 remains outside the repeat alignment coverage.
