# Scientific model and revision record

The initial checkpoint uses the complete sum of supplied marginal log frequency
ratios. All 250 candidates have coverage 1.0; neither a partial sum nor a missing
term is interpreted as a measured neutral effect. Source and sequence hashes,
mutation notation and actual changed residues are verified. Chain P remains
GTPPPPYTVG. No coordinate evidence is supplied, so no contacts, RSA, relaxation,
or mutant structures are claimed.

One substantive revision: use a conservative hybrid of the unchanged supplied
marginal ratios and explicit WW-domain biochemical priors. The WW domain is
identified from its sequence and assay context, with characteristic aromatic
and proline/glycine residues. Natural selection acts on more than binding to this
particular peptide. Moreover the supplied profile is an unweighted cap of 2,048
unique homologs, not an effective independent sample. Its strongest penalties
sometimes concern conservative substitutions, boundary residues, or sequence
choices whose chemistry remains compatible with a WW domain. Tiny pseudocount
frequencies should not imply arbitrarily precise functional damage.

The revision therefore bounds individual marginal penalties, gives conservative
chemical changes a mild generic contribution, and adds penalties for loss of
WW aromatic/core/turn integrity. Boundary D6 and R39 have smaller evolutionary
weights than the domain core. This is sequence-derived domain knowledge, not
verified solvent accessibility. Modest functional priors favor polar/basic
changes in the first loop and the D30/Q31 recognition-loop region, and selected
polar changes after the second tryptophan. In standard YAP numbering D6 is D170,
S19 is S183, D30 is D194, Q31 is Q195, and Q36 is Q200. These sites are plausible
WW-domain engineering hotspots; numerical corrections are qualitative priors,
not memorized or invented assay measurements. No pairwise coefficients,
epistasis, threshold selection against labels, or mutant geometry are fitted.

The full public alignment was inspected programmatically to verify an exact
46-column reference mapping and contextualize the first-2048 cap. The uncapped
alignment contains much more divergent WW sequences, and yields different
marginals (e.g. first-strand E14 and boundary R39). These broader family profiles
are not substituted for the supplied ones: they can span distinct ligand
preferences, and all raw rows are not independent samples. Raw and unique row
counts are never reported as effective Neff.

The hybrid has fixed, small, biochemically interpretable terms. No labels or
model search are used. Candidate latent values are mapped with one global
monotone tanh transformation. The scale is not calibrated to experimental
fitness or affinity. The initial checkpoint is retained for provenance, and the
final program writes a per-candidate feature audit and final_predictions.json.
Code verification establishes arithmetic, identity, coverage and output validity,
not scientific superiority of the revised heuristic. Stability/affinity tradeoffs
and multi-mutant epistasis remain substantial sources of uncertainty.
