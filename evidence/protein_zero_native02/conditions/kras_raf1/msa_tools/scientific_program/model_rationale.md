The initial usable checkpoint uses the supplied complete independent-site
log-frequency-ratio sums, with one global monotonic tanh transform. It remains
saved in checkpoint_predictions.json.

The final model makes one substantive revision, based on scientific scope:
natural GTPase conservation is not a direct measurement of KRAS binding to
fixed RAF1-RBD. G12/Q61 catalytic mutations can maintain a competent fold and
increase active-state occupancy despite natural rarity; canonical E37/D38/Y40
mutations can alter effector specificity; phosphate/Mg/guanine binding defects
can decrease occupancy. These established mechanisms warrant cautious priors
that replace the generic evolutionary penalty at the affected residues.
Conservative substitutions and proline/glycine changes in established Ras
secondary-structure ranges provide weak additional compatibility information.
K117T/A146F are treated cautiously because exchange-hotspot behavior cannot be
assumed from other substitutions. Activation has a modest saturating bonus;
this mechanistic assumption is not validated epistasis. Otherwise, mutation
effects add on an arbitrary latent log-performance scale before one global
tanh transform. No assay outcomes or parameter fitting were used.

Evidence validation: the raw public alignment hash matches the manifest;
the single exact full-query row matches chain A; source positions equal
canonical KRAS positions plus one. Selecting unique non-query aligned rows
in file order up to 2048 reproduces all 168 supplied depths and all supplied
frequencies exactly, with the documented total pseudocount of one. The
supplied scoring tool exactly reproduces msa_evidence.json. Frequencies and
fixed MSA scores are not changed. All 250 candidate MSA scores are complete;
263 distinct substitutions have non-gap depth 548–2048. Unique unweighted
depth is not Neff. There is no structural evidence or verified RSA. The
mutated-sequence differences, annotation, mutation order, fixed RAF1 partner,
reference sequence hashes, finite scores, direction and all 250 unique IDs
are checked by the saved prediction program.

The checks verify data and computation, not ranking accuracy. Distant/broad
homologs, unknown assay nucleotide state, altered mutant structure, expression
effects and double-mutant interactions remain material uncertainties. Scores
are ordinal prioritization estimates, not calibrated assay values.
