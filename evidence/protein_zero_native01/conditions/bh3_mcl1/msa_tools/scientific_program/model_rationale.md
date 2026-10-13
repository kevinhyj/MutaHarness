The first checkpoint uses the supplied complete additive MSA log-frequency
ratios. Source-file and reference-sequence hashes, mutation identities, fixed
chain A, reference-column mapping, depths and frequencies were checked. Calling
msa_tools.score_case reproduces the supplied sidecar exactly. All candidate
substitutions have observed coverage; positions 1–5 and 19–23 have none and are
not mutated. The 2048 unique nonquery homologs are unweighted, not effective Neff.

One substantive revision adds a BH3–Mcl-1 binding prior. The reason is biological:
the reference is a BH3 helix with hydrophobic anchors I6/L10/I13/F17 and conserved
Asp15. General natural-homolog compatibility is not specific to binding the fixed
Mcl-1 partner. The refined model therefore scores anchor hydrophobicity and size,
buried charge, the Asp interaction motif, small residues and helix propensity.
The later two hydrophobic anchors have weaker penalties, including a modest
allowance for neutral amides, reflecting Mcl-1's known broader tolerance there.
Neither favorable mutant contacts nor relaxed mutant coordinates are assumed.

Natural frequencies remain unchanged. Their contribution to this composite
model is downweighted and capped at six log units per substitution to avoid
letting rare states in an unweighted alignment dominate the physical prior.
This composite is not the fixed tool's LR score. Physical descriptors and the
small set of motif weights are specified a priori, without outcome fitting.
Helix compatibility is secondary to the binding motif, not a standalone folding
predictor. Mutation burdens are additive, with no fitted or claimed epistasis.
The same fixed tanh transform is applied globally to all 250 burdens.

Execution checks establish reproducibility and complete, finite output, not
predictive accuracy. No measured examples or target outcomes are available.
The major uncertainties are actual pocket relaxation, partner-specific polar
tolerance, assay effects other than intrinsic affinity, and combination effects.
