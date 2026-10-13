Initial checkpoint: independent homolog marginals with weak WT geometry penalties; explicit chemistry fallback for absent MSA. No phenotype supervision. Global tanh scale preserves ordering.

One substantive refinement, without outcome testing or parameter search:
1. Apply a 15% BLOSUM62 substitution kernel to finite homolog marginals: unseen conservative and radical substitutions otherwise receive nearly identical floor penalties. Preserve all deterministic tool values and report the new model separately.
2. Add a small diversity-weighted close-homolog profile, selected on the other RRM domains. Raw observations show context differences at H134/P135/F168/H172/E195; this addresses broad taxonomic marginal mismatch. Query-lowercase K164 remains absent MSA evidence.
3. Add canonical RRM RNA motif penalties and positive-phi Asn-turn compatibility to represent assay function and resolve the glycine-only backbone prior. No geometric RNA affinity, mutant relaxation, or experimentally validated epistasis claimed.
4. Use one global median/quantile tanh display scale; monotonicity preserves all ranking and does not estimate measured neutrality.
