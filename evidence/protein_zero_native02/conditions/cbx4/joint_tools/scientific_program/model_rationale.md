The initial complete checkpoint used additive WT burial, hydrophobicity, packing,
backbone Gly/Pro priors and a small bounded evolutionary contribution.

One substantive revision followed inspection of the supplied raw WT coordinates:

- R9/E16, E3/K20 and D39/R41 have directional ionic contacts. W21/T30
  and several other sites have directional polar contacts. Native hydrophobicity
  differences alone omit the cost of losing these observed interactions. The
  final predictor applies uncertain chemistry-retention penalties to each WT
  contact once; it predicts no new mutant contacts or compensating interactions.
- Whole-residue RSA can conflate a buried aliphatic stem with an exposed charged
  head. Coordinate-derived exposure of the terminal polar group attenuates
  hydrophobic substitution bonuses for native charged residues.
- Isolated alpha-like torsions in turns do not establish a helix. The final
  backbone prior requires a sustained alpha-like run and an i-to-i-4 backbone
  hydrogen bond, and uses direct WT NH bonding and phi compatibility for Pro.
- Cavity/overpacking terms and bounded conservation have modest fixed weights;
  helix propensities provide an established physical prior. No coefficients
  were optimized against assay outcomes, and no model search was performed.

All 250 candidates were checked against the WT and supplied candidate sequences.
All four permitted raw source hashes match. The coordinate sequence matches all
50 WT residues on the sole chain A, and recomputed SASA matches the supplied raw
coordinate evidence. The 11 zero-depth homolog sites are positions 1-10 and 50.
Only 147 candidates have complete MSA evidence. Observed evolutionary terms are
site-level supplements to physical predictions; partial sums are never treated
as complete evolutionary candidate scores. There are no supplied fixed partners.

All candidates use the same monotone score transform, tanh(latent/4). Zero is an
arbitrary WT-referenced compatibility baseline, not measured neutral fitness.
Execution checks verify consistency and coverage, not predictive accuracy.
The model is uncalibrated and has no relaxed mutant coordinates or experimentally
validated epistasis; mutation chemistry and WT contact retention remain uncertain.
