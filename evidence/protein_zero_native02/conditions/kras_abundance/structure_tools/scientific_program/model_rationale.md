The initial complete checkpoint used additive residue-RSA-weighted hydrophobic,
charge, volume and backbone penalties. No measured data or MSA were available.

One substantive refinement used the supplied raw WT coordinates to correct
limitations of the initial structural heuristic:

- Replaced residue total ASA/theoretical maximum normalization with matched
  heavy-atom side-chain ASA in the full WT chain versus the isolated WT residue.
  The former can confuse backbone burial with side-chain exposure, and the
  supplied all-atom ASA and theoretical maximum have a stated compatibility caveat.
- Used nonlocal side-chain contact density to moderate packing penalties.
  For glycine, a carbon-sized solvent-space probe in the canonical CB direction
  avoids using backbone ASA as if it described a pre-existing side chain. This
  is a WT geometric probe, not a relaxed mutant, clash or free-energy calculation.
- Required an observed backbone NH hydrogen bond for a large helix proline
  penalty. A66 has alpha-like torsions but no such observed bond, so torsions
  alone overstated its predicted disruption. Positive-phi/incompatible-phi
  proline substitutions still receive a backbone penalty.
- Added weak losses of observed WT donor/acceptor polar-contact networks.
  Buried polar residues can be important to stability even when substituted
  to hydrophobic residues. Distance contacts do not establish bond energies.
- Downweighted canonical mobile switch regions (30–40 and 60–74) and the
  hypervariable tail, because one WT geometry is a weaker folding prior there.
  Slightly strengthened buried volume enlargement and reduced double-counting
  of added-charge penalties. These are physical heuristic choices, not fitted
  or selected against outcomes.

Mutation penalties are summed without inferred compensatory epistasis. All
variants use tanh((0.80 - total_penalty)/1.65), a single monotonic global scale.
No measured WT neutrality or absolute assay units are assumed. Protein abundance
is prioritized over catalytic/effector function: catalytic mutations are not
automatically assigned a loss of abundance. Nucleotide occupancy, expression,
degradation and assay-specific behavior remain unmodeled uncertainties.

Reference and source hashes, exact whole-chain residue mapping, mutation identity,
complete observed WT heavy atoms, absence of fixed protein partners, finite
scores, coverage of all IDs exactly once and score direction were checked.
Execution checks cannot establish which unsupervised heuristic is more accurate.
