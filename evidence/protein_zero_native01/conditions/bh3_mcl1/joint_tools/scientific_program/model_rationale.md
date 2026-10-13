Model rationale

Checkpoint: supplied complete additive evolutionary log-frequency ratios, with a bound-helix proline penalty.

Structural refinement: unpaired BH3 homolog frequencies do not encode fixed Mcl-1 affinity, and effective sequence depth is unknown. Bound extreme marginal contributions and combine with standard hydrophobicity, volume, helix propensity and verified WT-interface descriptors. WT geometry supports tighter pockets at Ile6/Leu10 and more space at Ile13/Phe17; Asp15 contacts partner Arg92 and Asn89. Shorter acids at Arg11 may recover an interaction with partner Arg92, but this is discounted and unvalidated. No outcome supervision, optimization search or learned epistasis.

Consistency correction: intrinsic helix propensity addresses preorganization but undercounts removal of a bound backbone NH donor by proline. All five candidate proline sites have WT N(i)-O(i-4) distances 2.81–2.97 A; add a modest donor-loss cost.

Validation: 250 unique IDs M001–M250; mutation positions/identities, complete candidate sequences and unchanged partner chain A verified. Both public source byte hashes and chain sequence hashes verified. Recomputed fixed MSA score dictionaries exactly match supplied values. All mutated positions have nonzero homolog depth and complete marginal scores; zero-depth terminal positions excluded. WT coordinates cover all C residues; A unknown positions 25–31 remain unmapped. Every used WT sidechain is complete; static modeled coordinates and null AF2 confidence are not treated as mutant predictions. Scores use one global tanh(raw/7), higher is better. These execution checks do not validate ranking accuracy.
