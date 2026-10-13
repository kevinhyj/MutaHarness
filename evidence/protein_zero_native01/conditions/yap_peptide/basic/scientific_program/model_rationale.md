# Model and revision rationale

No measured support or outcome fitting, and no MSA or coordinates are available. Only supplied sequences and general WW-domain knowledge enter the predictor. Case and manifest SHA-256 hashes are recorded in input_audit.json. Positions are 1-based source-chain positions, not full-length YAP numbering. Mutation and partner evidence covers all 250 candidates; evolutionary and coordinate evidence coverage is zero.

The initial complete checkpoint is preserved as initial_checkpoint_predictions.json. One bounded scientific review revised that initial model:
- Include Q22/R23 in the approximate second-strand region and distinguish the H28/I29 recognition loop from canonical strand positions. This repairs an omitted backbone proline penalty, particularly R23P.
- Remove an unjustified favorable D30G turn prior: glycine increases flexibility but does not supply the Asn side-chain hydrogen-bond network. Retain a moderate D30N amide/turn stabilization prior and explicitly penalize loss of conserved N27 amide chemistry.
- Treat histidine as partially retaining aromatic contacts and strengthen loss of the conserved P38 fold cap. Include weak, peripheral Q36 chemistry, below central binding-site effects.
- Average folding contributions over three modest WT stability assumptions rather than implying that a specific WT folded fraction is experimentally established. Interface terms remain separate, with the complete fixed PPxY partner retained.

Weights are low-dimensional, qualitative physical priors, not fitted thermodynamic parameters. There is no validated pairwise epistasis: only nonlinear folding occupancy applied to additive mutation contributions. A single tanh transform is applied to every candidate. Chemistry and fold membership remain uncertain without coordinates; the checks establish input consistency and numerical validity, not predictive accuracy.
