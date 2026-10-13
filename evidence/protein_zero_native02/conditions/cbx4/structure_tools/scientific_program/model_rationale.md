The endpoint is protein stability measured by cDNA-display proteolysis, with
higher ProteinGym assay score being better. There are no measured supports,
WT assay measurements, or MSA evidence. Predictions are relative, uncalibrated
stability priors, not predictions of CBX4 binding or SUMO-ligase activity.

The first complete checkpoint used additive, burial-dependent hydrophobic,
volume and backbone-compatibility terms. One substantive refinement followed:
WT atoms reveal an R9/E16 ionic pair (2.72/2.82 A donor/acceptor distances),
W21/T30 (2.96 A) and T30/E32 (2.72 A) hydrogen bonds, and additional polar
contacts. Treating buried polar replacement as a full hydrophobic-transfer
gain ignored these native contacts. The revised model limits that gain and
adds directional hydrogen-bond and ionic-pair loss terms. Each ionic residue
pair is counted once, including for double substitutions; complementary
charge reversal receives partial, uncertain retention. Contacting double
substitutions receive only a small volume-compensation correction. No mutant
coordinates, relaxation, fitted epistasis or outcome-guided model search are
used.

Burial combines supplied WT RSA with a heavy-atom sidechain exposure fraction
recomputed relative to the same isolated WT residue. Both exposure calculations
use the same radii/probe/sampling scheme. This avoids comparing heavy-atom-only
sidechain area from explicit-H structures against incompatible residue-area
maxima. The supplied RSA is retained as an imperfect complementary descriptor;
values above one are retained in evidence and bounded only when computing a
burial weight. Atom-specific WT exposure also weights polar contacts.

Model constants are hand-selected stability priors. Changes in hydrophobicity,
sidechain volume, charge insertion, backbone preferences and native contacts
are combined additively, with limited contact-based pair corrections. The
global score is tanh((0.3 - total_penalty)/3.5) for all 250 variants together.
Its zero and scale do not constitute an inferred measured neutral point.

Validation confirms SHA-256 hashes of the raw WT PDB and feature source,
reference-sequence hashes, exact chain-A mapping of all 50 residues, complete
backbone and standard sidechains, every mutation identity and resulting
candidate sequence, higher-is-better direction, 250 unique complete IDs and
finite scores in [-1,1]. There are no supplied fixed partners or other chains.
The source asserts AlphaFold provenance without independent verification;
pLDDT is not used as fitness evidence. No missing measurements are assigned
zero, and no absent MSA evidence is used.

Code checks establish consistency and reproducibility, not predictive accuracy.
Native polar contacts may reorganize, cavities can repack, and proteolysis
readout effects or epistasis can depart from this model. Ranking uncertainty is
substantial, particularly for double mutants and charge/hydrophobic redesigns.
