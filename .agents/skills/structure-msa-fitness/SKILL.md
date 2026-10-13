---
name: structure-msa-fitness
description: Analyze structure and MSA evidence for zero-shot protein or protein–protein fitness ranking in MutaHarness. Use for canonical case preparation, evidence-tool calls and review of the four saved input conditions.
---

# Structure + MSA Fitness

Work from the repository root. Read `docs/method.md` for the tested workflow and `docs/results.md` for its four-arm comparison. The packaged zero-shot cases have 250 queries and no measured supports. Default to saved results; generate a new predictor or launch model sessions only when the user requests that work.

## Prepare evidence

Use a canonical `case.json` with `reference_sequence`, `query` and `evidence.format=protein_site_table_v1`; PPI cases also specify `reference_chains`. Check the assay endpoint, effect direction, candidate IDs, chain mapping and WT residue at each numbered site. Keep candidate fitness labels outside the model workspace.

The AMFR example includes an aligned MSA summary and a raw WT PDB. From the repository root, create evidence sidecars in a fresh output directory:

```bash
python3 harness.py evidence --out outputs/amfr_evidence
```

The entry point checks candidate IDs, excludes query fitness labels, verifies evidence mapping and writes `msa.json`, `structure.json` and a common `manifest.json` with input/output hashes. It uses the supplied evidence and standard library. For your own canonical case, add `--case path/to/case.json`; see [input contracts](../../../docs/getting-started.md).

The MSA sidecar returns per-substitution frequency/log-odds terms and completeness. The structure sidecar returns WT exposure, local geometry, confidence and available pair distances. Treat these as evidence for prediction, not an already fitted assay predictor.

When full contact geometry is needed, use `tools/structure_tools.py` with `--raw-pdb examples/zero_shot/amfr/AMFR_HUMAN.pdb` and NumPy. It recomputes WT contact graphs and optional SASA from supplied coordinates. Keep this output separate from the summary sidecar and preserve input provenance.

## Use evidence for the assay

Combine evolutionary compatibility and structural context according to the measured endpoint. Let Codex choose the scientific model rather than hard-coding a universal structure/MSA weighting. Inspect evidence conflicts and missing coverage before ranking.

- Preserve null terms and observed partial sums; incomplete MSA evidence is not a complete candidate score.
- Distinguish effective MSA depth from raw depth. Independent chain profiles do not establish cross-chain coevolution.
- WT geometry and pLDDT are evidence, not mutant energies or measured fitness. Summary nearest-neighbor lists are not complete contact graphs.
- Absolute SASA is not RSA. Pass `--verified-rsa` to the MSA tool only for a separately verified, sequence-bound `verified_rsa_v1` artifact.

For a requested new prediction, deliver an executable program, one finite score per candidate ID, a ranking and a short account of the evidence used. Validate candidate coverage and score direction without consulting held-out outcomes.

## Review the saved comparison

Inspect `evidence/protein_zero_native01/` and `evidence/protein_zero_native02/` for the baseline, MSA, structure and joint conditions. `python3 -m scripts.audit_existing` verifies their saved metrics without running predictors. Report the joint-input comparison independently from Scientific Program Ensemble: the zero-shot study has 0 supports/250 queries; the ensemble study has 96 supports/186 queries.
