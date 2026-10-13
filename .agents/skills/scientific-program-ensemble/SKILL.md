---
name: scientific-program-ensemble
description: Combine three scientific predictors with blind equal-weight rank fusion, inspect the packaged Codex cases, or replay their frozen programs. Use for measured-feedback fitness tasks with three complete candidate score files.
---

# Scientific Program Ensemble

This repository packages already completed experiments. Default to existing inputs, programs and predictions. Do not launch new model sessions without an explicit user request.

Read `docs/method.md` for the tested workflow and `docs/results.md` for the paired controls. Each protein case has 96 measured supports and 186 unmeasured queries. Keep the assay endpoint, direction and reference sequence intact; the GCN4 source context contains a disclosed Binding/transcriptional-activity annotation mismatch.

## One-command example

```bash
python3 harness.py ensemble --out outputs/gfp_ensemble
```

This fuses the three stored GFP predictions into `ensemble.json` and `ranking.csv`, including program disagreement. `manifest.json` records input/output hashes and the operation performed. It does not execute or refit the predictors. For custom inputs, pass `--case` and three `--predictions` together; see [input contracts](../../../docs/getting-started.md).

For an existing-result review, inspect `inputs/<case>.json`, `programs/<case>/candidate_00..02`, and `reference_predictions/<case>/candidate_00..02.json`. The programs may choose different features, nonlinear models, transformations and validation strategies. Do not replace them with three generic formulas or pick members using held-out outcomes.

If the user requests deterministic cold replay, run `python demo.py --case <case> --out <new-directory>` using the dependencies in `requirements.txt`. This refits all three frozen programs from passed support, checks source/input hashes, verifies exact equality with stored predictions and writes `ensemble.json` and `ranking.csv`. It makes no LLM calls and does not open hidden evaluator labels. Preserve existing output directories.

For a user-requested new scientific task, give three independent sessions the same reference sequence, endpoint, measured supports and unmeasured candidates. Let each choose its own hypothesis, features, model and validation. Each delivers `proposal.py:propose(case, assignment)` and a complete candidate score file. Keep their workspaces separate, validate IDs and finite values, then use the ensemble entry point. Read `docs/method.md` for the tested program contract. The 96/186 sizes describe the frozen benchmark, not a restriction on custom tasks. Model-session orchestration remains part of the scientific workflow, not the offline fusion CLI.

Combine all three outputs by equal weights over average-tie percentile ranks; do not hand-select signs or weights. `scripts/combine.py` implements the same aggregation contract. Report ranking and program disagreement; disagreement is a review signal, not a calibrated probability.

For comparisons, report both the fixed first Codex program and the single program selected by feedback CV from the same three-program pool. The former has less generation budget; the latter shares the pool. Support-CV is non-nested with respect to program architecture selection, so the untouched query is the performance check.

Keep the 30 original scientific sessions, the capacity failure and the single completion-only recovery distinct. Existing-result arithmetic verification, frozen-program replay, fresh model generation and wet-lab validation are different evidence levels.
