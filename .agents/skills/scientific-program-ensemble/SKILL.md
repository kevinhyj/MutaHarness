---
name: scientific-program-ensemble
description: Inspect or replay the three frozen Codex scientific predictors in MutaHarness and combine their rankings from the same measured feedback. Use for the packaged protein fitness cases and existing-result review.
---

# Scientific Program Ensemble

This repository packages already completed experiments. Default to existing inputs, programs and predictions. Do not launch new model sessions without an explicit user request.

Read `docs/method.md` for the tested workflow and `docs/results.md` for the paired controls. Each protein case has 96 measured supports and 186 unmeasured queries. Keep the assay endpoint, direction and reference sequence intact; the GCN4 source context contains a disclosed Binding/transcriptional-activity annotation mismatch.

For an existing-result review, inspect `inputs/<case>.json`, `programs/<case>/candidate_00..02`, and `reference_predictions/<case>/candidate_00..02.json`. The programs may choose different features, nonlinear models, transformations and validation strategies. Do not replace them with three generic formulas or pick members using held-out outcomes.

If the user requests deterministic cold replay, run `python demo.py --case <case> --out <new-directory>` using the dependencies in `requirements.txt`. This refits all three frozen programs from passed support, checks source/input hashes, verifies exact equality with stored predictions and writes `ensemble.json` and `ranking.csv`. It makes no LLM calls and does not open hidden evaluator labels. Preserve existing output directories.

Combine all three outputs by equal weights over average-tie percentile ranks; do not hand-select signs or weights. `scripts/combine.py` implements the same aggregation contract. Report ranking and program disagreement; disagreement is a review signal, not a calibrated probability.

For comparisons, report both the fixed first Codex program and the single program selected by feedback CV from the same three-program pool. The former has less generation budget; the latter shares the pool. Support-CV is non-nested with respect to program architecture selection, so the untouched query is the performance check.

Keep the 30 original scientific sessions, the capacity failure and the single completion-only recovery distinct. Existing-result arithmetic verification, frozen-program replay, fresh model generation and wet-lab validation are different evidence levels.
