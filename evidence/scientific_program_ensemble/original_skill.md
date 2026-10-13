---
name: native-program-portfolio
description: Build and execute three reusable RNA or protein variant ranking programs from the same measured feedback, then combine their predictions without consulting unmeasured outcomes. Use when a feedback-based ranking task justifies additional native Codex inference.
---

Preserve native Codex's freedom to choose scientific features, nonlinear models and validation. The harness intervention is independent program generation and deterministic ranking aggregation.

Use one immutable public case containing measured supports, unmeasured candidate IDs/sequences, assay context and ranking direction. Every candidate receives exactly that same case and the same instructions in its own isolated workspace. Generate three candidates with the same model, reasoning setting, tools and per-session time limit. Candidate00 is the preregistered direct comparator. The frozen campaign runner is `run.py` in the project root; it also enforces tool-network denial and prevents access to hidden evaluator labels and other workspaces.

Choose the tested RNA or protein template with `run.py --domain rna|protein|protein-extra --run-id <new-id>`. Interpret the supplied assay endpoint and molecular alphabet; RNA base-pair assumptions should not become generic protein features. Stability, fluorescence, growth and other activity scores have different interpretations. Preserve input annotation uncertainty in the decision note rather than silently rewriting the measured target.

Each candidate submits a deterministic `proposal.py:propose(case,assignment)` and is independently executed by the protected executor. Models refit from the passed support records, rather than cached full-data fitted states. Keep scientific choices and model failures in their original records. A failure means the three-program policy did not complete for that condition; do not silently replace a candidate after seeing ranking quality.

Combine every completed member with equal weights over average-tie percentile ranks. Use `scripts/combine.py`; do not hand-select programs, signs or weights using unmeasured outcomes. The helper requires exactly three predictions and exact coverage of the current unmeasured candidate IDs.

Report both quality and cost. This policy consumes three native scientific sessions versus one for direct Codex; an invocation can contain many underlying model turns. Additional inference is part of the treatment. Whole-program support validation can supply a separate best-single comparator with the same three generated candidates. If the architecture was chosen using all public supports, describe subsequent held-out refitting as non-nested architecture validation. Scientific superiority remains unproven until the frozen held-out evaluation passes.
