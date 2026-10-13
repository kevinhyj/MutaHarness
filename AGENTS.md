# MutaHarness submission

This repository organizes existing experiments. Do not launch model calls, fresh inference, or new experiments unless the human user explicitly requests them.

Keep the two demonstrated interventions distinct: Scientific Program Ensemble uses 96 measured supports and 186 queries; structure/MSA zero-shot uses no measured supports and 250 queries. They were evaluated separately. Do not present their deltas as an additive combined result.

Preserve original program sources, failures, delivery recovery and historical metrics. Report stored-result verification separately from a fresh model run. Cold replay (`demo.py`) refits frozen programs without calling an LLM; it is optional and should not be confused with fresh inference.

The packaged Skill defaults to inspecting existing outputs. Read `.agents/skills/scientific-program-ensemble/SKILL.md` when using or replaying the frozen programs.
