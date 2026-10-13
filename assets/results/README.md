# Result figures

The figures read existing metrics only. They do not invoke models, execute predictors, refit programs or open hidden test labels.

- `results_overview.svg/.png`: four zero-shot evidence conditions and three feedback-program conditions. Thin bars show equal-assay means, with the exact four-decimal means above each condition. The y-axes are intentionally truncated to 0.480–0.610 (structure/MSA) and 0.575–0.630 (program ensemble); axis-break marks and captions disclose these limits. Brackets show the mean gain over each experiment's primary control. No confidence intervals or individual-assay points are drawn in this zoomed view.
- `results_by_assay.svg/.png`: all ten assay differences in each experiment. Left: joint structure/MSA evidence minus native Codex. Right: ensemble minus the CV-selected single from the same generated program pool. Both horizontal axes have the same limits. Positive and negative differences are retained.
- `figure_source_data.csv`: all 70 plotted method/assay values, paired deltas, support/query sizes and source paths.
- `plot_manifest.json`: original input hashes, output hashes, plotting-library versions and EVAgent style references.

The visual style follows the active EVAgent `figure2.svg` and `extended_data_3.svg`: Arial, white background, pastel pink/blue/gold/teal, thin axes and panel letters. PNGs preserve the exact rendered font; SVGs remain editable. Different machines can fall back to DejaVu Sans if Arial is unavailable.

Regenerate from the repository root using an environment with Matplotlib and NumPy:

```bash
python scripts/plot_results.py
```

Install plotting dependencies from `requirements-figures.txt` if needed. The plotted results are unchanged by regeneration. The two experiments use distinct assay cohorts and feedback settings; their improvements cannot be added. Assay-level variation is not the uncertainty from repeated model generation. The program-ensemble statistics include the disclosed completion-only recovery of unchanged code.
