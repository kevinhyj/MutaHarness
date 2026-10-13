Current four single-chain comparison figures — 2026-10-11

Unified current titles (no Zero-shot prefix):
Protein Variant Effect Prediction
RNA Variant Effect Prediction
Ligand ADMET Prediction (Regression)
Ligand ADMET Prediction (Classification)

Figures: paper/fig/single_chain/protein/, rna/, ligand/
Ligand regression and classification share ligand/. Each figure has three formats.
PNG 300 dpi, editable-text SVG, embedded-Arial vector PDF.
Current font sizes (pt): title 23; x-axis model names / assay counts 12;
y-axis title 16; y-axis ticks 12; input-group headings 12; model details 10;
bar means 10; bottom legend 9.5; Better 14.
Bottom legend is placed 8 pt below the lowest model-name/type text, avoiding
the previous fixed-position gap and label overlap.
The SMILES+3D input-group heading is kept on one line.
Notes below model names now describe model type in 10 pt: LLM, Protein LM,
Potts model, PSSM, Hybrid LM/model, Structure LM, Inverse folding,
Transformer, StripedHyena, MPNN, Random forest,
Boosted trees, or 3D Transformer. ESM2/PoET/ProSST variants are in model names.
Data, exclusions, exact means and source hashes: one subfolder per task here.

Reproduce from repository root:
/data/huangyanjie/miniconda3/bin/python paper/data/zero_shot/single_chain/input_group_figures_20261011/plot_comparison.py --tasks protein rna ligand_regression ligand_classification

Protein retains the previous nine baseline identities/versions and uses only
Gemini sequence_msa_prompt_structure; common cohort is 201 assays.
RNA retains all eight official baselines and now uses the highest non-fusion
Gemini arm, sequence_msa_prompt; common cohort is 10 assays.
The additional Gemini sequence-only bar is removed.

Ligand has no MSA arm. The current retained Gemini choice is
CR3+3D (SMILES + RDKit 3D shape summary) for regression, CR3 (SMILES only) for
classification. Historical classification 3D runs do exist; the ablation
summary uses a 10-assay cohort, distinct from this figure's 13-assay cohort.
Ligand regression/classification common cohorts are 8/13 assays respectively.
ADMET-AI was removed from both current ligand figures at the user's request.
Both retain four baselines: Chemprop, Uni-Mol, ECFP4 + RF, RDKit + XGBoost.
Uni-Mol uses generated 3D
conformers with possible 2D fallback; remaining baselines use SMILES-derived
2D molecular graphs, fingerprints or descriptors. No MSA is claimed.

Bars: assay-equal arithmetic mean of saved signed Spearman.
Dots: one real common assay each; no simulated data or error bars.
All models within each task share assay IDs and saved test candidate counts.
Missing/incomplete scores are excluded, never filled with zero.

Current Gemini means on the figure's baseline intersection:
Protein sequence+MSA+structure: 0.4824457688366049 / 201 assays.
RNA sequence+MSA: 0.42709503073192295 / 10 assays.
Ligand regression SMILES+3D CR3: 0.6189916370570654 / 8 assays.
Ligand classification SMILES CR3: 0.5561327706855318 / 13 assays.
All four Gemini arms are now chosen by highest common-cohort mean, with
MSA rank 50/50 fusion excluded. BEST_ARM_SELECTION.json documents sources,
shared-assay means, selection and precision. Protein/RNA/regression use
saved actual per-assay scores for selection; classification includes the
user's common-10-assay summary to compare the partial historical 3D arms.
On Protein's four-prompt shared 189 assays,
joint input is highest (0.4847777316745657 vs structure 0.48394902097270825).
On RNA's shared 10 assays, MSA alone is higher (0.42709503073192295
vs joint input 0.4176255934380675). Rank fusion is excluded.
Ligand regression CR3+3D is highest among the saved Gemini choices on
the same 8 assays. Classification CR3 is highest among CR1/CR3/Elo3
on the same 13 assays; classification 3D does not cover all 13 assays.
Grouping is based on information used in scoring, not architecture or general
pretrained capability. Inverse folding is placed in the structure group.

Gemini is pink and first, inside its actual input group, never a separate
Gemini group. Its input group goes first; remaining groups follow sequence,
MSA, structure, MSA+structure (or SMILES/2D and SMILES+3D for ligand).
Group headings use seq. and struct. abbreviations. For example, Protein
starts with seq.+MSA+struct. (Gemini then VenusREM); regression starts with
SMILES+3D (Gemini then Uni-Mol). RNA's first seq.+MSA group currently contains
only Gemini because the retained RNA baselines are all sequence-only.
Within each baseline input group, order is descending common-cohort mean;
the best baseline is colored and other baselines gray. Gemini remains pink
regardless of its score relative to baselines. Legend entries identify the
pink Gemini and each input group's best baseline; no extra empty groups.
White group backgrounds; vertical dashed separators replace shaded spans.
All fonts are actual Arial; original per-assay score files are unchanged.
No inference, model fitting, new candidate sampling or assay label reading.

Protein title is now Protein Variant Effect Prediction: the plotted
201 assays cover OrganismalFitness 69, Stability 66, Activity 39, Expression 16,
Binding 11. Classification is matched by assay ID to the saved ProteinGym
reference metadata, not inferred from score names. See protein/assay_measurement_types.csv.
Protein scatter uses s=8 (radius 1.414 pt), down from s=20 (2.236 pt).
Other three figures still use s=20. All points are read from saved sources.
RNA's cohort and baseline means are recomputed for the new 10-assay
intersection; Protein and ligand cohort IDs and scores remain unchanged.
