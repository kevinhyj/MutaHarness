#!/usr/bin/env python3
"""Draw EVAgent-style figures from historical metrics, with no inference."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'results'
PALETTE = {
    'basic': '#F5B5BF',
    'msa_tools': '#B79B63',
    'structure_tools': '#43968D',
    'joint_tools': '#D9788D',
    'direct': '#F5B5BF',
    'cv': '#728AB9',
    'ensemble': '#D9788D',
}
INK, MUTED, GRID = '#29323A', '#74777F', '#D6DBDF'
NAMES = {
    'amfr': 'AMFR', 'his7': 'HIS7', 'yap_peptide': 'YAP–peptide',
    'bh3_mcl1': 'BH3–Mcl-1', 'ube4b': 'UBE4B', 'cbx4': 'CBX4',
    'gfp': 'GFP', 'pabp': 'PABP', 'kras_abundance': 'KRAS abundance',
    'kras_raf1': 'KRAS–RAF1', 'gcn4': 'GCN4', 'grb2': 'GRB2',
    'yap1': 'YAP1', 'phot': 'PHOT', 'ras_abundance': 'KRAS abundance',
    'ynzc': 'YNZC',
}
INPUTS = [
    ROOT / 'evidence/protein_zero_native01/SUMMARY.json',
    ROOT / 'evidence/protein_zero_native02/SUMMARY.json',
    ROOT / 'evidence/scientific_program_ensemble/pooled_protein_metrics.json',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    summaries = [json.loads(p.read_text()) for p in INPUTS]
    zero_rows = summaries[0]['rows'] + summaries[1]['rows']
    cases = list(dict.fromkeys(r['case_id'] for r in zero_rows))
    arms = ['basic', 'msa_tools', 'structure_tools', 'joint_tools']
    zero = np.array([[next(r['spearman'] for r in zero_rows
                          if r['case_id'] == case and r['arm'] == arm)
                      for arm in arms] for case in cases])
    spe_rows = summaries[2]['rows']
    spe_cases = [r['case_id'] for r in spe_rows]
    spe = np.array([[r['candidate_rho'][0], r['cv_selected_single_rho'], r['ensemble_rho']]
                    for r in spe_rows])
    assert zero.shape == (10, 4) and spe.shape == (10, 3)
    assert np.isfinite(zero).all() and np.isfinite(spe).all()
    # Check the plotted means against the already reported metrics.
    assert abs(zero[:, 0].mean() - .5050867801038631) < 1e-12
    assert abs(zero[:, 3].mean() - .5735689527270961) < 1e-12
    for index, name in [(0, 'delta_vs_direct00'), (1, 'delta_vs_cv_single')]:
        assert abs((spe[:, 2] - spe[:, index]).mean()
                   - summaries[2]['comparisons'][name]['mean_delta']) < 1e-12
    return cases, zero, spe_cases, spe


def style():
    # The active EVAgent figure2.svg uses Arial. Keep a portable fallback.
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
        font = 'Arial'
    except ValueError:
        font = 'DejaVu Sans'
    plt.rcParams.update({
        'font.family': font, 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 12,
        'axes.linewidth': .7, 'axes.edgecolor': MUTED,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    return font


def heading(ax, letter, title, subtitle):
    ax.text(-.14, 1.16, letter, transform=ax.transAxes,
            fontsize=17, weight='bold', va='bottom', color='#111111')
    ax.text(0, 1.16, title, transform=ax.transAxes,
            fontsize=12, weight='bold', va='bottom', color='#111111')
    ax.text(0, 1.08, subtitle, transform=ax.transAxes,
            fontsize=9, color=MUTED, va='bottom')


def save(fig, name):
    files = []
    for suffix in ['svg', 'png']:
        p = OUT / f'{name}.{suffix}'
        fig.savefig(p, dpi=240, bbox_inches='tight', pad_inches=.15,
                    metadata={'Creator': 'MutaHarness: stored-result visualization'} if suffix == 'svg' else None)
        files.append(p)
    plt.close(fig)
    return files


def mean_panel(ax, values, keys, labels, ylim, control_index, tick_step, tick_format):
    means = values.mean(axis=0)
    x = np.arange(len(keys))
    colors = [PALETTE[k] for k in keys]
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=.45, alpha=.7)
    ax.bar(x, means, width=.32, color=colors, edgecolor=colors,
           linewidth=.65, alpha=.72, zorder=2)
    span = ylim[1] - ylim[0]
    for j, color in enumerate(colors):
        ax.text(x[j], means[j] + span*.026, f'{means[j]:.4f}',
                ha='center', va='bottom', fontsize=11, weight='bold', color=INK)
    ax.set_xticks(x, labels)
    ax.set_xlim(-.55, len(keys)-.45)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
    ax.set_ylabel('Mean Spearman ρ')
    ax.tick_params(axis='x', length=0, pad=8)
    # Axis-break marks make the nonzero baseline explicit.
    for offset in [0., -.025]:
        ax.plot([-.013, .013], [offset-.01, offset+.01], transform=ax.transAxes,
                color=INK, linewidth=.9, clip_on=False, zorder=8)
    last = len(keys)-1
    bracket = float(means.max()) + span*.17
    ax.plot([control_index, control_index, last, last],
            [bracket-span*.028, bracket, bracket, bracket-span*.028],
            color=PALETTE['ensemble'], lw=.95, clip_on=False)
    delta = means[last] - means[control_index]
    ax.text((control_index+last)/2, bracket + span*.024,
            f'Δ = {delta:+.4f}', ha='center', va='bottom',
            fontsize=10.5, weight='bold', color=PALETTE['ensemble'])


def overview(zero, spe):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    fig.subplots_adjust(left=.075, right=.985, top=.77, bottom=.18, wspace=.34)
    mean_panel(axes[0], zero,
               ['basic', 'msa_tools', 'structure_tools', 'joint_tools'],
               ['Native\nCodex', '+ MSA', '+ structure', '+ MSA\n+ structure'],
               (.48, .61), 0, .02, '%.2f')
    mean_panel(axes[1], spe, ['direct', 'cv', 'ensemble'],
               ['Single native\nCodex', 'CV-selected\nsingle program', 'Scientific program\nensemble'],
               (.575, .63), 1, .01, '%.2f')
    heading(axes[0], 'a', 'Structure + MSA',
            'Zero-shot · 10 assays · 250 candidates per assay')
    heading(axes[1], 'b', 'Scientific Program Ensemble',
            '96 measured supports · 10 assays · 186 candidates per assay')
    fig.text(.075, .055, 'Bars: 10-assay means     Truncated y-axes: a, 0.480–0.610; b, 0.575–0.630',
             fontsize=8.5, color=MUTED)
    fig.text(.075, .014, 'The two experiments use different cohorts and feedback; their gains are not additive.',
             fontsize=8.5, color=MUTED)
    return save(fig, 'results_overview')


def delta_panel(ax, cases, deltas, title, letter, subtitle):
    y = np.arange(len(cases))
    # Common range across panels avoids magnifying the smaller ensemble effect.
    ax.set_xlim(-.115, .28)
    ax.set_ylim(-.8, len(cases)-.2)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis='x', color=GRID, linewidth=.5, alpha=.65)
    ax.axvline(0, color=MUTED, linewidth=.9, zorder=2)
    for yi, delta in zip(y, deltas):
        color = PALETTE['ensemble'] if delta >= 0 else PALETTE['cv']
        ax.plot([0, delta], [yi, yi], color=color, linewidth=2.1, alpha=.7, zorder=3)
        ax.scatter([delta], [yi], color=color, s=48, edgecolors='white', linewidths=.65, zorder=4)
        on_right = delta >= 0 or delta < -.05
        x = delta + (.009 if on_right else -.009)
        number = f'{delta:+.4f}' if abs(delta) < .001 else f'{delta:+.3f}'
        ax.text(x, yi, number, ha='left' if on_right else 'right',
                va='center', color=INK, fontsize=8.5)
    ax.set_yticks(y, [NAMES[c] for c in cases])
    ax.tick_params(axis='y', length=0, pad=9)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(.1))
    ax.set_xlabel('Paired ΔSpearman ρ', labelpad=8)
    mean = deltas.mean()
    ax.text(.98, 1.075, f'Mean Δ = {mean:+.4f}', transform=ax.transAxes,
            ha='right', fontsize=9, weight='bold', color=PALETTE['ensemble'])
    # The common heading layout is adapted for the longer per-assay panels.
    ax.text(-.30, 1.18, letter, transform=ax.transAxes, fontsize=17,
            weight='bold', color='#111111')
    ax.text(0, 1.18, title, transform=ax.transAxes, fontsize=12,
            weight='bold', color='#111111')
    ax.text(0, 1.12, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED)


def by_assay(zero_cases, zero, spe_cases, spe):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.65))
    fig.subplots_adjust(left=.14, right=.98, top=.78, bottom=.18, wspace=.64)
    delta_panel(axes[0], zero_cases, zero[:, 3]-zero[:, 0],
                'Structure + MSA', 'a', 'Joint evidence − native Codex · 7/10 assays improved')
    delta_panel(axes[1], spe_cases, spe[:, 2]-spe[:, 1],
                'Scientific Program Ensemble', 'b', 'Ensemble − same-budget CV single · 8/10 assays improved')
    fig.text(.14, .059, 'Each point = one assay     Pink: improvement     Blue: regression     Identical horizontal scales',
             fontsize=8.5, color=MUTED)
    fig.text(.14, .019, 'Descriptive paired differences from completed historical experiments; no new predictions.',
             fontsize=8.5, color=MUTED)
    return save(fig, 'results_by_assay')


def export_source(zero_cases, zero, spe_cases, spe):
    p = OUT / 'figure_source_data.csv'
    with p.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'case_id', 'assay_label', 'method', 'spearman',
                         'delta_vs_primary_control', 'measured_supports', 'query_candidates', 'source_file'])
        arms = ['basic', 'msa_tools', 'structure_tools', 'joint_tools']
        for i, case in enumerate(zero_cases):
            source = INPUTS[0] if i < 4 else INPUTS[1]
            for j, arm in enumerate(arms):
                writer.writerow(['structure_msa_zero', case, NAMES[case], arm,
                                 repr(float(zero[i, j])), repr(float(zero[i, j]-zero[i, 0])),
                                 0, 250, str(source.relative_to(ROOT))])
        for i, case in enumerate(spe_cases):
            for j, method in enumerate(['direct', 'cv', 'ensemble']):
                writer.writerow(['program_ensemble_feedback', case, NAMES[case], method,
                                 repr(float(spe[i, j])), repr(float(spe[i, j]-spe[i, 1])),
                                 96, 186, str(INPUTS[2].relative_to(ROOT))])
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = style()
    zero_cases, zero, spe_cases, spe = load_data()
    files = overview(zero, spe) + by_assay(zero_cases, zero, spe_cases, spe)
    files.append(export_source(zero_cases, zero, spe_cases, spe))
    reference = Path('/Users/hyj/Desktop/EvaDDE/EVAgent/overleaf/figures/figure2.svg')
    reference2 = reference.parent / 'extended_data/extended_data_3.svg'
    manifest = {
        'purpose': 'EVAgent-style presentation of existing results',
        'new_model_calls': 0, 'predictor_executions': 0, 'test_labels_opened': False,
        'data_sources': {str(p.relative_to(ROOT)): digest(p) for p in INPUTS},
        'style': {'font': font, 'palette': PALETTE, 'white_background': True,
                  'reference': 'EVAgent active figure2 and Extended Data 3: Arial, pastel pink/blue/gold/teal, thin axes, paired data points',
                  'reference_sha256': {p.name: digest(p) for p in [reference, reference2] if p.exists()}},
        'aggregation': 'equal assay means; 10 assays per experiment; separate cohorts',
        'error_bars': 'none; overview shows means only; the by-assay figure shows all paired differences',
        'overview_y_axes': {'truncated': True, 'structure_msa': [.48, .61],
                            'program_ensemble': [.575, .63], 'bar_width': .32,
                            'disclosure': 'axis-break marks and explicit figure/README captions'},
        'outputs': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__,
    }
    (OUT / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Saved 2 figures (SVG/PNG), source CSV and provenance. No model/predictor calls.')


if __name__ == '__main__':
    main()
