#!/usr/bin/env python3
"""Draw EVAgent-style figures from historical metrics, with no inference."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from runtime.artifacts import ROOT, read_json, sha256
from scripts.results import ENSEMBLE_SOURCE, LABELS, ZERO_SOURCES, load_metrics

OUT = ROOT / "assets" / "results"
PALETTE = {
    "basic": "#F5B5BF",
    "msa_tools": "#B79B63",
    "structure_tools": "#43968D",
    "joint_tools": "#D9788D",
    "direct": "#F5B5BF",
    "cv": "#728AB9",
    "ensemble": "#D9788D",
}
INK, MUTED, GRID = "#29323A", "#74777F", "#D6DBDF"
NAMES = {
    key: label.replace("肽", "peptide").replace("丰度", " abundance")
    for key, label in LABELS.items()
}
INPUTS = [ROOT / source for source in [*ZERO_SOURCES, ENSEMBLE_SOURCE]]


def load_data():
    data = load_metrics()
    conditions = read_json(ROOT / "artifacts_manifest.json")["conditions"]
    cases = [case["id"] for case in data["zero"]["cases"]]
    spe_cases = [conditions[case["id"]]["case_id"] for case in data["ensemble"]["cases"]]
    zero = np.array([list(case["scores"].values()) for case in data["zero"]["cases"]])
    spe = np.array([list(case["scores"].values()) for case in data["ensemble"]["cases"]])
    return cases, zero, spe_cases, spe


def style():
    # The active EVAgent figure2.svg uses Arial. Keep a portable fallback.
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        font = "Arial"
    except ValueError:
        font = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "axes.linewidth": 0.7,
            "axes.edgecolor": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return font


def heading(ax, letter, title, subtitle):
    ax.text(
        -0.14,
        1.16,
        letter,
        transform=ax.transAxes,
        fontsize=17,
        weight="bold",
        va="bottom",
        color="#111111",
    )
    ax.text(
        0,
        1.16,
        title,
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        va="bottom",
        color="#111111",
    )
    ax.text(0, 1.08, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")


def save(fig, name):
    files = []
    for suffix in ["svg", "png"]:
        p = OUT / f"{name}.{suffix}"
        fig.savefig(
            p,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.15,
            metadata={"Creator": "MutaHarness: stored-result visualization"}
            if suffix == "svg"
            else None,
        )
        files.append(p)
    plt.close(fig)
    return files


def mean_panel(ax, values, keys, labels, ylim, control_index, tick_step, tick_format):
    means = values.mean(axis=0)
    x = np.arange(len(keys))
    colors = [PALETTE[k] for k in keys]
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.7)
    ax.bar(
        x, means, width=0.32, color=colors, edgecolor=colors, linewidth=0.65, alpha=0.72, zorder=2
    )
    span = ylim[1] - ylim[0]
    for j, color in enumerate(colors):
        ax.text(
            x[j],
            means[j] + span * 0.026,
            f"{means[j]:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color=INK,
        )
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
    ax.set_ylabel("Mean Spearman ρ")
    ax.tick_params(axis="x", length=0, pad=8)
    # Axis-break marks make the nonzero baseline explicit.
    for offset in [0.0, -0.025]:
        ax.plot(
            [-0.013, 0.013],
            [offset - 0.01, offset + 0.01],
            transform=ax.transAxes,
            color=INK,
            linewidth=0.9,
            clip_on=False,
            zorder=8,
        )
    last = len(keys) - 1
    bracket = float(means.max()) + span * 0.17
    ax.plot(
        [control_index, control_index, last, last],
        [bracket - span * 0.028, bracket, bracket, bracket - span * 0.028],
        color=PALETTE["ensemble"],
        lw=0.95,
        clip_on=False,
    )
    delta = means[last] - means[control_index]
    ax.text(
        (control_index + last) / 2,
        bracket + span * 0.024,
        f"Δ = {delta:+.4f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        weight="bold",
        color=PALETTE["ensemble"],
    )


def overview(zero, spe):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.77, bottom=0.18, wspace=0.34)
    mean_panel(
        axes[0],
        zero,
        ["basic", "msa_tools", "structure_tools", "joint_tools"],
        ["Native\nCodex", "+ MSA", "+ structure", "+ MSA\n+ structure"],
        (0.48, 0.61),
        0,
        0.02,
        "%.2f",
    )
    mean_panel(
        axes[1],
        spe,
        ["direct", "cv", "ensemble"],
        ["Single native\nCodex", "CV-selected\nsingle program", "Scientific program\nensemble"],
        (0.575, 0.63),
        1,
        0.01,
        "%.2f",
    )
    heading(axes[0], "a", "Structure + MSA", "Zero-shot · 10 assays · 250 candidates per assay")
    heading(
        axes[1],
        "b",
        "Scientific Program Ensemble",
        "96 measured supports · 10 assays · 186 candidates per assay",
    )
    fig.text(
        0.075,
        0.055,
        "Bars: 10-assay means     Truncated y-axes: a, 0.480–0.610; b, 0.575–0.630",
        fontsize=8.5,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.014,
        "The two experiments use different cohorts and feedback; their gains are not additive.",
        fontsize=8.5,
        color=MUTED,
    )
    return save(fig, "results_overview")


def delta_panel(ax, cases, deltas, title, letter, subtitle):
    y = np.arange(len(cases))
    # Common range across panels avoids magnifying the smaller ensemble effect.
    ax.set_xlim(-0.115, 0.28)
    ax.set_ylim(-0.8, len(cases) - 0.2)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.65)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=2)
    for yi, delta in zip(y, deltas):
        color = PALETTE["ensemble"] if delta >= 0 else PALETTE["cv"]
        ax.plot([0, delta], [yi, yi], color=color, linewidth=2.1, alpha=0.7, zorder=3)
        ax.scatter([delta], [yi], color=color, s=48, edgecolors="white", linewidths=0.65, zorder=4)
        on_right = delta >= 0 or delta < -0.05
        x = delta + (0.009 if on_right else -0.009)
        number = f"{delta:+.4f}" if abs(delta) < 0.001 else f"{delta:+.3f}"
        ax.text(
            x, yi, number, ha="left" if on_right else "right", va="center", color=INK, fontsize=8.5
        )
    ax.set_yticks(y, [NAMES[c] for c in cases])
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Paired ΔSpearman ρ", labelpad=8)
    mean = deltas.mean()
    ax.text(
        0.98,
        1.075,
        f"Mean Δ = {mean:+.4f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        weight="bold",
        color=PALETTE["ensemble"],
    )
    # The common heading layout is adapted for the longer per-assay panels.
    ax.text(
        -0.30, 1.18, letter, transform=ax.transAxes, fontsize=17, weight="bold", color="#111111"
    )
    ax.text(0, 1.18, title, transform=ax.transAxes, fontsize=12, weight="bold", color="#111111")
    ax.text(0, 1.12, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED)


def by_assay(zero_cases, zero, spe_cases, spe):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.65))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.78, bottom=0.18, wspace=0.64)
    delta_panel(
        axes[0],
        zero_cases,
        zero[:, 3] - zero[:, 0],
        "Structure + MSA",
        "a",
        "Joint evidence − native Codex · 7/10 assays improved",
    )
    delta_panel(
        axes[1],
        spe_cases,
        spe[:, 2] - spe[:, 1],
        "Scientific Program Ensemble",
        "b",
        "Ensemble − same-budget CV single · 8/10 assays improved",
    )
    fig.text(
        0.14,
        0.059,
        "Each point = one assay     Pink: improvement     Blue: regression     Identical horizontal scales",
        fontsize=8.5,
        color=MUTED,
    )
    fig.text(
        0.14,
        0.019,
        "Descriptive paired differences from completed historical experiments; no new predictions.",
        fontsize=8.5,
        color=MUTED,
    )
    return save(fig, "results_by_assay")


def export_source(zero_cases, zero, spe_cases, spe):
    p = OUT / "figure_source_data.csv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "case_id",
                "assay_label",
                "method",
                "spearman",
                "delta_vs_primary_control",
                "measured_supports",
                "query_candidates",
                "source_file",
            ]
        )
        arms = ["basic", "msa_tools", "structure_tools", "joint_tools"]
        for i, case in enumerate(zero_cases):
            source = INPUTS[0] if i < 4 else INPUTS[1]
            for j, arm in enumerate(arms):
                writer.writerow(
                    [
                        "structure_msa_zero",
                        case,
                        NAMES[case],
                        arm,
                        repr(float(zero[i, j])),
                        repr(float(zero[i, j] - zero[i, 0])),
                        0,
                        250,
                        str(source.relative_to(ROOT)),
                    ]
                )
        for i, case in enumerate(spe_cases):
            for j, method in enumerate(["direct", "cv", "ensemble"]):
                writer.writerow(
                    [
                        "program_ensemble_feedback",
                        case,
                        NAMES[case],
                        method,
                        repr(float(spe[i, j])),
                        repr(float(spe[i, j] - spe[i, 1])),
                        96,
                        186,
                        str(INPUTS[2].relative_to(ROOT)),
                    ]
                )
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = style()
    zero_cases, zero, spe_cases, spe = load_data()
    files = overview(zero, spe) + by_assay(zero_cases, zero, spe_cases, spe)
    files.append(export_source(zero_cases, zero, spe_cases, spe))
    reference_hashes = read_json(OUT / "plot_manifest.json")["style"]["reference_sha256"]
    manifest = {
        "purpose": "EVAgent-style presentation of existing results",
        "new_model_calls": 0,
        "predictor_executions": 0,
        "test_labels_opened": False,
        "data_sources": {str(p.relative_to(ROOT)): sha256(p) for p in INPUTS},
        "style": {
            "font": font,
            "palette": PALETTE,
            "white_background": True,
            "reference": "EVAgent active figure2 and Extended Data 3: Arial, pastel pink/blue/gold/teal, thin axes, paired data points",
            "reference_sha256": reference_hashes,
        },
        "aggregation": "equal assay means; 10 assays per experiment; separate cohorts",
        "error_bars": "none; overview shows means only; the by-assay figure shows all paired differences",
        "overview_y_axes": {
            "truncated": True,
            "structure_msa": [0.48, 0.61],
            "program_ensemble": [0.575, 0.63],
            "bar_width": 0.32,
            "disclosure": "axis-break marks and explicit figure/README captions",
        },
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
    }
    (OUT / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved 2 figures (SVG/PNG), source CSV and provenance. No model/predictor calls.")


if __name__ == "__main__":
    main()
