"""Draw a compact comparison from the frozen server snapshot; no new predictions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from runtime.artifacts import ROOT, sha256, write_json
from scripts.audit_evagent import SOURCE, audit, read_csv
from scripts.plot_results import GRID, INK, MUTED, style

OUT = ROOT / "assets/evagent"


def panel(ax, task: str, assays: int) -> None:
    rows = sorted(
        read_csv(SOURCE / "zero_shot" / f"{task}_means.csv"),
        key=lambda row: -float(row["mean_spearman"]),
    )
    scores = [float(row["mean_spearman"]) for row in rows]
    labels = ["EVAgent / Gemini 3.8" if row["model"] == "Gemini" else row["model"] for row in rows]
    ax.barh(range(len(rows)), scores, height=0.40, color=[row["color"] for row in rows], alpha=0.86)
    for index, value in enumerate(scores):
        ax.text(
            value + 0.012,
            index,
            f"{value:.4f}",
            va="center",
            fontsize=10.5,
            weight="bold" if value == max(scores) else "normal",
            color=INK,
        )
    ax.set_yticks(range(len(rows)), labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 0.62)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.7)
    ax.axvline(0, color=MUTED, linewidth=0.7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel("Mean signed Spearman ρ", labelpad=9)
    ax.set_title(
        f"{'Protein' if task == 'protein' else 'RNA'} · {assays} common assays",
        loc="left",
        pad=18,
        fontsize=14,
        weight="bold",
    )
    for label, row in zip(ax.get_yticklabels(), rows, strict=True):
        if row["model"] == "Gemini":
            label.set_color("#AA4F65")
            label.set_weight("bold")


def main() -> None:
    verification = audit()
    font = style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0))
    fig.subplots_adjust(left=0.19, right=0.985, top=0.88, bottom=0.20, wspace=0.80)
    panel(axes[0], "protein", 201)
    panel(axes[1], "rna", 10)
    fig.text(
        0.19,
        0.075,
        "Gemini inputs: protein, sequence + MSA + structure; RNA, sequence + MSA.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.19,
        0.028,
        "250 candidates per assay · Retrospectively selected input configurations · All saved baselines retained",
        fontsize=9,
        color=MUTED,
    )
    files = []
    for suffix in ["png", "svg"]:
        path = OUT / f"evagent_overview.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.15)
        files.append(path)
    plt.close(fig)
    write_json(
        OUT / "plot_manifest.json",
        {
            "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
            "font": font,
            "source_csv_sha256": {
                f"{task}_means.csv": sha256(SOURCE / "zero_shot" / f"{task}_means.csv")
                for task in ["protein", "rna"]
            },
            "axis_range": [-0.03, 0.62],
            "bar_height": 0.40,
            "error_bars": "none; saved assay-equal means",
            "verification": verification,
            "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        },
    )
    print("Saved EVAgent overview from archived metrics; no inference or model fitting.")


if __name__ == "__main__":
    main()
