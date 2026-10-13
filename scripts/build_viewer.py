"""Render the report and offline viewer from historical artifacts."""

import json
from string import Template

from runtime.artifacts import ROOT
from scripts.results import load_viewer


def result_table(group: dict, headers: list[str], control: str, improved: str) -> str:
    methods = list(group["means"])
    lines = ["| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1)]
    for case in group["cases"]:
        scores = case["scores"]
        best = max(scores.values())
        cells = [
            f"**{scores[key]:.4f}**" if abs(scores[key] - best) < 1e-12 else f"{scores[key]:.4f}"
            for key in methods
        ]
        delta = scores[improved] - scores[control]
        lines.append("| " + " | ".join([case["label"], *cells, f"{delta:+.4f}"]) + " |")
    means = group["means"]
    cells = [f"**{means[key]:.4f}**" for key in methods]
    delta = means[improved] - means[control]
    lines.append("| " + " | ".join(["**平均**", *cells, f"**{delta:+.4f}**"]) + " |")
    return "\n".join(lines)


def main() -> None:
    data = load_viewer()
    tables = {
        "zero_table": result_table(
            data["zero"],
            ["assay", "基础Codex", "仅MSA", "仅结构", "联合", "联合−基础"],
            "basic",
            "joint_tools",
        ),
        "ensemble_table": result_table(
            data["ensemble"],
            ["assay", "单次Codex", "同预算CV选一个", "三程序集成", "集成−CV单程序"],
            "cv",
            "ensemble",
        ),
    }
    template = Template((ROOT / "docs/results.template.md").read_text(encoding="utf-8"))
    (ROOT / "docs/results.md").write_text(template.substitute(tables), encoding="utf-8")
    (ROOT / "demo/data.js").write_text(
        "window.MUTA_DATA = " + json.dumps(data, ensure_ascii=False, allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    print("Built report and viewer from existing artifacts; no predictor or model execution.")


if __name__ == "__main__":
    main()
