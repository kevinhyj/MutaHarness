"""Verify the imported EVAgent snapshot using saved metrics, never evaluator labels."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from runtime.artifacts import ROOT, read_json, verify_hash

SOURCE = ROOT / "evidence/evagent"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_zero(task: str, assays: int, methods: int) -> None:
    summaries = read_csv(SOURCE / "zero_shot" / f"{task}_means.csv")
    rows = read_csv(SOURCE / "zero_shot" / f"{task}_per_assay.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
        value = float(row["spearman"])
        check(math.isfinite(value) and -1 <= value <= 1, "Invalid signed Spearman")
        check(row["n"] == "250" and row["seed"] == "42", "Unexpected zero-shot protocol")
    check(len(summaries) == len(grouped) == methods, f"Incorrect method count: {task}")
    cohort = {row["assay_id"] for row in rows}
    check(len(cohort) == assays, f"Incorrect common cohort: {task}")
    for summary in summaries:
        model = summary["model"]
        records = grouped[model]
        check(len(records) == assays, f"Missing or duplicate assay: {task}, {model}")
        check({row["assay_id"] for row in records} == cohort, f"Unmatched cohort: {model}")
        check(int(summary["common_assays"]) == assays, f"Incorrect denominator: {model}")
        value = mean(float(row["spearman"]) for row in records)
        check(abs(value - float(summary["mean_spearman"])) < 1e-12, f"Incorrect mean: {model}")


def audit_feedback(task: str, assays: int, methods: int) -> int:
    rows = read_csv(SOURCE / "feedback" / f"{task}_per_assay_rounds.csv")
    if task == "rna":
        rows = [row for row in rows if row["cohort"] == "rna_common"]
    display_file = "ppi_curve_statistics.csv" if task == "ppi" else f"{task}_display_series.csv"
    display = read_csv(SOURCE / "feedback" / display_file)
    grouped = defaultdict(list)
    for row in rows:
        round_ = int(row["round"])
        check(1 <= round_ <= 5, "Invalid feedback round")
        if task != "ppi":
            check(int(row["support_before"]) == 32 * round_, "Incorrect feedback size")
            check(int(row["unique_labels_after"]) == 32 * (round_ + 1), "Incorrect label budget")
            check(row["selected"] == "32", "Incorrect selection budget")
        check(0 <= float(row["hit_rate_percent"]) <= 100, "Invalid hit rate")
        grouped[row["method"], round_].append(row)
    cohort = {row["assay"] for row in rows}
    check(len(cohort) == assays and len(rows) == assays * 5 * methods, "Incorrect feedback cohort")
    check(len(grouped) == 5 * methods, "Incorrect feedback method or round count")
    for records in grouped.values():
        check(len(records) == assays and {row["assay"] for row in records} == cohort, "Unpaired round")
    for series in display:
        records = grouped[series["method"], int(series["round"])]
        column = (
            "cumulative_unique_hits"
            if series.get("metric") == "cumulative"
            else "hit_rate_percent"
        )
        values = [float(row[column]) for row in records]
        mean_key = {
            "protein": "actual_mean",
            "rna": "mean_percent",
            "ppi": "observed_mean_percent",
        }[task]
        sem_key = "sem" if task == "protein" else "sem_percent"
        check(abs(mean(values) - float(series[mean_key])) < 1e-10, "Incorrect feedback mean")
        check(abs(stdev(values) / math.sqrt(assays) - float(series[sem_key])) < 1e-10, "Incorrect SEM")
    return len(rows)


def audit() -> dict:
    manifest = read_json(SOURCE / "manifest.json")
    for filename, record in manifest["files"].items():
        verify_hash(ROOT / filename, record["sha256"])
    verify_hash(ROOT / manifest["hero"]["path"], manifest["hero"]["sha256"])
    audit_zero("protein", assays=201, methods=10)
    audit_zero("rna", assays=10, methods=9)
    feedback_rows = sum(
        audit_feedback(task, assays, methods)
        for task, assays, methods in (("protein", 47, 8), ("rna", 18, 9), ("ppi", 37, 7))
    )
    return {
        "status": "passed",
        "source_files_hash_verified": len(manifest["files"]),
        "protein_zero_assays": 201,
        "rna_zero_assays": 10,
        "zero_metric_rows_verified": 2100,
        "protein_feedback_assays": 47,
        "rna_feedback_assays": 18,
        "ppi_feedback_assays": 37,
        "feedback_metric_rows_verified": feedback_rows,
        "hero_pixels": manifest["hero"]["pixels"],
        "new_model_calls": 0,
        "predictor_executions": 0,
        "test_labels_opened": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(audit(), indent=2))
