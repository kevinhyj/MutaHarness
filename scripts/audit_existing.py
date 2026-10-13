"""Check frozen hashes and stored-result arithmetic without executing predictors."""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from runtime.artifacts import ROOT, read_json, verify_hash, write_json
from runtime.ranking import query_ids, validate_scores
from scripts.results import ENSEMBLE_SOURCE, ZERO_SOURCES


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_frozen(manifest: dict) -> int:
    count = 0
    for condition, details in manifest["conditions"].items():
        input_path = ROOT / "inputs" / f"{condition}.json"
        verify_hash(input_path, details["input_sha256"])
        case = read_json(input_path)
        ids = query_ids(case)
        check(len(case["support"]) == 96 and len(ids) == 186, f"Invalid case size: {condition}")
        for index, candidate in enumerate(details["candidates"]):
            for filename, digest in candidate["code_sha256"].items():
                verify_hash(
                    ROOT / "programs" / condition / f"candidate_{index:02}" / filename, digest
                )
            prediction = read_json(
                ROOT / "reference_predictions" / condition / f"candidate_{index:02}.json"
            )
            validate_scores(prediction["scores"], ids, bounded=True)
            count += 1
    return count


def audit_metrics(pooled: dict) -> list[dict]:
    for key in ["delta_vs_direct00", "delta_vs_cv_single"]:
        delta = mean(row[key] for row in pooled["rows"])
        check(
            abs(delta - pooled["comparisons"][key]["mean_delta"]) < 1e-12, f"Incorrect mean: {key}"
        )
    zero_rows = []
    for source in ZERO_SOURCES:
        summary = read_json(ROOT / source)
        check(
            summary["complete_conditions"] == summary["planned_conditions"],
            f"Incomplete campaign: {source}",
        )
        check(
            summary["independent_arithmetic_verified"],
            f"Missing independent verification: {source}",
        )
        for row in summary["rows"]:
            check(
                abs(row["spearman"] - row["independent_spearman"]) < 1e-12,
                f"Spearman mismatch: {row['case_id']}, {row['arm']}",
            )
        zero_rows.extend(summary["rows"])
    for arm, expected in [("basic", 0.5050867801038631), ("joint_tools", 0.5735689527270961)]:
        value = mean(row["spearman"] for row in zero_rows if row["arm"] == arm)
        check(abs(value - expected) < 1e-12, f"Incorrect zero-shot mean: {arm}")
    return zero_rows


def reference_rank(scores: dict, value: float) -> float:
    # Deliberately independent of the sorting-based implementation in runtime/ranking.py.
    lower = sum(score < value for score in scores.values())
    equal = sum(score == value for score in scores.values())
    return (lower + equal / 2) / len(scores)


def audit_candidates(case: dict, input_case: dict) -> int:
    ids = query_ids(input_case)
    raw = [
        read_json(ROOT / "reference_predictions" / case["id"] / f"candidate_{index:02}.json")[
            "scores"
        ]
        for index in range(3)
    ]
    candidates = {candidate["id"]: candidate for candidate in case["candidates"]}
    check(
        len(candidates) == len(case["candidates"]) == 186, f"Invalid candidate count: {case['id']}"
    )
    check(set(candidates) == set(ids), f"Candidate IDs mismatch: {case['id']}")
    for query in input_case["query"]:
        candidate = candidates[query["id"]]
        scores = [prediction[query["id"]] for prediction in raw]
        check(candidate["mutation"] == query["mutation"], f"Mutation mismatch: {query['id']}")
        check(scores == candidate["program_scores"], f"Program scores mismatch: {query['id']}")
        votes = [
            reference_rank(prediction, score) for prediction, score in zip(raw, scores, strict=True)
        ]
        check(
            abs(candidate["score"] - (2 * mean(votes) - 1)) < 1e-12,
            f"Fusion mismatch: {query['id']}",
        )
        check(
            abs(candidate["disagreement"] - pstdev(votes)) < 1e-12,
            f"Disagreement mismatch: {query['id']}",
        )
    return len(candidates)


def audit_viewer(manifest: dict, pooled: dict, zero_rows: list[dict]) -> int:
    script = (ROOT / "demo/data.js").read_text(encoding="utf-8")
    prefix = "window.MUTA_DATA = "
    check(script.startswith(prefix) and script.endswith(";\n"), "Invalid viewer data envelope")
    viewer = json.loads(script[len(prefix) : -2])
    for mode, feedback, queries in [("zero", 0, 250), ("ensemble", 96, 186)]:
        group = viewer[mode]
        check(
            group["feedback_count"] == feedback and group["query_count"] == queries,
            f"Incorrect scope: {mode}",
        )
        check(
            len(group["cases"]) == len({case["id"] for case in group["cases"]}) == 10,
            f"Incorrect cohort: {mode}",
        )
        for method, value in group["means"].items():
            expected = mean(case["scores"][method] for case in group["cases"])
            check(abs(value - expected) < 1e-12, f"Incorrect viewer mean: {mode}, {method}")
    zero_scores = {(row["case_id"], row["arm"]): row["spearman"] for row in zero_rows}
    for case in viewer["zero"]["cases"]:
        for arm, value in case["scores"].items():
            check(
                value == zero_scores[case["id"], arm],
                f"Incorrect viewer metric: {case['id']}, {arm}",
            )
        check((ROOT / case["source"]).is_file(), f"Missing source: {case['source']}")
    for arm, value in viewer["zero"]["top25_means"].items():
        expected = mean(row["top25_overlap"] for row in zero_rows if row["arm"] == arm)
        check(abs(value - expected) < 1e-12, f"Incorrect Top-25 mean: {arm}")
    count = 0
    ensemble_rows = {row["case_id"]: row for row in pooled["rows"]}
    for case in viewer["ensemble"]["cases"]:
        input_case = read_json(ROOT / "inputs" / f"{case['id']}.json")
        row = ensemble_rows[manifest["conditions"][case["id"]]["case_id"]]
        expected = {
            "direct": row["candidate_rho"][0],
            "cv": row["cv_selected_single_rho"],
            "ensemble": row["ensemble_rho"],
        }
        check(case["scores"] == expected, f"Incorrect viewer metrics: {case['id']}")
        check(
            case["cv_selected"] == row["cv_selected_candidate"],
            f"Incorrect CV selection: {case['id']}",
        )
        count += audit_candidates(case, input_case)
    return count


def audit() -> dict:
    manifest = read_json(ROOT / "artifacts_manifest.json")
    programs = audit_frozen(manifest)
    pooled = read_json(ROOT / ENSEMBLE_SOURCE)
    zero_rows = audit_metrics(pooled)
    candidates = audit_viewer(manifest, pooled, zero_rows)
    return {
        "status": "passed",
        "frozen_protein_conditions": 10,
        "programs_hash_verified": programs,
        "existing_zero_shot_conditions": len(zero_rows),
        "stored_metric_arithmetic_verified": True,
        "viewer_metric_values_verified": 70,
        "viewer_candidates_verified": candidates,
        "predictor_executions": 0,
        "new_model_calls": 0,
        "test_labels_opened": False,
        "scope": "hash/contract checks and stored-metric arithmetic; not fresh inference or full experiment reproduction",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
