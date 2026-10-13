"""Label-free workflow entry points shared by the CLI and repository Skills."""

from collections.abc import Sequence
from pathlib import Path

from runtime.artifacts import read_json, sha256, write_json
from runtime.ranking import query_ids, rank_ensemble, write_ranking
from tools.msa_tools import score_case
from tools.structure_tools import describe_case


def run_workflow(
    mode: str, case_path: Path, output: Path, predictions: Sequence[Path] = ()
) -> dict:
    """Validate and compute before creating a new output directory."""
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    case = read_json(case_path)
    ids = query_ids(case)
    sources = [case_path, *predictions]
    if mode == "ensemble":
        ranked = rank_ensemble(ids, [read_json(path)["scores"] for path in predictions])
        results = {
            "ensemble.json": {
                "scores": {row["id"]: row["score"] for row in ranked},
                "policy": "all-three uniform average-tie percentile ranks",
                "score_direction": "higher is preferred",
            }
        }
    elif mode == "evidence":
        if case.get("support"):
            raise ValueError("Zero-shot evidence mode requires an empty measured support set")
        if predictions:
            raise ValueError("Evidence mode does not consume predictions")
        results = {"msa.json": score_case(case), "structure.json": describe_case(case)}
    else:
        raise ValueError(f"Unknown workflow: {mode}")
    manifest = {
        "format": "mutaharness_run_v1",
        "workflow": mode,
        "operation": "saved_prediction_fusion" if mode == "ensemble" else "evidence_analysis",
        "case_id": case.get("case_id"),
        "support_count": len(case.get("support", [])),
        "query_count": len(ids),
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in sources],
        "new_model_calls": 0,
        "predictor_executions": 0,
        "artifacts": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in results.items():
        write_json(output / name, data)
    if mode == "ensemble":
        write_ranking(output / "ranking.csv", ranked)
    manifest["artifacts"] = {path.name: sha256(path) for path in sorted(output.iterdir())}
    write_json(output / "manifest.json", manifest)
    return manifest
