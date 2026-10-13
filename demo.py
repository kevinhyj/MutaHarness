"""Optional cold replay: refit frozen programs and compare with their stored predictions."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from runtime.artifacts import ROOT, read_json, verify_hash, write_json
from runtime.ranking import query_ids, rank_ensemble, write_ranking


def stage_program(source: Path, workspace: Path, hashes: dict[str, str]) -> None:
    for filename, digest in hashes.items():
        verify_hash(source / filename, digest)
        target = workspace / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / filename, target)


def replay_program(
    condition: str, index: int, hashes: dict[str, str], workspace: Path, output: Path
) -> dict:
    stage_program(ROOT / "programs" / condition / f"candidate_{index:02}", workspace, hashes)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "runtime.replay_one",
            "--case",
            str(ROOT / "inputs" / f"{condition}.json"),
            "--module",
            str(workspace / "proposal.py"),
            "--out",
            str(output),
        ],
        cwd=workspace,
        env=env,
        timeout=120,
        check=True,
    )
    prediction = read_json(output)
    reference = read_json(ROOT / "reference_predictions" / condition / f"candidate_{index:02}.json")
    if prediction != reference:
        raise ValueError(
            f"Stored predictions did not reproduce exactly: {condition}, program {index}"
        )
    for filename, digest in hashes.items():
        verify_hash(workspace / filename, digest)
    return prediction["scores"]


def replay(condition: str, output: Path) -> None:
    details = read_json(ROOT / "artifacts_manifest.json")["conditions"][condition]
    input_path = ROOT / "inputs" / f"{condition}.json"
    verify_hash(input_path, details["input_sha256"])
    case = read_json(input_path)
    ids = query_ids(case)
    if len(case["support"]) != 96 or len(ids) != 186:
        raise ValueError("Packaged cases require 96 supports and 186 queries")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="mutaharness-") as temporary:
        predictions = [
            replay_program(
                condition,
                index,
                candidate["code_sha256"],
                Path(temporary) / f"candidate_{index:02}",
                output / f"candidate_{index:02}.json",
            )
            for index, candidate in enumerate(details["candidates"])
        ]
    ranked = rank_ensemble(ids, predictions)
    write_json(
        output / "ensemble.json",
        {"scores": {row["id"]: row["score"] for row in ranked}, "selected_ids": []},
    )
    write_ranking(output / "ranking.csv", ranked)
    write_json(
        output / "verification.json",
        {
            "condition": condition,
            "programs": 3,
            "all_program_hashes_verified": True,
            "predictions_match_stored_results": True,
            "support_count": 96,
            "query_count": 186,
            "new_model_calls": 0,
            "evaluator_labels_opened": False,
        },
    )
    print(
        "PASS: frozen programs refitted and matched stored predictions; no model calls or evaluator labels."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="gfp_s2501",
        choices=read_json(ROOT / "artifacts_manifest.json")["conditions"],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(args.case, args.out)


if __name__ == "__main__":
    main()
