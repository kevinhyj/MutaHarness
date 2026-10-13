"""Combine three stored predictions without reading evaluator labels."""

import argparse
from pathlib import Path

from runtime.artifacts import read_json, write_json
from runtime.ranking import query_ids, rank_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; choose a new path")
    predictions = [read_json(path)["scores"] for path in args.candidate]
    ranked = rank_ensemble(query_ids(read_json(args.case)), predictions)
    output = {
        "scores": {candidate["id"]: candidate["score"] for candidate in ranked},
        "policy": "all-three uniform average-tie percentile ranks",
        "candidate_files": [str(path) for path in args.candidate],
        "model_or_oracle_calls": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, output)


if __name__ == "__main__":
    main()
