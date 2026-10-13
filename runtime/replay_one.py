"""Execute one frozen program in its workspace and enforce the prediction contract."""

import argparse
import importlib.util
import sys
from pathlib import Path

from runtime.artifacts import read_json, write_json
from runtime.ranking import query_ids, validate_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    case = read_json(args.case)
    ids = query_ids(case)
    assignment = {
        "round": 3,
        "arm": "direct",
        "batch_size": 0,
        "eligible_ids": ids,
        "audit_ids": ids,
        "remaining_budget": 0,
    }
    sys.path.insert(0, str(args.module.parent))
    spec = importlib.util.spec_from_file_location("frozen_predictor", args.module)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load predictor: {args.module}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.propose(case, assignment)
    validate_scores(output["scores"], ids, bounded=True)
    if output.get("selected_ids", []) != []:
        raise ValueError("Replay predicts all queries without selecting new experiments")
    write_json(args.out, {"scores": output["scores"], "selected_ids": []})


if __name__ == "__main__":
    main()
