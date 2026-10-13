"""Run either packaged Harness workflow using Python's standard library."""

import argparse
import json
from pathlib import Path

from runtime.artifacts import ROOT
from runtime.workflows import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="workflow", required=True)
    for mode in ("ensemble", "evidence"):
        command = commands.add_parser(mode)
        command.add_argument(
            "--case", type=Path, help="JSON input; defaults to the packaged example"
        )
        command.add_argument("--out", type=Path, required=True, help="New output directory")
        if mode == "ensemble":
            command.add_argument("--predictions", type=Path, nargs=3, metavar="JSON")
    args = parser.parse_args()
    predictions = getattr(args, "predictions", None)
    if args.workflow == "ensemble":
        if (args.case is None) != (predictions is None):
            parser.error("For custom data, supply --case and exactly three --predictions together")
        case = args.case or ROOT / "inputs/gfp_s2501.json"
        predictions = predictions or [
            ROOT / "reference_predictions/gfp_s2501" / f"candidate_{index:02}.json"
            for index in range(3)
        ]
    else:
        case = args.case or ROOT / "examples/zero_shot/amfr/case.json"
        predictions = []
    try:
        result = run_workflow(args.workflow, case, args.out, predictions)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"status": "passed", "output": str(args.out.resolve()), **result}, indent=2))


if __name__ == "__main__":
    main()
