"""Exercise the public CLI, provenance and failure behavior without scientific runs."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from runtime.artifacts import ROOT, read_json, sha256, write_json
from runtime.workflows import run_workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "harness.py"), *map(str, args)],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_both_examples_work_outside_repository_and_record_hashes(self):
        for mode, count in (("ensemble", 186), ("evidence", 250)):
            with self.subTest(mode=mode):
                out = self.root / mode
                process = self.cli(mode, "--out", out)
                self.assertEqual(process.returncode, 0, process.stderr)
                manifest = read_json(out / "manifest.json")
                self.assertEqual(manifest["query_count"], count)
                self.assertEqual(manifest["new_model_calls"], 0)
                for source in manifest["inputs"]:
                    self.assertEqual(sha256(Path(source["path"])), source["sha256"])
                for name, digest in manifest["artifacts"].items():
                    self.assertEqual(sha256(out / name), digest)
                if mode == "ensemble":
                    with (out / "ranking.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    scores = [float(row["ensemble_score"]) for row in rows]
                    self.assertEqual(len(rows), count)
                    self.assertEqual(scores, sorted(scores, reverse=True))
                else:
                    self.assertEqual(len(read_json(out / "msa.json")["candidates"]), count)

    def test_custom_case_never_falls_back_to_packaged_predictions(self):
        process = self.cli("ensemble", "--case", "custom.json", "--out", self.root / "out")
        self.assertEqual(process.returncode, 2)
        self.assertIn("together", process.stderr)
        self.assertFalse((self.root / "out").exists())

    def test_existing_output_is_preserved(self):
        sentinel = self.root / "sentinel.txt"
        sentinel.write_text("keep")
        process = self.cli("evidence", "--out", self.root)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(self.root.iterdir()), [sentinel])

    def test_invalid_evidence_leaves_no_partial_output(self):
        case = read_json(ROOT / "examples/zero_shot/amfr/case.json")
        case["evidence"]["sites"][0]["WT"] = "X"
        path = self.root / "case.json"
        write_json(path, case)
        with self.assertRaises(ValueError):
            run_workflow("evidence", path, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_custom_prediction_ids_are_aligned_and_bad_inputs_rejected(self):
        case = self.root / "case.json"
        write_json(case, {"query": [{"id": "a"}, {"id": "b"}]})
        predictions = [self.root / f"p{i}.json" for i in range(3)]
        for i, path in enumerate(predictions):
            write_json(path, {"scores": {"b": 10 * (i + 1), "a": 0}})
        out = self.root / "good"
        run_workflow("ensemble", case, out, predictions)
        self.assertEqual(read_json(out / "ensemble.json")["scores"], {"a": -0.5, "b": 0.5})
        for malformed in ({"a": 0}, {"a": 0, "b": True}, [0, 1]):
            write_json(predictions[0], {"scores": malformed})
            with self.subTest(scores=malformed), self.assertRaises(ValueError):
                run_workflow("ensemble", case, self.root / "bad", predictions)
            self.assertFalse((self.root / "bad").exists())

    def test_query_labels_are_rejected_before_tool_execution(self):
        path = self.root / "leaked.json"
        write_json(path, {"query": [{"id": "a", "fitness": 1}]})
        with self.assertRaisesRegex(ValueError, "fitness labels"):
            run_workflow("evidence", path, self.root / "out")
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
