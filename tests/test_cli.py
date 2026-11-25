"""End-to-end CLI tests: generate -> train -> score, via subprocess.

These run the real entry points (``python -m credit_fraud.<module>``)
in a temporary directory so generated files never touch the repo.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(SRC_DIR)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", *args],
            cwd=self.dir,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_full_pipeline_generate_train_score(self):
        data_path = self.dir / "applications.csv"
        gen = self._run(
            "credit_fraud.generate_data",
            "--rows", "200", "--seed", "11", "--out", str(data_path),
        )
        self.assertEqual(gen.returncode, 0, gen.stderr)
        self.assertTrue(data_path.is_file())

        metrics_path = self.dir / "metrics.json"
        train = self._run(
            "credit_fraud.train",
            "--data", str(data_path), "--out", str(metrics_path),
            "--quarantine-out", str(self.dir / "quarantine.csv"),
        )
        self.assertEqual(train.returncode, 0, train.stderr)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["rows_read"], 200)
        self.assertEqual(metrics["rows_valid"], 200)
        self.assertEqual(metrics["rows_quarantined"], 0)
        for key in ("precision", "recall", "accuracy", "f1_score", "calibration"):
            self.assertIn(key, metrics)
        self.assertIn("confusion_matrix", metrics)

        score = self._run(
            "credit_fraud.score",
            "--income", "72000", "--delinquencies", "0",
            "--utilisation", "0.31",
        )
        self.assertEqual(score.returncode, 0, score.stderr)
        result = json.loads(score.stdout)
        self.assertIn("fraud_probability", result)
        self.assertIn("top_drivers", result)
        self.assertIn("review_required", result)
        self.assertFalse(result["review_required"])

    def test_score_invalid_input_exits_nonzero(self):
        result = self._run(
            "credit_fraud.score",
            "--income", "72000", "--delinquencies", "0",
            "--utilisation", "1.5",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("utilisation", result.stderr)

    def test_train_missing_data_file_exits_nonzero(self):
        result = self._run(
            "credit_fraud.train",
            "--data", str(self.dir / "missing.csv"),
            "--out", str(self.dir / "metrics.json"),
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stderr)

    def test_generate_is_idempotent_for_same_seed(self):
        first = self.dir / "first.csv"
        second = self.dir / "second.csv"
        self._run("credit_fraud.generate_data", "--rows", "50", "--seed", "5",
                  "--out", str(first))
        self._run("credit_fraud.generate_data", "--rows", "50", "--seed", "5",
                  "--out", str(second))
        self.assertEqual(
            first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8")
        )


if __name__ == "__main__":
    unittest.main()
