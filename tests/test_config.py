"""Tests for configuration (credit_fraud.config)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from credit_fraud.config import default_config, load_config
from credit_fraud.exceptions import ConfigurationError


class TestConfig(unittest.TestCase):
    def test_defaults_match_documented_values(self):
        config = default_config()
        self.assertEqual(config.intercept, -2.2)
        self.assertEqual(config.weight_delinquencies, 1.1)
        self.assertEqual(config.weight_utilisation, 1.6)
        self.assertEqual(config.weight_low_income, 0.5)
        self.assertEqual(config.low_income_cutoff, 35_000)
        self.assertEqual(config.review_threshold, 0.25)
        self.assertEqual(config.synthetic_rows, 2_000)
        self.assertEqual(config.synthetic_seed, 11)

    def test_paths_resolve_under_project_root(self):
        config = default_config()
        self.assertTrue(str(config.data_path).endswith("data/applications.csv"))
        self.assertTrue(str(config.quarantine_path).endswith("data/quarantine.csv"))
        self.assertTrue(str(config.metrics_path).endswith("reports/metrics.json"))
        self.assertTrue(config.data_path.is_absolute())

    def test_json_override_applies(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"review_threshold": 0.5, "synthetic_rows": 10}),
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.review_threshold, 0.5)
        self.assertEqual(config.synthetic_rows, 10)
        self.assertEqual(config.intercept, -2.2)  # untouched defaults stay

    def test_unknown_key_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"not_a_setting": 1}), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_wrong_type_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"review_threshold": "high"}), encoding="utf-8"
            )
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_missing_file_rejected(self):
        with self.assertRaises(ConfigurationError):
            load_config("/nonexistent/config.json")

    def test_non_object_json_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_no_path_gives_defaults(self):
        self.assertEqual(load_config(None), default_config())


if __name__ == "__main__":
    unittest.main()
