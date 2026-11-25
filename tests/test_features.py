"""Tests for feature building (credit_fraud.features)."""

import unittest

from credit_fraud.config import ModelConfig
from credit_fraud.features import build_features
from credit_fraud.validation import ApplicationInput


class TestBuildFeatures(unittest.TestCase):
    def test_low_income_flag_boundary(self):
        self.assertTrue(
            build_features(ApplicationInput(34_999, 0, 0.1)).low_income
        )
        self.assertFalse(
            build_features(ApplicationInput(35_000, 0, 0.1)).low_income
        )

    def test_custom_cutoff_respected(self):
        config = ModelConfig(low_income_cutoff=50_000)
        self.assertTrue(
            build_features(ApplicationInput(40_000, 0, 0.1), config).low_income
        )
        self.assertFalse(
            build_features(ApplicationInput(60_000, 0, 0.1), config).low_income
        )

    def test_features_pass_through(self):
        features = build_features(ApplicationInput(80_000, 2, 0.75))
        self.assertEqual(features.delinquencies, 2)
        self.assertEqual(features.utilisation, 0.75)
        self.assertFalse(features.low_income)
        self.assertEqual(
            features.as_dict(),
            {"delinquencies": 2, "utilisation": 0.75, "low_income": False},
        )


if __name__ == "__main__":
    unittest.main()
