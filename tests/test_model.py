"""Tests for the scoring model (credit_fraud.model)."""

import math
import unittest

from credit_fraud.config import ModelConfig
from credit_fraud.exceptions import ValidationError
from credit_fraud.model import FraudRiskModel
from credit_fraud.validation import ApplicationInput


def _app(income=90_000, delinquencies=0, utilisation=0.1):
    return ApplicationInput(income=income, delinquencies=delinquencies, utilisation=utilisation)


class TestFraudRiskModel(unittest.TestCase):
    def setUp(self):
        self.model = FraudRiskModel()

    def test_higher_risk_scores_higher(self):
        low = self.model.score(_app()).fraud_probability
        high = self.model.score(_app(30_000, 3, 0.95)).fraud_probability
        self.assertLess(low, high)

    def test_probability_is_bounded(self):
        for app in (_app(0, 50, 1.0), _app(10_000_000, 0, 0.0), _app(30_000, 3, 0.95)):
            prob = self.model.score(app).fraud_probability
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_low_risk_below_review_threshold_high_risk_above(self):
        self.assertFalse(self.model.score(_app()).review_required)
        self.assertTrue(self.model.score(_app(30_000, 3, 0.95)).review_required)

    def test_probability_matches_hand_computed_logit(self):
        # z = -2.2 + 1.1*0 + 1.6*0.1 + 0 = -2.04
        expected = 1 / (1 + math.exp(2.04))
        self.assertAlmostEqual(
            self.model.score(_app()).fraud_probability, round(expected, 4), places=4
        )

    def test_contributions_sum_to_logit_minus_intercept(self):
        result = self.model.score(_app(30_000, 2, 0.5))
        total = sum(result.contributions.values())
        # z = intercept + contributions; p = sigmoid(z)
        z = self.model.config.intercept + total
        expected = 1 / (1 + math.exp(-z))
        self.assertAlmostEqual(result.fraud_probability, round(expected, 4), places=4)
        self.assertEqual(result.contributions["low_income"], 0.5)
        self.assertEqual(result.contributions["delinquencies"], 2.2)

    def test_top_drivers_ranked_by_contribution_and_exclude_zero(self):
        result = self.model.score(_app(90_000, 1, 0.9))
        # delinquencies contributes 1.1, utilisation 1.44, low_income 0
        self.assertEqual(result.top_drivers, ["utilisation", "delinquencies"])
        self.assertNotIn("low_income", result.top_drivers)

    def test_clean_application_has_no_drivers(self):
        result = self.model.score(_app(90_000, 0, 0.0))
        self.assertEqual(result.top_drivers, [])

    def test_score_is_deterministic(self):
        app = _app(45_000, 2, 0.7)
        self.assertEqual(self.model.score(app), self.model.score(app))

    def test_custom_config_changes_score(self):
        strict = FraudRiskModel(ModelConfig(review_threshold=0.99))
        result = strict.score(_app(30_000, 3, 0.95))
        self.assertFalse(result.review_required)
        self.assertGreater(result.fraud_probability, 0.9)

    def test_score_raw_validates_inputs(self):
        with self.assertRaises(ValidationError):
            self.model.score_raw(-1, 0, 0.1)
        with self.assertRaises(ValidationError):
            self.model.score_raw(50_000, 0, 1.5)

    def test_score_raw_accepts_valid_inputs(self):
        result = self.model.score_raw(72_000, 0, 0.31)
        self.assertIsInstance(result.fraud_probability, float)

    def test_model_version_reported(self):
        from credit_fraud.config import MODEL_VERSION

        self.assertEqual(self.model.score(_app()).model_version, MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
