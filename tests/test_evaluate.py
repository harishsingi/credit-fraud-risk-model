"""Tests for evaluation (credit_fraud.evaluate)."""

import unittest

from credit_fraud.evaluate import evaluate
from credit_fraud.model import FraudRiskModel
from credit_fraud.validation import ApplicationRecord


def _record(income, delinquencies, utilisation, fraud):
    return ApplicationRecord(
        income=income, delinquencies=delinquencies,
        utilisation=utilisation, fraud=fraud,
    )


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.model = FraudRiskModel()

    def test_hand_computed_confusion_matrix(self):
        # score(90000, 0, 0.1): z=-2.04, p~0.115 -> no review
        # score(30000, 3, 0.95): z=3.12, p~0.958 -> review
        rows = [
            _record(30_000, 3, 0.95, 1),  # TP
            _record(30_000, 3, 0.95, 1),  # TP
            _record(30_000, 3, 0.95, 0),  # FP
            _record(90_000, 0, 0.10, 1),  # FN
            _record(90_000, 0, 0.10, 0),  # TN
            _record(90_000, 0, 0.10, 0),  # TN
        ]
        report = evaluate(self.model, rows)
        self.assertEqual(report.n_rows, 6)
        self.assertEqual(report.true_positives, 2)
        self.assertEqual(report.false_positives, 1)
        self.assertEqual(report.false_negatives, 1)
        self.assertEqual(report.true_negatives, 2)
        self.assertAlmostEqual(report.precision, 2 / 3, places=4)
        self.assertAlmostEqual(report.recall, 2 / 3, places=4)
        self.assertAlmostEqual(report.accuracy, 4 / 6, places=4)
        self.assertAlmostEqual(report.f1_score, 2 / 3, places=4)
        self.assertAlmostEqual(report.fraud_rate, 3 / 6, places=4)

    def test_calibration_buckets_cover_all_rows(self):
        rows = [
            _record(30_000, 3, 0.95, 1),
            _record(90_000, 0, 0.10, 0),
            _record(50_000, 1, 0.50, 0),
            _record(80_000, 0, 0.20, 0),
        ]
        report = evaluate(self.model, rows)
        total = sum(bucket.count for bucket in report.calibration)
        self.assertEqual(total, 4)
        for bucket in report.calibration:
            self.assertGreaterEqual(bucket.mean_predicted, bucket.bin_start)
            self.assertLessEqual(bucket.mean_predicted, bucket.bin_end)
            self.assertGreaterEqual(bucket.observed_fraud_rate, 0.0)
            self.assertLessEqual(bucket.observed_fraud_rate, 1.0)

    def test_no_positive_predictions_gives_zero_precision_and_recall(self):
        rows = [_record(90_000, 0, 0.10, 1), _record(90_000, 0, 0.10, 0)]
        report = evaluate(self.model, rows)
        self.assertEqual(report.precision, 0.0)
        self.assertEqual(report.recall, 0.0)
        self.assertEqual(report.f1_score, 0.0)

    def test_empty_rows_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.model, [])

    def test_report_serialises_to_json(self):
        import json

        report = evaluate(self.model, [_record(90_000, 0, 0.10, 0)])
        payload = json.dumps(report.to_dict())
        self.assertIn("confusion_matrix", json.loads(payload))


if __name__ == "__main__":
    unittest.main()
