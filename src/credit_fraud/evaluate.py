"""Evaluation: confusion matrix, standard metrics, and calibration.

All numbers reported by ``train`` come from here and are computed at
runtime from whatever dataset was loaded. No metric is hardcoded or
quoted from a previous run: re-running on different data gives different
numbers, as it should.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import ModelConfig, default_config
from .model import FraudRiskModel
from .validation import ApplicationRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationBucket:
    """One probability bin: predicted vs observed fraud rate."""

    bin_start: float
    bin_end: float
    count: int
    mean_predicted: float
    observed_fraud_rate: float


@dataclass(frozen=True)
class EvaluationReport:
    """Everything ``train`` writes to its metrics file."""

    n_rows: int
    fraud_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    accuracy: float
    f1_score: float
    calibration: tuple[CalibrationBucket, ...]

    def to_dict(self) -> dict:
        """JSON-serialisable view."""
        return {
            "n_rows": self.n_rows,
            "fraud_rate": self.fraud_rate,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "true_negatives": self.true_negatives,
            },
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "calibration": [
                {
                    "bin_start": b.bin_start,
                    "bin_end": b.bin_end,
                    "count": b.count,
                    "mean_predicted": b.mean_predicted,
                    "observed_fraud_rate": b.observed_fraud_rate,
                }
                for b in self.calibration
            ],
        }


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    model: FraudRiskModel,
    rows: list[ApplicationRecord],
    config: ModelConfig | None = None,
) -> EvaluationReport:
    """Score ``rows`` with ``model`` and summarise the outcomes.

    The review threshold decides the predicted label; calibration bins
    are equal-width slices of predicted probability. A well-calibrated
    model has ``mean_predicted`` close to ``observed_fraud_rate`` in
    each bin that has enough rows to read.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("cannot evaluate an empty row list")
    cfg = config or default_config()

    tp = fp = fn = tn = 0
    scored: list[tuple[float, int]] = []
    for row in rows:
        result = model.score(row)
        scored.append((result.fraud_probability, row.fraud))
        predicted = result.review_required
        if predicted and row.fraud:
            tp += 1
        elif predicted and not row.fraud:
            fp += 1
        elif not predicted and row.fraud:
            fn += 1
        else:
            tn += 1

    n_rows = len(rows)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    accuracy = _safe_div(tp + tn, n_rows)
    f1_score = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    calibration: list[CalibrationBucket] = []
    n_bins = max(1, cfg.calibration_bins)
    for bin_index in range(n_bins):
        bin_start = bin_index / n_bins
        bin_end = (bin_index + 1) / n_bins
        bucket = [
            (prob, label)
            for prob, label in scored
            if bin_start <= prob < bin_end or (bin_index == n_bins - 1 and prob == 1.0)
        ]
        if not bucket:
            continue
        mean_predicted = sum(prob for prob, _ in bucket) / len(bucket)
        observed = sum(label for _, label in bucket) / len(bucket)
        calibration.append(
            CalibrationBucket(
                bin_start=round(bin_start, 2),
                bin_end=round(bin_end, 2),
                count=len(bucket),
                mean_predicted=round(mean_predicted, 4),
                observed_fraud_rate=round(observed, 4),
            )
        )

    report = EvaluationReport(
        n_rows=n_rows,
        fraud_rate=round(sum(row.fraud for row in rows) / n_rows, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        accuracy=round(accuracy, 4),
        f1_score=round(f1_score, 4),
        calibration=tuple(calibration),
    )
    logger.info(
        "evaluated %d rows: precision=%.3f recall=%.3f accuracy=%.3f",
        n_rows,
        report.precision,
        report.recall,
        report.accuracy,
    )
    return report
