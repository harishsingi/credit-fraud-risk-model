"""Interpretable fraud-risk scoring model.

A small logistic-style model in pure Python (no ML framework dependency).
Every score ships with per-feature contributions and its top drivers, so
a human reviewer can see *why* a case scored the way it did and override
it. The model supports a decision; it never makes a final eligibility
decision on its own.

Weights are illustrative, hand-picked for this synthetic demo, and are
not fitted to real data. See :mod:`credit_fraud.config`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import ModelConfig, default_config
from .features import build_features
from .validation import ApplicationInput, validate_application


@dataclass(frozen=True)
class ScoreResult:
    """The outcome of scoring one application."""

    fraud_probability: float
    top_drivers: list[str]
    review_required: bool
    contributions: dict[str, float] = field(default_factory=dict)
    model_version: str = ""

    def to_dict(self) -> dict:
        """JSON-serialisable view, as printed by the ``score`` CLI."""
        return {
            "fraud_probability": self.fraud_probability,
            "top_drivers": self.top_drivers,
            "review_required": self.review_required,
            "contributions": self.contributions,
            "model_version": self.model_version,
        }


class FraudRiskModel:
    """Logistic-style scorer configured by a :class:`ModelConfig`."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or default_config()

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid (guards against overflow)."""
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def score(self, application: ApplicationInput) -> ScoreResult:
        """Score one validated application.

        Args:
            application: A validated application (see
                :func:`credit_fraud.validation.validate_application`).

        Returns:
            A :class:`ScoreResult` with the fraud probability, the
            features ranked by their contribution to the score, and
            whether the illustrative review threshold was crossed.
        """
        cfg = self.config
        features = build_features(application, cfg)
        contributions = {
            "delinquencies": cfg.weight_delinquencies * features.delinquencies,
            "utilisation": cfg.weight_utilisation * features.utilisation,
            "low_income": cfg.weight_low_income if features.low_income else 0.0,
        }
        logit = cfg.intercept + sum(contributions.values())
        probability = self._sigmoid(logit)
        # Drivers are the features that actually pushed the score up,
        # ranked by contribution size; a zero contribution is not a driver.
        top_drivers = [
            name
            for name, _ in sorted(
                contributions.items(), key=lambda item: -item[1]
            )
            if contributions[name] > 0
        ]
        return ScoreResult(
            fraud_probability=round(probability, 4),
            top_drivers=top_drivers,
            review_required=probability > cfg.review_threshold,
            contributions={
                name: round(value, 4) for name, value in contributions.items()
            },
            model_version=_model_version(),
        )

    def score_raw(
        self,
        income: object,
        delinquencies: object,
        utilisation: object,
    ) -> ScoreResult:
        """Validate raw inputs, then score. Convenience wrapper for CLIs."""
        return self.score(validate_application(income, delinquencies, utilisation, self.config))


def _model_version() -> str:
    from .config import MODEL_VERSION

    return MODEL_VERSION
