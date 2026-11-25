"""Feature building: turn a validated application into model features.

This module is the single place where raw inputs become features, so the
scoring model never re-derives them and the definitions stay consistent
between training/evaluation and live scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig, default_config
from .validation import ApplicationInput


@dataclass(frozen=True)
class Features:
    """Numeric features the scoring model consumes."""

    delinquencies: int
    utilisation: float
    low_income: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        """Plain mapping, useful for logging and debugging."""
        return {
            "delinquencies": self.delinquencies,
            "utilisation": self.utilisation,
            "low_income": self.low_income,
        }


def build_features(
    application: ApplicationInput,
    config: ModelConfig | None = None,
) -> Features:
    """Derive model features from a validated application.

    The only derived feature is ``low_income``: 1 when annual income is
    below ``config.low_income_cutoff``, else 0. No protected attributes
    are used as features, anywhere in this package.
    """
    cfg = config or default_config()
    return Features(
        delinquencies=application.delinquencies,
        utilisation=application.utilisation,
        low_income=application.income < cfg.low_income_cutoff,
    )
