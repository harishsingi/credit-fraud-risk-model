"""Shared exception types for the credit fraud risk model.

Every failure that a caller (CLI, test, or future service) might need to
distinguish gets its own type so error handling is explicit instead of
catching bare ``Exception`` or relying on ``print`` + exit codes.
"""

from __future__ import annotations


class FraudRiskError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(FraudRiskError):
    """Raised when a config file or environment override is invalid."""


class ValidationError(FraudRiskError):
    """Raised when a single application (or one of its fields) is invalid."""


class RowValidationError(ValidationError):
    """One row of an input CSV failed schema/range validation.

    Carries the source line number and a human-readable reason so the row
    can be quarantined instead of silently dropped or crashing the run.
    """

    def __init__(self, line_no: int, reason: str) -> None:
        super().__init__(f"line {line_no}: {reason}")
        self.line_no = line_no
        self.reason = reason


class DataQualityError(FraudRiskError):
    """Raised when an input dataset is unusable as a whole.

    Examples: file missing, header missing required columns, or zero valid
    rows after quarantine. Individual bad rows do not raise this; they are
    quarantined and counted (see :mod:`credit_fraud.data`).
    """
