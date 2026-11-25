"""Input validation for single applications and CSV rows.

Validation is deliberately strict and explicit: a bad value is rejected
with a named error instead of being coerced, clamped, or silently
ignored. The scoring model therefore only ever sees values inside the
ranges documented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ModelConfig, default_config
from .exceptions import RowValidationError, ValidationError

#: Columns every applications CSV must carry, in this order.
REQUIRED_COLUMNS = ("income", "delinquencies", "utilisation", "fraud")


@dataclass(frozen=True)
class ApplicationInput:
    """One validated application, ready for feature building."""

    income: int
    delinquencies: int
    utilisation: float


@dataclass(frozen=True)
class ApplicationRecord(ApplicationInput):
    """A validated row from the applications CSV, including its label."""

    fraud: int


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_application(
    income: object,
    delinquencies: object,
    utilisation: object,
    config: ModelConfig | None = None,
) -> ApplicationInput:
    """Validate raw application fields and return a typed record.

    Args:
        income: Annual income in whole currency units. Must be a
            non-negative number no larger than ``config.max_income``.
        delinquencies: Count of past delinquencies. Must be a whole
            number between 0 and ``config.max_delinquencies``.
        utilisation: Credit utilisation ratio, 0.0 to 1.0 inclusive.
        config: Settings to validate against; defaults are used when
            omitted.

    Raises:
        ValidationError: With a message naming the offending field.
    """
    cfg = config or default_config()

    if not _is_number(income) or isinstance(income, float) and not income.is_integer():
        raise ValidationError(f"income must be a whole number, got {income!r}")
    if math.isnan(float(income)) or math.isinf(float(income)):
        raise ValidationError(f"income must be finite, got {income!r}")
    income_int = int(income)
    if income_int < 0:
        raise ValidationError(f"income must be >= 0, got {income_int}")
    if income_int > cfg.max_income:
        raise ValidationError(
            f"income must be <= {cfg.max_income}, got {income_int}"
        )

    if not _is_number(delinquencies) or isinstance(delinquencies, float) and not delinquencies.is_integer():
        raise ValidationError(
            f"delinquencies must be a whole number, got {delinquencies!r}"
        )
    delinquencies_int = int(delinquencies)
    if delinquencies_int < 0:
        raise ValidationError(
            f"delinquencies must be >= 0, got {delinquencies_int}"
        )
    if delinquencies_int > cfg.max_delinquencies:
        raise ValidationError(
            f"delinquencies must be <= {cfg.max_delinquencies}, "
            f"got {delinquencies_int}"
        )

    if not _is_number(utilisation):
        raise ValidationError(
            f"utilisation must be a number, got {utilisation!r}"
        )
    utilisation_float = float(utilisation)
    if math.isnan(utilisation_float) or math.isinf(utilisation_float):
        raise ValidationError("utilisation must be finite")
    if not 0.0 <= utilisation_float <= 1.0:
        raise ValidationError(
            f"utilisation must be between 0.0 and 1.0, got {utilisation_float}"
        )

    return ApplicationInput(
        income=income_int,
        delinquencies=delinquencies_int,
        utilisation=utilisation_float,
    )


def _parse_int(raw: str | None, field: str, line_no: int) -> int:
    if raw is None or raw.strip() == "":
        raise RowValidationError(line_no, f"missing value for {field!r}")
    try:
        return int(raw)
    except ValueError:
        raise RowValidationError(
            line_no, f"{field!r} is not a whole number: {raw!r}"
        ) from None


def _parse_float(raw: str | None, field: str, line_no: int) -> float:
    if raw is None or raw.strip() == "":
        raise RowValidationError(line_no, f"missing value for {field!r}")
    try:
        return float(raw)
    except ValueError:
        raise RowValidationError(
            line_no, f"{field!r} is not a number: {raw!r}"
        ) from None


def parse_application_row(
    row: dict[str, str | None],
    line_no: int,
    config: ModelConfig | None = None,
) -> ApplicationRecord:
    """Validate one CSV row (as returned by :class:`csv.DictReader`).

    Args:
        row: Raw string values keyed by column name.
        line_no: 1-based line number in the source file, used in errors
            and in the quarantine file so a bad row can be traced back.
        config: Settings to validate against; defaults when omitted.

    Raises:
        RowValidationError: If any field is missing, unparseable, or
            out of range, or if the label is not 0/1.
    """
    cfg = config or default_config()
    income = _parse_int(row.get("income"), "income", line_no)
    delinquencies = _parse_int(row.get("delinquencies"), "delinquencies", line_no)
    utilisation = _parse_float(row.get("utilisation"), "utilisation", line_no)
    fraud = _parse_int(row.get("fraud"), "fraud", line_no)
    if fraud not in (0, 1):
        raise RowValidationError(
            line_no, f"'fraud' must be 0 or 1, got {fraud}"
        )
    try:
        validated = validate_application(income, delinquencies, utilisation, cfg)
    except ValidationError as exc:
        raise RowValidationError(line_no, str(exc)) from exc
    return ApplicationRecord(
        income=validated.income,
        delinquencies=validated.delinquencies,
        utilisation=validated.utilisation,
        fraud=fraud,
    )
