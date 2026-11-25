"""Synthetic data generation and loading with data-quality handling.

Two responsibilities, kept separate from the model:

1. **Generation** — build a fully synthetic applications dataset with a
   seeded RNG (deterministic: the same seed always yields the same file).
   The generator's risk formula is documented inline; it is a teaching
   device, not a fitted model of real fraud.
2. **Loading** — read a CSV, validate every row, and *quarantine* bad
   rows (missing/unparseable/out-of-range values, duplicates) instead of
   silently dropping them or crashing. Every run logs rows read, valid,
   and quarantined so the counts are auditable.
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import ModelConfig, default_config
from .exceptions import DataQualityError, RowValidationError
from .validation import REQUIRED_COLUMNS, ApplicationRecord, parse_application_row

logger = logging.getLogger(__name__)

# Synthetic generator parameters. These describe the *fake* data only;
# they are not estimates from any real population.
_GENERATOR_INCOME_MIN = 25_000
_GENERATOR_INCOME_MAX = 180_000
_GENERATOR_DELINQUENCY_CHOICES = (0, 0, 0, 1, 2, 3)
_GENERATOR_FRAUD_SCALE = 0.35


@dataclass(frozen=True)
class QuarantinedRow:
    """One row that failed validation, kept for inspection."""

    line_no: int
    reason: str
    raw: dict[str, str | None] = field(default_factory=dict)


@dataclass
class LoadResult:
    """Outcome of loading an applications CSV."""

    rows: list[ApplicationRecord]
    quarantined: list[QuarantinedRow]
    rows_read: int

    @property
    def rows_valid(self) -> int:
        return len(self.rows)

    @property
    def rows_quarantined(self) -> int:
        return len(self.quarantined)


def generate_applications(
    n_rows: int,
    seed: int,
    config: ModelConfig | None = None,
) -> list[ApplicationRecord]:
    """Generate ``n_rows`` synthetic applications with a seeded RNG.

    Deterministic for a given ``seed``: the RNG is local to this call,
    so generating data never disturbs global random state elsewhere.

    The label is drawn from a simple hand-written risk formula
    (delinquencies and utilisation raise risk; low income adds a bit),
    purely so the demo dataset has signal for the model to score. The
    formula is not fitted to, or representative of, any real data.

    Raises:
        ValueError: If ``n_rows`` is not positive.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    cfg = config or default_config()
    rng = random.Random(seed)
    rows: list[ApplicationRecord] = []
    for _ in range(n_rows):
        income = rng.randint(_GENERATOR_INCOME_MIN, _GENERATOR_INCOME_MAX)
        delinquencies = rng.choice(_GENERATOR_DELINQUENCY_CHOICES)
        utilisation = round(rng.random(), 2)
        risk = (
            delinquencies * 0.35
            + utilisation * 0.5
            + (0.2 if income < cfg.low_income_cutoff else 0)
        )
        fraud = 1 if rng.random() < risk * _GENERATOR_FRAUD_SCALE else 0
        rows.append(
            ApplicationRecord(
                income=income,
                delinquencies=delinquencies,
                utilisation=utilisation,
                fraud=fraud,
            )
        )
    return rows


def write_applications_csv(path: str | Path, rows: list[ApplicationRecord]) -> Path:
    """Write validated rows to CSV. Idempotent: overwrites ``path``.

    Creates parent directories as needed. Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(REQUIRED_COLUMNS)
        for row in rows:
            writer.writerow(
                [row.income, row.delinquencies, row.utilisation, row.fraud]
            )
    logger.info("wrote %d rows to %s", len(rows), path)
    return path


def write_quarantine_csv(path: str | Path, quarantined: list[QuarantinedRow]) -> Path:
    """Write quarantined rows (with line number + reason) for review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_no", "reason", "raw"])
        for item in quarantined:
            writer.writerow([item.line_no, item.reason, dict(item.raw)])
    logger.info("wrote %d quarantined rows to %s", len(quarantined), path)
    return path


def load_applications(
    path: str | Path,
    config: ModelConfig | None = None,
) -> LoadResult:
    """Load and validate an applications CSV, quarantining bad rows.

    Validation rules (see :mod:`credit_fraud.validation`):

    - header must contain every required column;
    - each row must parse and fall inside the documented ranges;
    - exact duplicate rows are quarantined (first occurrence kept).

    Args:
        path: CSV file to read.
        config: Settings to validate against; defaults when omitted.

    Returns:
        A :class:`LoadResult` with valid rows, quarantined rows, and
        counts. The caller decides what to do with quarantined rows;
        the counts are always logged.

    Raises:
        DataQualityError: If the file cannot be read, the header is
            missing required columns, or no valid rows remain.
    """
    cfg = config or default_config()
    path = Path(path)
    if not path.is_file():
        raise DataQualityError(f"applications file not found: {path}")
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise DataQualityError(f"could not read {path}: {exc}") from exc

    valid: list[ApplicationRecord] = []
    quarantined: list[QuarantinedRow] = []
    seen: set[tuple[int, int, float, int]] = set()
    rows_read = 0
    with handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DataQualityError(
                f"{path} header is missing required column(s): "
                f"{', '.join(missing)}"
            )
        for line_no, raw_row in enumerate(reader, start=2):
            rows_read += 1
            try:
                record = parse_application_row(raw_row, line_no, cfg)
            except RowValidationError as exc:
                quarantined.append(
                    QuarantinedRow(line_no=exc.line_no, reason=exc.reason, raw=dict(raw_row))
                )
                continue
            key = (record.income, record.delinquencies, record.utilisation, record.fraud)
            if key in seen:
                quarantined.append(
                    QuarantinedRow(
                        line_no=line_no,
                        reason="duplicate of an earlier row",
                        raw=dict(raw_row),
                    )
                )
                continue
            seen.add(key)
            valid.append(record)

    if not valid:
        raise DataQualityError(
            f"{path}: no valid rows after quarantine "
            f"({rows_read} read, {len(quarantined)} quarantined)"
        )
    logger.info(
        "loaded %s: %d rows read, %d valid, %d quarantined",
        path,
        rows_read,
        len(valid),
        len(quarantined),
    )
    return LoadResult(rows=valid, quarantined=quarantined, rows_read=rows_read)
