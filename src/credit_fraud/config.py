"""Configuration for the credit fraud risk model.

All tunable values live here, separated from the scoring logic, so a
reviewer can see every weight, threshold, and path in one place. Nothing
in :mod:`credit_fraud.model` or the CLIs hardcodes a number that belongs
here.

Weights and thresholds are illustrative values for this synthetic demo.
They were chosen by hand for readability, not fitted to real data, and no
production threshold is reproduced here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_type_hints

from .exceptions import ConfigurationError

MODEL_VERSION = "0.2.0"

# Project root is the directory containing ``src/``; paths below resolve
# against it so the CLIs work from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    """All model, data, and output settings in one immutable object."""

    # --- scoring model (logistic-style, hand-picked illustrative weights) ---
    intercept: float = -2.2
    weight_delinquencies: float = 1.1
    weight_utilisation: float = 1.6
    weight_low_income: float = 0.5
    low_income_cutoff: int = 35_000
    review_threshold: float = 0.25

    # --- synthetic data generation ---
    synthetic_rows: int = 2_000
    synthetic_seed: int = 11

    # --- evaluation ---
    calibration_bins: int = 10

    # --- paths (relative to the project root unless absolute) ---
    data_dir: str = "data"
    applications_filename: str = "applications.csv"
    quarantine_filename: str = "quarantine.csv"
    metrics_filename: str = "reports/metrics.json"

    # --- validation bounds for a single application ---
    max_income: int = 10_000_000
    max_delinquencies: int = 50

    @property
    def data_path(self) -> Path:
        """Full path of the applications CSV."""
        return _resolve(self.data_dir) / self.applications_filename

    @property
    def quarantine_path(self) -> Path:
        """Full path of the quarantine CSV written during load."""
        return _resolve(self.data_dir) / self.quarantine_filename

    @property
    def metrics_path(self) -> Path:
        """Full path of the metrics JSON written by ``train``."""
        return _resolve(self.metrics_filename)

    def to_dict(self) -> dict:
        """Serialisable view, used when writing metrics files."""
        return asdict(self)


def _resolve(path: str) -> Path:
    """Resolve ``path`` against the project root unless it is absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def default_config() -> ModelConfig:
    """Defaults, with the data directory overridable via ``CF_DATA_DIR``."""
    data_dir = os.environ.get("CF_DATA_DIR", "data")
    return ModelConfig(data_dir=data_dir)


def load_config(path: str | Path | None = None) -> ModelConfig:
    """Load configuration, optionally overriding defaults from a JSON file.

    Args:
        path: Optional JSON file with a subset of :class:`ModelConfig`
            fields. Unknown keys and wrong-typed values are rejected
            loudly instead of being silently ignored.

    Raises:
        ConfigurationError: If the file is missing, not valid JSON, or
            contains unknown keys or values of the wrong type.
    """
    config = default_config()
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"config file not found: {config_path}")
    try:
        overrides = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"could not read config file {config_path}: {exc}"
        ) from exc
    if not isinstance(overrides, dict):
        raise ConfigurationError(
            f"config file {config_path} must contain a JSON object"
        )

    known = get_type_hints(ModelConfig)
    unknown = sorted(set(overrides) - set(known))
    if unknown:
        raise ConfigurationError(
            f"unknown config key(s) in {config_path}: {', '.join(unknown)}"
        )
    for key, value in overrides.items():
        expected = known[key]
        # bool is a subclass of int; reject it explicitly for int fields.
        if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigurationError(
                f"config key {key!r} must be an int, got {value!r}"
            )
        if expected is float and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ConfigurationError(
                f"config key {key!r} must be a number, got {value!r}"
            )
        if expected is str and not isinstance(value, str):
            raise ConfigurationError(
                f"config key {key!r} must be a string, got {value!r}"
            )
    return ModelConfig(**{**config.to_dict(), **overrides})
