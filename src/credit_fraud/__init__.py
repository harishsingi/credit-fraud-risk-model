"""Credit fraud risk model: an interpretable, synthetic-data scoring demo.

Public surface: the config, the model, the validation helpers, and the
data-loading/evaluation pieces. CLIs live in ``generate_data``, ``train``,
and ``score`` and are run as ``python -m credit_fraud.<module>``.
"""

from __future__ import annotations

from .config import MODEL_VERSION, ModelConfig, default_config, load_config
from .data import LoadResult, QuarantinedRow, generate_applications, load_applications
from .evaluate import CalibrationBucket, EvaluationReport, evaluate
from .exceptions import (
    ConfigurationError,
    DataQualityError,
    FraudRiskError,
    RowValidationError,
    ValidationError,
)
from .features import Features, build_features
from .model import FraudRiskModel, ScoreResult
from .validation import (
    ApplicationInput,
    ApplicationRecord,
    parse_application_row,
    validate_application,
)

__version__ = MODEL_VERSION

__all__ = [
    "MODEL_VERSION",
    "ModelConfig",
    "default_config",
    "load_config",
    "LoadResult",
    "QuarantinedRow",
    "generate_applications",
    "load_applications",
    "CalibrationBucket",
    "EvaluationReport",
    "evaluate",
    "ConfigurationError",
    "DataQualityError",
    "FraudRiskError",
    "RowValidationError",
    "ValidationError",
    "Features",
    "build_features",
    "FraudRiskModel",
    "ScoreResult",
    "ApplicationInput",
    "ApplicationRecord",
    "parse_application_row",
    "validate_application",
    "__version__",
]
