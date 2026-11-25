"""CLI: load, validate, and evaluate the model on a dataset.

Usage:
    python -m credit_fraud.train [--data PATH] [--out PATH]

Steps: load the CSV (bad rows quarantined, counts logged), score every
valid row, and write a metrics JSON with the confusion matrix, precision,
recall, accuracy, F1, and calibration buckets. Re-running overwrites the
same metrics file, so it is idempotent. All metrics are computed at
runtime from the loaded data; none are hardcoded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import MODEL_VERSION, load_config
from .data import load_applications, write_quarantine_csv
from .evaluate import evaluate
from .exceptions import FraudRiskError
from .logging_config import setup_logging
from .model import FraudRiskModel

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the fraud risk model on a dataset."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="applications CSV (default: <project>/data/applications.csv)")
    parser.add_argument("--out", type=str, default=None,
                        help="metrics JSON output (default: <project>/reports/metrics.json)")
    parser.add_argument("--quarantine-out", type=str, default=None,
                        help="quarantine CSV output (default: <project>/data/quarantine.csv)")
    parser.add_argument("--config", type=str, default=None,
                        help="optional JSON config file overriding defaults")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="logging level (default: INFO)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        config = load_config(args.config)
        data_path = Path(args.data) if args.data is not None else config.data_path
        out_path = Path(args.out) if args.out is not None else config.metrics_path
        quarantine_path = (
            Path(args.quarantine_out)
            if args.quarantine_out is not None
            else config.quarantine_path
        )

        loaded = load_applications(data_path, config)
        if loaded.quarantined:
            write_quarantine_csv(quarantine_path, loaded.quarantined)
            logger.warning(
                "%d row(s) quarantined; see %s", loaded.rows_quarantined, quarantine_path
            )

        report = evaluate(FraudRiskModel(config), loaded.rows, config)
        payload = {
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_path": str(data_path),
            "rows_read": loaded.rows_read,
            "rows_valid": loaded.rows_valid,
            "rows_quarantined": loaded.rows_quarantined,
            "config": config.to_dict(),
            **report.to_dict(),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("metrics written to %s", out_path)
        logger.info(
            "n=%d precision=%.3f recall=%.3f accuracy=%.3f f1=%.3f",
            report.n_rows, report.precision, report.recall,
            report.accuracy, report.f1_score,
        )
    except FraudRiskError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
