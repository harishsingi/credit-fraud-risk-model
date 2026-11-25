"""CLI: generate a synthetic applications dataset.

Usage:
    python -m credit_fraud.generate_data [--rows N] [--seed S] [--out PATH]

Writes ``applications.csv`` (default location from config). Deterministic
for a given seed, and safe to re-run: it overwrites the same file, so
re-running with the same arguments is idempotent.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .data import generate_applications, write_applications_csv
from .exceptions import FraudRiskError
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic credit applications dataset."
    )
    parser.add_argument("--rows", type=int, default=None,
                        help="number of rows to generate (default: from config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed; same seed gives the same file (default: from config)")
    parser.add_argument("--out", type=str, default=None,
                        help="output CSV path (default: <project>/data/applications.csv)")
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
        n_rows = args.rows if args.rows is not None else config.synthetic_rows
        seed = args.seed if args.seed is not None else config.synthetic_seed
        out_path = args.out if args.out is not None else config.data_path
        rows = generate_applications(n_rows, seed, config)
        write_applications_csv(out_path, rows)
        logger.info("generated %d synthetic rows (seed=%d) -> %s", n_rows, seed, out_path)
    except (FraudRiskError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
