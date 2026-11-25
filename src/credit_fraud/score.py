"""CLI: score a single application and print the result as JSON.

Usage:
    python -m credit_fraud.score --income 72000 --delinquencies 0 --utilisation 0.31

The result is the only thing printed to stdout, so it can be piped to
other tools. Diagnostics go to stderr via logging. Invalid inputs exit
with code 2 and an error message; nothing is scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config
from .exceptions import FraudRiskError
from .logging_config import setup_logging
from .model import FraudRiskModel

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one credit application for fraud risk."
    )
    parser.add_argument("--income", type=int, required=True,
                        help="annual income, whole currency units, >= 0")
    parser.add_argument("--delinquencies", type=int, required=True,
                        help="count of past delinquencies, >= 0")
    parser.add_argument("--utilisation", type=float, required=True,
                        help="credit utilisation ratio, 0.0 to 1.0")
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
        result = FraudRiskModel(config).score_raw(
            args.income, args.delinquencies, args.utilisation
        )
    except FraudRiskError as exc:
        logger.error("%s", exc)
        return 2
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
