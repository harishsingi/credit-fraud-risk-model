"""Logging setup shared by every module and CLI in this package.

The old scripts used ``print`` for everything, which mixes user-facing
output with diagnostics and gives no severity or source. All modules log
through the standard library; only the ``score`` CLI prints a JSON result
to stdout, and only the ``train`` CLI writes a metrics file.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once; safe to call repeatedly.

    Args:
        level: One of the standard logging level names (``DEBUG``,
            ``INFO``, ``WARNING``, ...). Invalid names fall back to INFO
            with a warning instead of crashing the CLI.
    """
    try:
        numeric_level = getattr(logging, level.upper())
    except AttributeError:
        logging.basicConfig(level=logging.INFO, format=_DEFAULT_FORMAT)
        logging.getLogger(__name__).warning(
            "unknown log level %r, falling back to INFO", level
        )
        return
    logging.basicConfig(level=numeric_level, format=_DEFAULT_FORMAT, stream=sys.stderr)
