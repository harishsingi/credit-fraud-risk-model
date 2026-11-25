"""Make the ``credit_fraud`` package importable for pytest without install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
