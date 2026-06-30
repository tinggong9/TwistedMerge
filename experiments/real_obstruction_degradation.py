#!/usr/bin/env python
"""Paper-facing entry point for real obstruction-degradation verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.model_merging_fixed_setting_verification import main  # noqa: E402


if __name__ == "__main__":
    main()
