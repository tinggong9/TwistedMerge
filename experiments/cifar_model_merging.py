#!/usr/bin/env python
"""CIFAR-10 model-merging entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.model_merging_benchmark import main as benchmark_main  # noqa: E402


def main() -> None:
    sys.argv = [sys.argv[0], "--datasets", "cifar10", *sys.argv[1:]]
    benchmark_main()


if __name__ == "__main__":
    main()
