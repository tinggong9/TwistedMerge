#!/usr/bin/env python
"""CIFAR model-merging entry point.

This scaffold exists so reports can distinguish unsupported image-task claims
from completed synthetic obstruction experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:
        save_json(
            args.reports_dir / "configs" / "cifar_model_merging_status.json",
            {
                "status": "unsupported",
                "reason": "PyTorch/torchvision are not installed in this environment.",
                "error": str(exc),
                "environment": capture_environment(),
            },
        )
        raise SystemExit(
            "CIFAR experiment not run: install PyTorch/torchvision with `python -m pip install -r requirements.txt`."
        )
    save_json(
        args.reports_dir / "configs" / "cifar_model_merging_status.json",
        {
            "status": "scaffolded",
            "reason": "Dependency check passed, but image-task training/merging is not implemented yet.",
            "environment": capture_environment(),
        },
    )
    if not args.check_only:
        raise SystemExit("CIFAR experiment scaffold only; no image-task claim generated.")
    print("CIFAR dependency check passed; implementation is scaffolded.")


if __name__ == "__main__":
    main()
