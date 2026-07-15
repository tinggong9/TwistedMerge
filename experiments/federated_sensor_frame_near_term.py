#!/usr/bin/env python3
"""N5: fresh real-data federated frame discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compact_federated_frame as federated
from experiments.future_benchmark_common import OUT, patch_compact_paths, stage_result

DEST = OUT / "near_term"


def main() -> None:
    patch_compact_paths(federated, DEST); federated.main()
    claims = json.loads((DEST / "federated_claims.json").read_text())
    gate = bool(claims.get("persistent_lift_gain_found"))
    stage_result("N5", "confirmation" if gate else "negative", f"federated frame gate {'passed' if gate else 'did not pass'}", gate_passed=gate, positive_regimes=claims.get("positive_regimes", []))


if __name__ == "__main__":
    main()
