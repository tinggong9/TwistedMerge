#!/usr/bin/env python3
"""N4: fresh compact pretrained-vision discovery with gated confirmation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compact_pretrained_vision as vision
from experiments.future_benchmark_common import OUT, patch_compact_paths, stage_result, write_csv

DEST = OUT / "near_term"


def main() -> None:
    patch_compact_paths(vision, DEST); vision.main()
    claims = json.loads((DEST / "vision_claims.json").read_text())
    baselines = [
        {"baseline": "Git Re-Basin", "implementation": "faithful internal permutation-compatible approximation", "confirmation_required": bool(claims.get("discovery_gate_passed"))},
        {"baseline": "cycle-consistent model merging", "implementation": "faithful internal alignment-compatible approximation", "confirmation_required": bool(claims.get("discovery_gate_passed"))},
        {"baseline": "Task Arithmetic", "implementation": "internal delta merge", "confirmation_required": False},
        {"baseline": "TIES", "implementation": "internal sign-election merge", "confirmation_required": False},
        {"baseline": "DARE", "implementation": "internal random delta rescaling", "confirmation_required": False},
    ]
    write_csv(DEST / "vision_baselines.csv", baselines)
    state = "blocked" if claims.get("resource_blocked") else ("confirmation" if claims.get("discovery_gate_passed") else "negative")
    stage_result("N4", state, "pretrained ResNet-18 discovery executed" if not claims.get("resource_blocked") else "pretrained vision blocked", gate_passed=bool(claims.get("discovery_gate_passed")), confirmation_executed=bool(claims.get("confirmation_executed")))


if __name__ == "__main__":
    main()
