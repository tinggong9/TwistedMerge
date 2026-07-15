#!/usr/bin/env python3
"""E6: public numerical evidence packet assembled only from completed stages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_benchmark_common import OUT, stage_result, write_json

DEST = OUT / "emergency"


def main() -> None:
    required = ["e0_artifact_status.csv", "level2_claims.json", "calibration_summary.csv", "mechanism_claims.json", "central_summary.csv", "period_index_summary.csv", "practical_selector_paired.csv"]
    missing = [name for name in required if not (DEST / name).exists()]
    if missing:
        stage_result("E6", "failed", f"missing prerequisite evidence: {missing}")
        raise SystemExit(1)
    level2 = json.loads((DEST / "level2_claims.json").read_text())
    mechanism = json.loads((DEST / "mechanism_claims.json").read_text())
    practical = pd.read_csv(DEST / "practical_selector_paired.csv")
    central = pd.read_csv(DEST / "central_summary.csv")
    period = pd.read_csv(DEST / "period_index_summary.csv")
    calibration = pd.read_csv(DEST / "calibration_summary.csv")
    packet = {
        "controlled_context_confirmation": level2,
        "mechanistic_attribution": mechanism,
        "practical_selector_mean_delta": float(practical.selector_minus_greedy.mean()),
        "controlled_central_rows": int(len(central)),
        "representation_rank_cases": int(len(period)),
        "calibration_rows": int(len(calibration)),
        "strongest_supported_evidence_level": 2 if level2.get("independent_gate_passed") else 1,
        "broad_natural_checkpoint_superiority": False,
    }
    write_json(DEST / "emergency_evidence_packet.json", packet)
    text = [
        "# Emergency evidence packet",
        "",
        "This packet contains only numerical evidence and mechanical gates.",
        "",
        f"- Independent controlled confirmation: `{'passed' if level2.get('independent_gate_passed') else 'did not pass'}`.",
        f"- Strict mechanism-attribution criterion: `{'passed' if mechanism.get('full_gain_attributed_to_combination') else 'did not pass'}`.",
        f"- Practical selector minus ordinary greedy: `{packet['practical_selector_mean_delta']:+.6f}`.",
        f"- Strongest mechanically supported evidence level: `{packet['strongest_supported_evidence_level']}`.",
        "- Broad natural-checkpoint superiority remains unsupported.",
        "",
    ]
    (DEST / "emergency_evidence_packet.md").write_text("\n".join(text), encoding="utf-8")
    stage_result("E6", "completed", "assembled public emergency evidence packet", strongest_supported_evidence_level=packet["strongest_supported_evidence_level"])


if __name__ == "__main__":
    main()
