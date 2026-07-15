#!/usr/bin/env python3
"""N10: mechanical evidence-level decision."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_benchmark_common import OUT, stage_result, write_json

DEST = OUT / "near_term"


def read_claim(path: Path, key: str, default=False):
    if path.suffix == ".json" and path.exists(): return json.loads(path.read_text()).get(key, default)
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("claim") == key: return str(row.get("value", "")).lower() == "true"
    return default


def main() -> None:
    level1 = (OUT / "emergency" / "central_summary.csv").exists() and (OUT / "emergency" / "period_index_summary.csv").exists()
    level2 = bool(read_claim(OUT / "emergency" / "level2_claims.json", "independent_gate_passed"))
    families = {
        "natural_checkpoints": bool(read_claim(DEST / "realistic_claims.csv", "full_gate_passed")),
        "real_image_charts": bool(read_claim(DEST / "image_chart_claims.csv", "bridge_gate_passed")),
        "pretrained_vision": bool(read_claim(DEST / "vision_claims.json", "discovery_gate_passed")),
        "federated_frames": bool(read_claim(DEST / "federated_claims.json", "persistent_lift_gain_found")),
        "real_adapters": bool(read_claim(DEST / "lora_claims.csv", "full_gate_passed")),
        "transformers": bool(read_claim(DEST / "transformer_claims.csv", "gate_passed")),
        "projective_pose": bool(read_claim(DEST / "pose_claims.csv", "pose_lift_gate_passed")),
    }
    level3 = any(families.values()); level4 = sum(families.values()) >= 3 and families["pretrained_vision"] and (families["real_adapters"] or families["transformers"])
    strongest = 4 if level4 else 3 if level3 else 2 if level2 else 1 if level1 else 0
    ladder = {"level_1_controlled": level1, "level_2_structured_context": level2, "level_3_single_realistic_family": level3, "level_4_broad_practical": level4, "family_gates": families, "strongest_supported_level": strongest}
    write_json(DEST / "claim_ladder.json", ladder)
    lines = ["# Mechanical evidence ladder", "", "This artifact contains numerical claim gates only.", "", *[f"- Level {index}: `{'passed' if ladder[key] else 'not passed'}`" for index, key in enumerate(["level_1_controlled", "level_2_structured_context", "level_3_single_realistic_family", "level_4_broad_practical"], start=1)], "", f"Strongest supported level: `{strongest}`.", ""]
    (DEST / "claim_ladder.md").write_text("\n".join(lines), encoding="utf-8")
    (DEST / "final_near_term_report.md").write_text("\n".join(lines + ["Family decisions:", "", *[f"- {name}: `{'passed' if value else 'not passed'}`" for name, value in families.items()], ""]), encoding="utf-8")
    stage_result("N10", "completed", f"mechanical evidence ladder completed; strongest level={strongest}", strongest_supported_level=strongest)


if __name__ == "__main__":
    main()
