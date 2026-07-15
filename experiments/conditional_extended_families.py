#!/usr/bin/env python3
"""C1/C2: enforce prerequisite gates for extended vision and adapter families."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import OUT, git_head, write_csv

ICLR = OUT / "iclr"
DEST = OUT / "extended"


def claim(path: Path, name: str) -> bool:
    if not path.exists(): return False
    with path.open(encoding="utf-8", newline="") as handle:
        return any(row.get("claim") == name and row.get("value", "").lower() == "true" for row in csv.DictReader(handle))


def main() -> None:
    vision = claim(ICLR / "full_model_claims.csv", "complete_realistic_gate_passed")
    adapter_mechanism = claim(ICLR / "natural_claims.csv", "complete_new_family_gate_passed")
    rows = [
        {"stage": "C1", "family": "broader_pretrained_vision", "prerequisite": "B1 useful transition geometry", "prerequisite_passed": vision, "status": "requires_execution" if vision else "gated_off", "reason": "B1 gate passed" if vision else "B1 did not produce useful transition geometry"},
        {"stage": "C2", "family": "real_adapter_basis_holonomy", "prerequisite": "nontrivially_closing_adapter_transition_mechanism", "prerequisite_passed": adapter_mechanism, "status": "requires_execution" if adapter_mechanism else "gated_off", "reason": "new mechanism certified" if adapter_mechanism else "no nontrivially closing adapter transition mechanism was certified"},
    ]
    write_csv(DEST / "conditional_family_status.csv", rows)
    if vision or adapter_mechanism:
        raise RuntimeError("an extended conditional prerequisite passed; the corresponding expanded execution must complete")
    (DEST / "conditional_family_report.md").write_text(
        "# Conditional extended families\n\n"
        f"Execution commit: `{git_head()}`. C1 was gated off because B1 did not pass its complete transition-geometry "
        "gate. C2 was gated off because no new nontrivially closing adapter-basis mechanism was certified. No frozen-head "
        "vision or already-closing adapter cycle was rerun.\n",
        encoding="utf-8",
    )


if __name__ == "__main__": main()
