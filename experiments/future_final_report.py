#!/usr/bin/env python3
"""F: final public global benchmark report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_benchmark_common import OUT, stage_result


def main() -> None:
    status = json.loads((OUT / "status.json").read_text()) if (OUT / "status.json").exists() else {"stages": {}}
    ladder = json.loads((OUT / "near_term" / "claim_ladder.json").read_text())
    commands = pd.read_csv(OUT / "commands.csv") if (OUT / "commands.csv").exists() else pd.DataFrame()
    failures = pd.read_csv(OUT / "failures.csv") if (OUT / "failures.csv").exists() else pd.DataFrame()
    lines = [
        "# Final global benchmark evidence report",
        "",
        "This report summarizes numerical evidence and execution state only.",
        "",
        f"Strongest mechanically supported evidence level: `{ladder['strongest_supported_level']}`.",
        "",
        "## Stage decisions",
        "",
    ]
    for stage_id, item in status.get("stages", {}).items(): lines.append(f"- {stage_id}: `{item.get('state')}` — {item.get('summary', '')}")
    lines.extend(["", "## Family gates", ""])
    for name, value in ladder["family_gates"].items(): lines.append(f"- {name}: `{'passed' if value else 'not passed'}`")
    lines.extend(["", "## Safe numerical interpretation", "", "- Controlled central-sign and finite representation-rank results remain supported within their exact constructed systems.", "- Independent noncommutative context results are reported against matched generic context baselines.", "- Approximate natural cycle residuals are not called cohomological classes without centrality, closure, and distance-to-coboundary certificates.", "- Supplied-context, learned-router, real-image, pretrained, federated, adapter, transformer, and pose results remain separated.", "- All failed gates, download blockers, zero lift activations, and negative paired results are retained.", "", "## Reproduction", "", "Exact commands, execution commits, source hashes, runtimes, exit codes, and failures are recorded in the adjacent command and failure ledgers. Artifact checksums are recorded in the extended manifest.", ""])
    (OUT / "final_global_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    stage_result("F", "completed", f"global numerical report completed; strongest evidence level={ladder['strongest_supported_level']}", strongest_supported_level=ladder["strongest_supported_level"])


if __name__ == "__main__":
    main()
