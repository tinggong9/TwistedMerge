#!/usr/bin/env python3
"""D2: gated residual-aware correction; never runs without a D1 certificate."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import (  # noqa: E402
    OUT,
    dataset_checksum,
    dataset_ready,
    factual_report,
    record_command,
    stage_complete,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "transitions"
COMMAND = "python experiments/residual_aware_segmentation.py"


def _certificate() -> bool:
    path = DEST / "residuals.csv"
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [row for row in rows if row.get("layer") == "certificate"]
    return bool(values and values[0].get("certified_stable_residual", "").lower() == "true")


def run(smoke: bool = False) -> dict[str, Any]:
    if not _certificate():
        write_csv(DEST / "correction_runs.csv", [], ("seed", "method", "dice", "boundary_dice", "residual"))
        write_csv(DEST / "correction_paired.csv", [], ("comparison", "mean_delta", "ci_lower", "ci_upper"))
        claims = [{"claim": "d1_stable_residual_required", "passed": False, "reason": "D1 did not certify a stable residual"}, {"claim": "residual_correction_activated", "passed": False, "reason": "gate closed before correction fitting or test evaluation"}]
        write_csv(DEST / "correction_claims.csv", claims)
        factual_report(DEST / "correction_report.md", "Residual-aware segmentation correction", ["State: gate_closed.", "D1 did not certify a stable residual against every matched null with stable rank, closure, and centrality.", "No residual correction was fitted or activated."])
        update_status("D2_residual_correction", "gate_closed", "D1 stable-residual certificate was false")
        stage_complete(DEST / "correction_claims.csv", {"stage": "D2", "state": "gate_closed", "correction_activated": False})
        return {"state": "gate_closed", "rows": 0}
    # This branch is intentionally blocked until a separately reviewed correction
    # family is preregistered; D1 currently does not open it.
    update_status("D2_residual_correction", "blocked", "D1 passed but correction family is not preregistered")
    return {"state": "blocked", "rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    result = run(args.smoke)
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="gated by D1", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"correction rows={result['rows']}")


if __name__ == "__main__":
    main()
