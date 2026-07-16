#!/usr/bin/env python3
"""F1: bounded 3D medical retransport gate."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import OUT, factual_report, record_command, stage_complete, update_status, utc_now, write_csv  # noqa: E402

SCRIPT = Path(__file__).resolve()
DEST = OUT / "extended_3d"
COMMAND = "python experiments/medical_3d_retransport.py"


def positive_2d_result() -> bool:
    path = OUT / "biomedical" / "discovery" / "claims.csv"
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        return any(row["claim"] == "retransport_gate" and row["passed"].lower() == "true" for row in csv.DictReader(handle))


def run(smoke: bool = False) -> dict[str, object]:
    gate = positive_2d_result()
    state = "blocked" if gate else "gate_closed"
    reason = "no bounded 3D MRI/CT archive with affine metadata was resolved and audited" if gate else "positive 2D retransport gate was not established"
    write_csv(DEST / "runs.csv", [], ("seed", "method", "volumetric_dice", "surface_dice", "hausdorff95", "orientation_consistency", "latency_ms", "memory_mb"))
    factual_report(DEST / "report.md", "Three-dimensional medical retransport", [f"State: {state}.", f"Reason: {reason}.", "No 2D or synthetic-shape substitute was used and no affine metadata was discarded."])
    update_status("F1_medical_3d", state, reason)
    stage_complete(DEST / "runs.csv", {"stage": "F1", "state": state, "positive_2d_gate": gate})
    return {"state": state, "rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    result = run(args.smoke)
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="gated extension", dataset_revision="no_resolved_3d_archive", started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary="no 3D model rows executed")


if __name__ == "__main__":
    main()
