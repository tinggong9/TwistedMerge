#!/usr/bin/env python3
"""E1: gate for a second, different biomedical segmentation dataset."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import OUT, dataset_checksum, dataset_ready, factual_report, record_command, stage_complete, update_status, utc_now, write_csv  # noqa: E402

SCRIPT = Path(__file__).resolve()
DEST = OUT / "confirmation"
COMMAND = "python experiments/second_biomedical_segmentation.py"


def b1_gate() -> bool:
    path = OUT / "biomedical" / "discovery" / "claims.csv"
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        claims = {row["claim"]: row["passed"].lower() == "true" for row in csv.DictReader(handle)}
    return bool(claims.get("retransport_gate") or claims.get("twistedmerge_specific_gate"))


def run(smoke: bool = False) -> dict[str, object]:
    gate = b1_gate()
    state = "blocked" if gate else "gate_closed"
    reason = "no second public dataset from a different biomedical task type was resolved and audited" if gate else "B1 retransport and TwistedMerge-specific gates were false"
    write_csv(DEST / "runs.csv", [], ("seed", "method", "dice", "boundary_dice"))
    write_csv(DEST / "summary.csv", [], ("method", "dice", "boundary_dice"))
    write_csv(DEST / "paired.csv", [], ("comparison", "mean_delta", "ci_lower", "ci_upper"))
    write_csv(DEST / "claims.csv", [{"claim": "second_biomedical_dataset_gate", "passed": False, "state": state, "reason": reason}])
    factual_report(DEST / "report.md", "Second biomedical segmentation dataset", [f"State: {state}.", f"Reason: {reason}.", "No synthetic or same-task substitute was used."])
    update_status("E1_second_biomedical_dataset", state, reason)
    stage_complete(DEST / "claims.csv", {"stage": "E1", "state": state, "gate_input": gate})
    return {"state": state, "rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    result = run(args.smoke)
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="gated discovery and confirmation", dataset_revision=dataset_checksum() if dataset_ready() else "unavailable", started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary="no second-dataset model rows executed")


if __name__ == "__main__":
    main()
