#!/usr/bin/env python3
"""E2: real landmark/heatmap task gate after exact output-action tests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import OUT, factual_report, record_command, stage_complete, update_status, utc_now, write_csv  # noqa: E402

SCRIPT = Path(__file__).resolve()
DEST = OUT / "landmarks"
COMMAND = "python experiments/biomedical_landmark_retransport.py"


def sanity_passed() -> bool:
    path = OUT / "sanity" / ".complete.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("state") == "completed"


def run(smoke: bool = False) -> dict[str, object]:
    sanity = sanity_passed()
    state = "blocked" if sanity else "gate_closed"
    reason = "selected biomedical dataset has no independent landmark, keypoint, or center annotations" if sanity else "exact mask sanity stage did not pass"
    write_csv(DEST / "runs.csv", [], ("seed", "method", "mean_landmark_error", "pck", "heatmap_iou", "calibration", "equivariance_error"))
    factual_report(DEST / "report.md", "Biomedical landmark retransport", [f"State: {state}.", f"Reason: {reason}.", "No landmarks or heatmaps were derived from test masks."])
    update_status("E2_biomedical_landmarks", state, reason)
    stage_complete(DEST / "runs.csv", {"stage": "E2", "state": state, "sanity_passed": sanity})
    return {"state": state, "rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    result = run(args.smoke)
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="gated real annotations", dataset_revision="no_resolved_landmark_archive", started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary="no real-landmark model rows executed")


if __name__ == "__main__":
    main()
