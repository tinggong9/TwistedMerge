#!/usr/bin/env python3
"""F2: public multiview microscopy task availability gate."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import OUT, factual_report, record_command, stage_complete, update_status, utc_now, write_csv  # noqa: E402

SCRIPT = Path(__file__).resolve()
DEST = OUT / "microscopy"
COMMAND = "python experiments/microscopy_multiview_retransport.py"


def run(smoke: bool = False) -> dict[str, object]:
    reason = "no public multiview microscopy archive with view metadata and segmentation annotations was resolved and audited"
    write_csv(DEST / "runs.csv", [], ("seed", "view_condition", "method", "dice", "center_error", "orientation_error"))
    factual_report(DEST / "report.md", "Microscopy multiview retransport", ["State: blocked.", f"Reason: {reason}.", "ModelNet and synthetic shapes were not substituted."])
    update_status("F2_microscopy_multiview", "blocked", reason)
    stage_complete(DEST / "runs.csv", {"stage": "F2", "state": "blocked"})
    return {"state": "blocked", "rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    result = run(args.smoke)
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="availability gate", dataset_revision="no_resolved_multiview_microscopy_archive", started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary="no microscopy model rows executed")


if __name__ == "__main__":
    main()
