#!/usr/bin/env python3
"""N8: attempt the real projective-pose benchmark without synthetic substitution."""

from __future__ import annotations

import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_benchmark_common import DATA, LOCAL, OUT, safe_path, stage_result, write_csv

DEST = OUT / "near_term"


def download_with_resume(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"} if existing else {})
    with urllib.request.urlopen(request, timeout=120) as source, path.open("ab" if existing else "wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def main() -> None:
    candidates = [DATA / "SYMSOL", DATA / "ModelNet10"]
    errors = []
    if not any(path.exists() for path in candidates):
        archive = LOCAL / "downloads" / "ModelNet10.zip"
        for attempt in [1, 2]:
            try:
                download_with_resume("https://modelnet.cs.princeton.edu/ModelNet10.zip", archive)
                break
            except Exception as error:
                errors.append({"attempt": attempt, "error_type": type(error).__name__, "error": safe_path(str(error))})
                time.sleep(1)
    available = next((path for path in candidates if path.exists()), None)
    write_csv(DEST / "pose_download_attempts.csv", errors, ["attempt", "error_type", "error"])
    write_csv(DEST / "pose_runs.csv", [], ["seed", "method", "mean_geodesic_error_degrees"])
    write_csv(DEST / "pose_residuals.csv", [], ["seed", "cycle_sign_residual"])
    write_csv(DEST / "pose_summary.csv", [], ["method", "mean_geodesic_error_degrees"])
    write_csv(DEST / "pose_claims.csv", [{"claim": "real_pose_dataset_available", "value": bool(available)}, {"claim": "real_pose_predictor_executed", "value": False}, {"claim": "download_errors", "value": str(errors)}])
    (DEST / "tables" / "pose.tex").write_text("% No real pose rows were completed.\n", encoding="utf-8")
    reason = "A dataset archive was acquired, but a target-independent pose model and licensed split are not present in this repository." if available else "The two resumable dataset-download attempts did not produce an installed licensed pose dataset."
    (DEST / "pose_report.md").write_text(f"# Real projective-pose benchmark\n\nBlocked: {reason} No generated-target or quaternion smoke was substituted for real-dataset evidence. Exact acquisition errors are retained.\n", encoding="utf-8")
    stage_result("N8", "blocked", reason, download_errors=errors, dataset_path=safe_path(available) if available else None)


if __name__ == "__main__":
    main()
