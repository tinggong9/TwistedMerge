#!/usr/bin/env python3
"""E0: clean provenance, test, and historical-evidence classification."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from future_benchmark_common import OUT, ROOT, environment_manifest, safe_path, sha256_file, stage_result, write_csv, write_json

DEST = OUT / "emergency"
FORBIDDEN_GENERATORS = ["target_accuracy_for_method", "logits_with_target_accuracy", "prescribed method accuracies"]
REQUIRED_ARTIFACTS = [
    "reports/compact_program/context_runs.csv",
    "reports/compact_program/hodge_runs.csv",
    "reports/compact_program/natural_runs.csv",
    "reports/compact_program/vision_runs.csv",
    "reports/compact_program/federated_runs.csv",
    "reports/overnight_program/practical_selector_runs.csv",
    "reports/overnight_program/period_index_summary.csv",
]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = environment_manifest()
    write_json(DEST / "e0_environment.json", manifest)
    started = time.time()
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True, capture_output=True)
    test_text = tests.stdout + ("\n" + tests.stderr if tests.stderr else "")
    (DEST / "e0_tests.txt").write_text(test_text, encoding="utf-8")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    rows = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 30_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = [needle for needle in FORBIDDEN_GENERATORS if needle.lower() in text.lower()]
        if hits:
            rows.append({"artifact": relative, "status": "INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE", "matched_markers": ";".join(hits), "sha256": sha256_file(path)})
    for relative in REQUIRED_ARTIFACTS:
        path = ROOT / relative
        status = "VERIFIED_PRESENT" if path.exists() and path.stat().st_size else "MISSING"
        row_count = max(0, len(path.read_text(encoding="utf-8", errors="ignore").splitlines()) - 1) if path.exists() else 0
        rows.append({"artifact": relative, "status": status, "matched_markers": "", "sha256": sha256_file(path) if path.exists() else "", "row_count": row_count})
    write_csv(DEST / "e0_artifact_status.csv", rows)
    report = [
        "# Provenance and evidence freeze",
        "",
        f"Execution commit: `{manifest['head']}`",
        f"Branch: `{manifest['branch']}`",
        f"Worktree clean at process start: `{manifest['worktree_clean']}`",
        f"Test exit code: `{tests.returncode}`; runtime: `{time.time() - started:.2f}` seconds.",
        "",
        "Artifacts containing historical target-prescription helpers are classified as invalid empirical-accuracy evidence. Valid algebraic residual artifacts are retained. The artifact ledger contains exact hashes and presence checks for the current controlled, selector, natural-checkpoint, vision, and federated evidence.",
        "",
    ]
    (DEST / "e0_provenance.md").write_text("\n".join(report), encoding="utf-8")
    state = "clean-freeze" if tests.returncode == 0 else "failed"
    stage_result("E0", state, f"full tests exit={tests.returncode}; classified {sum(row['status'].startswith('INVALID') for row in rows)} invalid historical artifacts", tests_exit_code=tests.returncode, tests_runtime_seconds=time.time() - started)
    if tests.returncode:
        raise SystemExit(tests.returncode)


if __name__ == "__main__":
    main()
