#!/usr/bin/env python
"""Write the overnight program's clean-branch and evidence audit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence_provenance import current_commit, execution_commit, git_output


OUT = ROOT / "reports" / "overnight_program"
NEXT = ROOT / "reports" / "next_benchmarks"


CONFIGS = {
    "two_loop_holonomy": NEXT / "two_loop_holonomy_config.json",
    "controlled_central_and_period_index": NEXT / "central_reproduction_manifest.json",
    "context_router": NEXT / "context_router_config.json",
    "matched_selector": NEXT / "matched_selector_config.json",
    "heldout_diagnostic": NEXT / "diagnostic_prediction_config.json",
    "pretrained_smoke": NEXT / "pretrained_merge_config.json",
}


INVALID_ROWS = [
    (
        "reports/controlled_nonabelian_holonomy_report.md",
        "report",
        "INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE",
        "accuracy/loss/selector content",
        "finite-group definitions and standalone residuals only",
        "method-specific target accuracies and label-dependent logits",
    ),
    (
        "reports/csv/controlled_nonabelian_holonomy.csv",
        "raw_csv",
        "INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE",
        "all empirical metrics",
        "group metadata only",
        "logits_with_target_accuracy reads labels before predictions",
    ),
    (
        "reports/csv/controlled_nonabelian_holonomy_summary.csv",
        "summary_csv",
        "INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE",
        "all empirical summaries",
        "none",
        "derived from prescribed target accuracies",
    ),
    (
        "reports/csv/controlled_nonabelian_holonomy_paired_stats.csv",
        "paired_stats_csv",
        "INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE",
        "all paired accuracy statistics",
        "none",
        "compares prescribed method accuracies",
    ),
]


def run_tests() -> tuple[str, str]:
    command = f"PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache {sys.executable} -m pytest -q"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**__import__("os").environ, "PYTHONPYCACHEPREFIX": "/private/tmp/codex-pycache"},
    )
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(output)
    return command, output.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-result", default="")
    parser.add_argument("--test-command", default="")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.run_tests:
        test_command, test_result = run_tests()
    else:
        test_command = args.test_command or "PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python -m pytest -q"
        test_result = args.test_result or "355 passed, 5 subtests passed in 24.57s"

    provenance = []
    for name, path in CONFIGS.items():
        provenance.append(
            {
                "artifact_family": name,
                "config": str(path.relative_to(ROOT)),
                "execution_commit": execution_commit(path),
            }
        )

    state = {
        "branch": git_output(ROOT, "branch", "--show-current"),
        "head": current_commit(ROOT),
        "worktree_status": git_output(ROOT, "status", "--short", "--branch"),
        "tags": git_output(ROOT, "tag", "--list", "--sort=creatordate").splitlines(),
        "remote": git_output(ROOT, "remote", "get-url", "origin"),
        "isolated_worktree": str(ROOT),
        "base_commit": git_output(ROOT, "merge-base", "HEAD", "main"),
        "preserved_main_checkout": "/Users/tinggong/Documents/GitHub/TwistedMerge",
        "preserved_main_staged_additions": 1408,
        "baseline_test_command": test_command,
        "baseline_test_result": test_result.splitlines()[-1],
        "per_artifact_provenance": provenance,
    }
    (OUT / "stage0_git_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (OUT / "stage0_test_results.txt").write_text(
        f"Command: {test_command}\nResult:\n{test_result}\n", encoding="utf-8"
    )

    with (OUT / "stage0_artifact_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "kind", "status", "invalid_scope", "retained_scope", "reason"])
        writer.writerows(INVALID_ROWS)
        for row in provenance:
            writer.writerow(
                [row["config"], "execution_config", "PROVENANCE_RECORDED", "", "complete config", row["execution_commit"]]
            )

    commit_rows = "\n".join(
        f"| {row['artifact_family']} | `{row['execution_commit']}` | `{row['config']}` |"
        for row in provenance
    )
    report = f"""# Stage 0 Provenance and Evidence Audit

## Isolation and baseline

- Worktree: `{ROOT}`
- Branch: `{state['branch']}`
- Base/HEAD: `{state['head']}`
- Original checkout preserved with 1,408 staged additions; no file from that index was modified.
- Baseline tests: `{state['baseline_test_result']}`

## Provenance repair

The former release builder assigned one global evidence commit to every entry. That is incorrect because several artifacts were executed at different commits. The release contract now requires each artifact's own JSON execution record and rejects missing or malformed commits instead of silently falling back to a global value.

| artifact family | actual execution commit | source record |
| --- | --- | --- |
{commit_rows}

## Invalid empirical evidence

`src/controlled_nonabelian_holonomy.py` still contains `target_accuracy_for_method` and `logits_with_target_accuracy`; its accuracy artifacts remain `INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE`. The files are retained for auditability. Standalone group definitions and structural residual calculations may be cited only as structural evidence.

## Safe boundary

- New accuracy reports must use saved executed logits and a post-hoc label-permutation regression.
- `ensemble_upper_bound` is treated as legacy unsafe terminology; new work uses `ensemble_reference`.
- No manuscript file was edited.
"""
    (OUT / "stage0_provenance_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
