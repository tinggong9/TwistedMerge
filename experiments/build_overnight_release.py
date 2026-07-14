#!/usr/bin/env python3
"""Build strict per-artifact arXiv/final manifests and manuscript revision map."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def require(paths: list[str]) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError(f"release inputs missing: {missing}")


def manifest_entries() -> list[dict]:
    practical = load_json("practical_selector_config.json")
    hodge = load_json("hodge_lr_smoke_config.json")
    context = load_json("two_loop_context_config.json")
    central = load_json("central_reproduction_manifest.json")
    return [
        {
            "id": "fresh_practical_selector",
            "stage": 1,
            "actual_execution_commit": practical["execution_commit"],
            "dirty_worktree_at_execution": practical["dirty_worktree_at_execution"],
            "script": "experiments/final_practical_selector.py",
            "exact_command": practical["command"],
            "config": "reports/overnight_program/practical_selector_config.json",
            "raw_csv": "reports/overnight_program/practical_selector_runs.csv",
            "summary_csv": "reports/overnight_program/practical_selector_summary.csv",
            "latex_table": "reports/overnight_program/tables/practical_selector_main.tex",
            "plot": "reports/overnight_program/plots/practical_selector_accuracy.pdf",
            "evidence_status": "executed_fresh_accuracy_evidence",
            "paper_eligibility": practical["mode"] == "full",
            "safe_wording": "Fresh MNIST selector evidence; no central or nonabelian lift activated.",
        },
        {
            "id": "hodge_lr_components",
            "stage": 2,
            "actual_execution_commit": hodge["execution_commit"],
            "dirty_worktree_at_execution": hodge["dirty_worktree_at_execution"],
            "script": "experiments/hodge_lr_smoke.py",
            "exact_command": hodge["command"],
            "config": "reports/overnight_program/hodge_lr_smoke_config.json",
            "raw_csv": None,
            "summary_csv": None,
            "latex_table": None,
            "plot": None,
            "evidence_status": "tested_component_smoke",
            "paper_eligibility": False,
            "safe_wording": "Implemented and unit-tested components; no natural-data gain claim.",
        },
        {
            "id": "context_dependent_two_loop",
            "stage": 3,
            "actual_execution_commit": context["execution_commit"],
            "dirty_worktree_at_execution": context["dirty_worktree_at_execution"],
            "script": "experiments/context_dependent_two_loop_holonomy.py",
            "exact_command": context["command"],
            "config": "reports/overnight_program/two_loop_context_config.json",
            "raw_csv": "reports/overnight_program/two_loop_context_runs.csv",
            "summary_csv": "reports/overnight_program/two_loop_context_summary.csv",
            "latex_table": "reports/overnight_program/tables/two_loop_context_main.tex",
            "plot": "reports/overnight_program/plots/two_loop_context_accuracy.pdf",
            "evidence_status": "executed_controlled_accuracy_evidence",
            "paper_eligibility": context["mode"] == "full",
            "safe_wording": "Controlled S3/D4 context task supports noncommuting structure and executed context-lift accuracy; not a natural-checkpoint claim.",
        },
        {
            "id": "controlled_mu2_and_period_index",
            "stage": 4,
            "actual_execution_commit": central["execution_commit"],
            "dirty_worktree_at_execution": central["dirty_worktree_at_execution"],
            "script": "experiments/central_release_overnight.py",
            "exact_command": central["command"],
            "config": "reports/overnight_program/central_reproduction_manifest.json",
            "raw_csv": "reports/overnight_program/central_mu2_runs.csv",
            "summary_csv": "reports/overnight_program/central_mu2_summary.csv",
            "latex_table": "reports/overnight_program/tables/central_mu2.tex",
            "plot": None,
            "evidence_status": "executed_controlled_and_exact_algebraic_evidence",
            "paper_eligibility": True,
            "safe_wording": "Controlled mu2 supplied-context result and exact finite-Heisenberg thresholds; no learned natural-router claim.",
        },
    ]


def scan_forbidden_eligible_artifacts(entries: list[dict]) -> list[dict]:
    forbidden = ("target_accuracy_for_method", "logits_with_target_accuracy")
    hits = []
    for entry in entries:
        if not entry["paper_eligibility"]:
            continue
        for key in ("config", "raw_csv", "summary_csv", "latex_table"):
            path = entry.get(key)
            if not path or not (ROOT / path).exists() or (ROOT / path).suffix.lower() in {".pdf", ".npz"}:
                continue
            text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                if token in text:
                    hits.append({"entry": entry["id"], "path": path, "token": token})
    return hits


def write_numbers(prefix: str) -> None:
    practical = pd.read_csv(OUT / "practical_selector_summary.csv")
    practical = practical[practical["scope"] == "overall"]
    context = pd.read_csv(OUT / "two_loop_context_paired_stats.csv")
    period = pd.read_csv(OUT / "period_index_summary.csv")
    selector = practical[practical.method == "twistedmerge_exact_gauge_soup_selector"].iloc[0]
    soup = practical[practical.method == "ordinary_greedy_soup"].iloc[0]
    context_delta = context[context.method == "supplied_context_oracle"].iloc[0]
    lines = [
        f"\\newcommand{{\\PracticalSelectorAccuracy}}{{{selector.mean_test_accuracy:.4f}}}",
        f"\\newcommand{{\\OrdinaryGreedySoupAccuracy}}{{{soup.mean_test_accuracy:.4f}}}",
        f"\\newcommand{{\\ContextLiftDelta}}{{{context_delta.mean_accuracy_delta:+.4f}}}",
        f"\\newcommand{{\\ContextLiftDeltaLow}}{{{context_delta.ci_low:+.4f}}}",
        f"\\newcommand{{\\PeriodIndexCases}}{{{len(period)}}}",
    ]
    (OUT / f"{prefix}_paper_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def revision_map(entries: list[dict]) -> str:
    practical = pd.read_csv(OUT / "practical_selector_paired_stats.csv")
    selector = practical[practical.method == "twistedmerge_exact_gauge_soup_selector"].iloc[0]
    context = load_json("two_loop_context_config.json")
    return f"""# Immediate manuscript revision map

## Strongest defensible claim

Retain claim-ladder **Level 1**. The new Stage 3 result strengthens the controlled evidence with an executed context-dependent accuracy task, but it does not establish naturally occurring twist-like residuals in independently trained checkpoints. Do not promote Level 2 as a broad real-obstruction claim.

## Retain

- The exact controlled H^2(mu2) construction and supplied-context q=2 representation, explicitly labelled controlled.
- Finite-Heisenberg scalar commutator order, d^k threshold, failed ranks, and direct-sum realizations.
- Executed structural S3/D4 noncommuting holonomy certificates.
- All negative boundaries around learned routing and natural-data evidence.

## Replace

- Replace any aggregated practical-selector numbers with the fresh Stage 1 table. The selector delta versus ordinary greedy soup is {selector.mean_accuracy_delta:+.6f}, 95% CI [{selector.accuracy_delta_ci_low:+.6f}, {selector.accuracy_delta_ci_high:+.6f}].
- Replace any global provenance commit in the evidence manifest with each artifact's `actual_execution_commit`.
- Replace the old context-independent two-loop accuracy discussion with the new controlled context-dependent result, decision {context['decision']}, while retaining the old structural construction as a separate result.

## Delete or quarantine

- Delete target-injected S3/D4 accuracy tables and figures from the empirical-evidence section. Keep only an audit note marked `INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE` if historical provenance matters.
- Delete claims that the validation face-table router generalizes to held-out group words.
- Delete any implication that the frozen-backbone one-seed vision smoke is publication-grade.

## Insert

- `tables/practical_selector_main.tex` and `tables/practical_selector_choices.tex`.
- `tables/two_loop_context_main.tex`, `tables/two_loop_context_structural.tex`, and `tables/two_loop_context_capacity.tex`.
- `tables/central_mu2.tex` and `tables/period_index.tex`.
- Add explicit negative findings: zero central/nonabelian activation in the natural MNIST selector; Hodge/LR remains component-tested; no natural twist discovery yet.

## Paste-ready controlled-framework abstract

We develop TwistedMerge, a descent-theoretic framework for diagnosing when model parameters can be merged after symmetry alignment and when residual transition structure requires a charted representation. We provide an executed controlled central H2(mu2) witness, exact finite-Heisenberg representation thresholds, and structural noncommuting S3/D4 holonomy certificates. A fresh MNIST selector rerun compares strict synchronization, monomial gauges, soups, and validation-only selection without activating unsupported lifts. In a controlled context-dependent two-loop task, an executed chart-aware representation improves over strict synchronization, while natural-checkpoint and broad practical-superiority claims remain open.

## Paste-ready real-twist positive abstract (conditional; do not use yet)

TwistedMerge detects persistent projective or holonomy residuals in independently trained model collections and corrects only their stable low-rank component. Across matched natural-data and pretrained-model benchmarks, certified twist-aware corrections improve held-out performance over synchronization, soups, routing, and capacity-matched controls. This variant is conditional on future Stages 7--11 passing their preregistered gates.

## Paste-ready practical-superiority abstract (conditional; do not use yet)

TwistedMerge is a regime-adaptive model-merging framework that selects ordinary merging in compatible regimes and certified low-rank lifts in obstruction-rich regimes. Across vision and language tasks it improves mean or worst-case accuracy at matched capacity and inference cost. This variant is unsupported until broad pretrained vision, adapter, and transformer evidence passes.

## Paste-ready controlled-framework conclusion

The current evidence supports a controlled taxonomy and exact structural certificates, plus a controlled chart-dependent accuracy advantage. It does not yet support natural twist discovery or broad practical superiority. The practical selector remains conservative and activates no unsupported central or nonabelian candidate.

## Paste-ready real-twist conclusion (conditional)

If natural residuals pass null, stability, prediction, correction, accuracy, capacity, and budget gates, the evidence would support obstruction-aware accuracy improvements beyond controlled constructions. Until then this wording must remain unused.

## Paste-ready practical-superiority conclusion (conditional)

If pretrained vision, adapter, and transformer runs show positive paired intervals against strong baselines at controlled cost, TwistedMerge could be described as broadly regime-adaptive. The present evidence does not meet that bar.

## Priority

1. Remove invalid empirical artifacts and repair provenance.
2. Insert clean controlled and practical-selector tables with negative boundaries.
3. Add the context-dependent controlled benchmark.
4. Keep natural/pretrained claims explicitly future work until Tier B/C gates pass.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=("arxiv", "final"), default="arxiv")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    prefix = "arxiv" if args.release == "arxiv" else "final"
    required = [
        "reports/overnight_program/practical_selector_config.json",
        "reports/overnight_program/practical_selector_runs.csv",
        "reports/overnight_program/hodge_lr_smoke_config.json",
        "reports/overnight_program/two_loop_context_config.json",
        "reports/overnight_program/two_loop_context_runs.csv",
        "reports/overnight_program/central_reproduction_manifest.json",
        "reports/overnight_program/period_index_summary.csv",
    ]
    require(required)
    release_commit = git("rev-parse", "HEAD")
    entries = manifest_entries()
    eligible_commits = {entry["actual_execution_commit"] for entry in entries if entry["paper_eligibility"]}
    dirty_eligible = [entry["id"] for entry in entries if entry["paper_eligibility"] and entry["dirty_worktree_at_execution"]]
    if eligible_commits != {release_commit}:
        raise RuntimeError(f"eligible artifacts were not all executed at release commit {release_commit}: {eligible_commits}")
    if dirty_eligible:
        raise RuntimeError(f"eligible artifacts started from a dirty worktree: {dirty_eligible}")
    forbidden_hits = scan_forbidden_eligible_artifacts(entries)
    if forbidden_hits:
        raise RuntimeError(f"forbidden target-injection tokens found in eligible evidence: {forbidden_hits}")

    test_path = OUT / f"{prefix}_test_results.txt"
    if not args.skip_tests:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True, capture_output=True)
        test_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"full test suite failed; see {test_path}")
    environment = {
        "release_commit": release_commit,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": {key: os.environ.get(key) for key in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")},
    }
    (OUT / f"{prefix}_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    manifest = {
        "release": args.release,
        "release_commit": release_commit,
        "per_artifact_execution_commit_verified": True,
        "clean_execution_verified": True,
        "invalid_target_injection_scan_passed": True,
        "strongest_supported_claim_level": 1,
        "entries": entries,
    }
    manifest_path = OUT / f"{prefix}_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [f"# {prefix.title()} evidence manifest", "", f"Release commit: `{release_commit}`", "", "| id | stage | evidence | paper eligible | execution commit | safe wording |", "|---|---:|---|---|---|---|"]
    for entry in entries:
        lines.append(f"| {entry['id']} | {entry['stage']} | {entry['evidence_status']} | {entry['paper_eligibility']} | `{entry['actual_execution_commit']}` | {entry['safe_wording']} |")
    (OUT / f"{prefix}_release_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_numbers(prefix)
    revision = revision_map(entries)
    (OUT / f"{prefix}_revision_map.md").write_text(revision, encoding="utf-8")
    if args.release == "final":
        # Required aliases in the final-clean-evidence section.
        (OUT / "paper_numbers.tex").write_text((OUT / "final_paper_numbers.tex").read_text(encoding="utf-8"), encoding="utf-8")
        (OUT / "paper_revision_map.md").write_text(revision, encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
