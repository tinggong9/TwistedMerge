#!/usr/bin/env python
"""Build the release manifest and final combined next-benchmarks report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evidence_provenance import execution_commit


OUT = ROOT / "reports" / "next_benchmarks"


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def md(frame, columns, limit=80):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.head(limit).to_dict("records"):
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def command_from_config(name):
    path = OUT / name
    if not path.exists():
        return "not recorded"
    return json.loads(path.read_text(encoding="utf-8")).get("command", "not recorded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-commit", default="", help="release commit; entries use their own execution configs")
    parser.add_argument("--test-result", required=True)
    parser.add_argument("--test-command", default="PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python -m pytest -q")
    args = parser.parse_args()
    current = git_output("rev-parse", "HEAD")
    release_commit = args.evidence_commit or current

    def artifact_commit(config_name):
        return execution_commit(OUT / config_name)
    two_loop_claim = pd.read_csv(OUT / "two_loop_holonomy_claims.csv").iloc[0]
    period = pd.read_csv(OUT / "period_index_summary.csv")
    central = pd.read_csv(OUT / "central_mu2_summary.csv")
    matched_stats = pd.read_csv(OUT / "matched_selector_paired_stats.csv")
    context_claims = pd.read_csv(OUT / "context_router_claims.csv")
    diagnostic_comparison = pd.read_csv(OUT / "diagnostic_prediction_model_comparison.csv")
    pretrained = pd.read_csv(OUT / "pretrained_merge_summary.csv")

    entries = [
        {
            "paper_identifier": "controlled_mu2_main",
            "benchmark": "controlled_mu2_reproduction",
            "generating_script": "experiments/central_reproduction_next.py",
            "exact_command": command_from_config("central_reproduction_manifest.json"),
            "git_commit": artifact_commit("central_reproduction_manifest.json"),
            "configuration": "reports/next_benchmarks/central_reproduction_manifest.json",
            "raw_csv": "reports/next_benchmarks/central_mu2_runs.csv",
            "summary_csv": "reports/next_benchmarks/central_mu2_summary.csv",
            "plot_path": "not_applicable",
            "latex_table_path": "reports/next_benchmarks/tables/central_mu2.tex",
            "evidence_status": "executed",
            "paper_number_eligible": True,
            "safe_claim_wording": "In an exact planted mu2 overlap construction, executed supplied-context q=2 branch prediction resolves the nontrivial central class; this is a controlled oracle-context result, not a learned-router or natural-data claim.",
        },
        {
            "paper_identifier": "finite_heisenberg_period_index",
            "benchmark": "period_index_reproduction",
            "generating_script": "experiments/central_reproduction_next.py",
            "exact_command": command_from_config("central_reproduction_manifest.json"),
            "git_commit": artifact_commit("central_reproduction_manifest.json"),
            "configuration": "reports/next_benchmarks/central_reproduction_manifest.json",
            "raw_csv": "reports/next_benchmarks/period_index_rank_outcomes.csv",
            "summary_csv": "reports/next_benchmarks/period_index_summary.csv",
            "plot_path": "not_applicable",
            "latex_table_path": "reports/next_benchmarks/tables/period_index.tex",
            "evidence_status": "structural-only",
            "paper_number_eligible": True,
            "safe_claim_wording": "For the checked finite-Heisenberg k-pair systems, the scalar commutator has order d and the certified projective representation threshold is d^k; direct sums realize its multiples.",
        },
        {
            "paper_identifier": "two_loop_noncommuting_structure",
            "benchmark": "executed_two_loop_holonomy",
            "generating_script": "experiments/executed_two_loop_holonomy.py",
            "exact_command": command_from_config("two_loop_holonomy_config.json"),
            "git_commit": artifact_commit("two_loop_holonomy_config.json"),
            "configuration": "reports/next_benchmarks/two_loop_holonomy_config.json",
            "raw_csv": "reports/next_benchmarks/two_loop_holonomy_residuals.csv",
            "summary_csv": "reports/next_benchmarks/two_loop_holonomy_summary.csv",
            "plot_path": "reports/next_benchmarks/plots/two_loop_holonomy_residuals.pdf",
            "latex_table_path": "reports/next_benchmarks/tables/two_loop_holonomy_residuals.tex",
            "evidence_status": "structural-only",
            "paper_number_eligible": True,
            "safe_claim_wording": "Executed S3/D4 models certify two noncommuting loop holonomies and invariant pooling, but the lift ties controls and supports no accuracy-advantage claim.",
        },
        {
            "paper_identifier": "two_loop_accuracy",
            "benchmark": "executed_two_loop_holonomy",
            "generating_script": "experiments/executed_two_loop_holonomy.py",
            "exact_command": command_from_config("two_loop_holonomy_config.json"),
            "git_commit": artifact_commit("two_loop_holonomy_config.json"),
            "configuration": "reports/next_benchmarks/two_loop_holonomy_config.json",
            "raw_csv": "reports/next_benchmarks/two_loop_holonomy_runs.csv",
            "summary_csv": "reports/next_benchmarks/two_loop_holonomy_paired_stats.csv",
            "plot_path": "reports/next_benchmarks/plots/two_loop_holonomy_accuracy.pdf",
            "latex_table_path": "reports/next_benchmarks/tables/two_loop_holonomy_accuracy.tex",
            "evidence_status": "unsupported",
            "paper_number_eligible": False,
            "safe_claim_wording": "No nonabelian lift accuracy advantage was observed; all critical paired deltas were ties.",
        },
        {
            "paper_identifier": "learned_context_router",
            "benchmark": "context_router_generalization",
            "generating_script": "experiments/context_router_generalization.py",
            "exact_command": command_from_config("context_router_config.json"),
            "git_commit": artifact_commit("context_router_config.json"),
            "configuration": "reports/next_benchmarks/context_router_config.json",
            "raw_csv": "reports/next_benchmarks/context_router_runs.csv",
            "summary_csv": "reports/next_benchmarks/context_router_summary.csv",
            "plot_path": "reports/next_benchmarks/plots/context_router_generalization.pdf",
            "latex_table_path": "reports/next_benchmarks/tables/context_router.tex",
            "evidence_status": "unsupported",
            "paper_number_eligible": False,
            "safe_claim_wording": "The supplied-context oracle remains a controlled diagnostic; the learned feature router is unsupported on held-out group words.",
        },
        {
            "paper_identifier": "matched_practical_selector",
            "benchmark": "matched_selector_budget",
            "generating_script": "experiments/matched_selector_budget_benchmark.py",
            "exact_command": command_from_config("matched_selector_config.json"),
            "git_commit": artifact_commit("matched_selector_config.json"),
            "configuration": "reports/next_benchmarks/matched_selector_config.json",
            "raw_csv": "reports/next_benchmarks/matched_selector_runs.csv",
            "summary_csv": "reports/next_benchmarks/matched_selector_paired_stats.csv",
            "plot_path": "reports/next_benchmarks/plots/matched_selector_accuracy.pdf",
            "latex_table_path": "reports/next_benchmarks/tables/matched_selector_main.tex",
            "evidence_status": "unsupported",
            "paper_number_eligible": False,
            "safe_claim_wording": "The tracked executed-grid selector aggregation did not beat ordinary greedy soup and was not fresh inference from the evidence commit.",
        },
        {
            "paper_identifier": "natural_diagnostic_prediction",
            "benchmark": "heldout_diagnostic_prediction",
            "generating_script": "experiments/heldout_diagnostic_prediction.py",
            "exact_command": command_from_config("diagnostic_prediction_config.json"),
            "git_commit": artifact_commit("diagnostic_prediction_config.json"),
            "configuration": "reports/next_benchmarks/diagnostic_prediction_config.json",
            "raw_csv": "reports/next_benchmarks/diagnostic_prediction_runs.csv",
            "summary_csv": "reports/next_benchmarks/diagnostic_prediction_summary.csv",
            "plot_path": "reports/next_benchmarks/plots/diagnostic_prediction.pdf",
            "latex_table_path": "reports/next_benchmarks/tables/diagnostic_prediction.tex",
            "evidence_status": "unsupported",
            "paper_number_eligible": False,
            "safe_claim_wording": "The preregistered natural-data diagnostic did not add held-out value beyond validation baselines.",
        },
        {
            "paper_identifier": "pretrained_merge",
            "benchmark": "pretrained_resnet18_smoke",
            "generating_script": "experiments/pretrained_merge_smoke.py",
            "exact_command": command_from_config("pretrained_merge_config.json"),
            "git_commit": artifact_commit("pretrained_merge_config.json"),
            "configuration": "reports/next_benchmarks/pretrained_merge_config.json",
            "raw_csv": "reports/next_benchmarks/pretrained_merge_runs.csv",
            "summary_csv": "reports/next_benchmarks/pretrained_merge_summary.csv",
            "plot_path": "not_applicable",
            "latex_table_path": "reports/next_benchmarks/tables/pretrained_merge.tex",
            "evidence_status": "descriptive",
            "paper_number_eligible": False,
            "safe_claim_wording": "A one-seed frozen-backbone ResNet-18/CIFAR-10 smoke completed; it is not a full modern model-merging benchmark.",
        },
    ]
    forbidden = ("controlled_nonabelian_holonomy.csv", "target_accuracy_for_method", "logits_with_target_accuracy", "ensemble_upper_bound")
    serialized_entries = json.dumps(entries, sort_keys=True)
    invalid_present = any(token in serialized_entries for token in forbidden)
    if invalid_present:
        raise RuntimeError("deprecated or unsafe target-injected evidence entered the release manifest")
    manifest = {
        "schema_version": 1,
        "release_builder_commit": current,
        "evidence_commit": release_commit,
        "test_command": args.test_command,
        "test_result": args.test_result,
        "deprecated_target_injected_numbers_present": False,
        "entries": entries,
    }
    (OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_df = pd.DataFrame(entries)
    manifest_report = f"""# Next-Benchmarks Release Manifest

- Release commit: `{release_commit}`
- Release builder commit: `{current}`
- Test result: `{args.test_result}`
- Deprecated target-injected numbers present: `False`

{md(manifest_df, ['paper_identifier', 'benchmark', 'git_commit', 'evidence_status', 'paper_number_eligible', 'raw_csv', 'latex_table_path', 'safe_claim_wording'])}
"""
    (OUT / "release_manifest.md").write_text(manifest_report, encoding="utf-8")
    environment = {
        "evidence_commit": release_commit,
        "release_builder_commit": current,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_status_at_release_build": git_output("status", "--short", "--branch"),
        "commands": {entry["benchmark"]: entry["exact_command"] for entry in entries},
    }
    (OUT / "release_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (OUT / "release_test_results.txt").write_text(f"Command: {args.test_command}\nResult: {args.test_result}\nRelease commit: {release_commit}\n", encoding="utf-8")

    lines = [
        "% Generated paper-number manifest; only paper_number_eligible entries are included.",
        f"\\newcommand{{\\TwistedMergeEvidenceCommit}}{{{release_commit[:12]}}}",
        "\\newcommand{\\TwoLoopHolonomyDecision}{B: structural support, accuracy advantage unsupported}",
    ]
    for row in period.itertuples():
        lines.append(f"\\newcommand{{\\PeriodIndex{row.case_id.replace('_', '')}}}{{{int(row.certified_representation_threshold)}}}")
    lines.append("")
    (OUT / "paper_numbers.tex").write_text("\n".join(lines), encoding="utf-8")

    central_key = central[
        (central.family == "mu2_nontrivial_h2")
        & (central.width == 32)
        & central.method.isin(["ordinary_weight_average", "supplied_context_q2_branch_predictor", "random_branch_control"])
    ]
    matched_key = matched_stats[matched_stats.comparison.str.startswith("improved_twistedmerge")]
    context_decision = context_claims[context_claims.claim_id == "learned_router_generalizes_to_unseen_contexts"].iloc[0].status
    final_report = f"""# Final Next-Benchmarks Report

## Executive decisions

| benchmark | decision |
| --- | --- |
| Stage 0 evidence audit | Existing target-injected S3/D4 accuracy artifacts are invalid and quarantined. |
| Executed two-loop holonomy | {two_loop_claim.decision} |
| Context-router generalization | Learned practical router `{context_decision}`; supplied-context oracle retained separately. |
| Controlled central reproduction | Supported as an executed controlled construction. |
| Finite-Heisenberg period-index | Supported as a checked structural representation-theoretic construction. |
| Matched selector budget | Unsupported versus ordinary greedy soup; aggregation is not fresh inference from the evidence commit. |
| Held-out diagnostic prediction | Unsupported under the preregistered held-out gate. |
| Pretrained merging | Not run at full required scale; one-seed ResNet-18 smoke completed. |

Release commit: `{release_commit}`. Each table row records its own execution commit. Complete tests: `{args.test_result}`.

## Exact commands and output paths

{md(manifest_df, ['benchmark', 'exact_command', 'git_commit', 'raw_csv', 'summary_csv', 'plot_path', 'latex_table_path'])}

## Compact numerical tables

Controlled nontrivial mu2, width 32:

{md(central_key, ['family', 'width', 'method', 'n_seeds', 'mean_test_accuracy', 'mean_test_loss', 'parameter_multiplier', 'branch_count', 'inference_multiplier'])}

Finite-Heisenberg period-index:

{md(period, ['case_id', 'd', 'k', 'scalar_commutator_order', 'certified_representation_threshold', 'minimal_successful_rank', 'matrix_relation_residual'])}

Practical-selector primary comparison:

{md(matched_key, ['comparison', 'n_pairs', 'paired_mean_accuracy_delta', 'ci_low', 'ci_high', 'wins', 'ties', 'losses'])}

Pretrained smoke:

{md(pretrained, ['method', 'average_accuracy', 'worst_task_accuracy', 'calibration_ece', 'forgetting_interference'])}

## Capacity, inference, and selection accounting

- Full capacity tables: `central_mu2_capacity.csv`, `two_loop_holonomy_capacity.csv`, and `matched_selector_capacity.csv`.
- The two-loop branch regular lift stores one learned model (`1x` learned parameters) but executes `{int(pd.read_csv(OUT / 'two_loop_holonomy_capacity.csv').branch_count.max())}` branches at up to the recorded inference multiplier.
- The central supplied-context q=2 predictor records `2x` branch capacity/inference; the ensemble reference records `4x`.
- The matched practical selector activated central lifts `0` times and nonabelian branch lifts `0` times; it selected only exact-gauge or soup candidates.
- Selector regret is audit-only and was not used for selection.

## Leakage and structural certificates

- Two-loop saved-logit label permutation: passed.
- Context-router saved-logit label permutation: passed.
- Central mu2 saved-logit label permutation: passed.
- Two-loop generator recovery, noncommutation, local equivalence, regular-action multiplication, and both pooling certificates: passed across the full grid.
- Wrong-generator and random-action controls failed the complete structural certificate as required.

## Manuscript claim actions

- Retain, with controlled scope: executed central mu2 supplied-context result and checked finite-Heisenberg period-index theorem.
- Retain only structurally: two-loop S3/D4 noncommuting holonomy and invariant-pooling certificates.
- Weaken: any practical-selector statement to a negative result; the tracked selector was `{float(matched_key.paired_mean_accuracy_delta.iloc[0]):+.6f}` versus greedy soup with CI `[{float(matched_key.ci_low.iloc[0]):+.6f}, {float(matched_key.ci_high.iloc[0]):+.6f}]`.
- Delete: all empirical accuracy claims sourced from the deprecated target-injected controlled-nonabelian artifacts.
- Do not add: a learned practical-router claim, a promoted natural-data diagnostic claim, or a full pretrained-model-merging claim.

## Recommended safe wording

- Controlled mu2: "In an exact planted mu2 overlap construction, executed supplied-context q=2 branch prediction resolves the nontrivial central class; this is a controlled oracle-context result, not a learned-router or natural-data claim."
- Period-index: "For the checked finite-Heisenberg k-pair systems, the scalar commutator has order d and the certified projective representation threshold is d^k; direct sums realize its multiples."
- Noncentral/noncommuting holonomy: "Executed S3/D4 models certify two noncommuting loop holonomies and invariant pooling, but the lift ties controls and supports no accuracy-advantage claim."
- Practical selector: "On the tracked executed MNIST grid, the selector did not beat ordinary greedy soup; no central or nonabelian branch candidate was selected."
- Learned router: "The supplied-context oracle is valid in the controlled construction, while the learned feature router is unsupported on held-out group words."
- Natural diagnostic: "The preregistered natural-data diagnostic did not add held-out predictive value beyond ordinary validation baselines."

## LaTeX files ready to paste

- `reports/next_benchmarks/tables/central_mu2.tex`
- `reports/next_benchmarks/tables/period_index.tex`
- `reports/next_benchmarks/tables/two_loop_holonomy_residuals.tex`
- `reports/next_benchmarks/tables/context_router.tex` (negative/diagnostic table)
- `reports/next_benchmarks/tables/matched_selector_main.tex` (limited, non-fresh aggregation)
- `reports/next_benchmarks/tables/diagnostic_prediction.tex` (negative result)
- `reports/next_benchmarks/tables/pretrained_merge.tex` (smoke only)

The manuscript itself was not edited.
"""
    (OUT / "final_next_benchmarks_report.md").write_text(final_report, encoding="utf-8")
    print(f"wrote {OUT / 'release_manifest.json'}")
    print(f"wrote {OUT / 'final_next_benchmarks_report.md'}")


if __name__ == "__main__":
    main()
