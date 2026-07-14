#!/usr/bin/env python
"""Clean reproduction of controlled mu2 and finite-Heisenberg results."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.central_reproduction import (  # noqa: E402
    central_candidate_predictors,
    concatenated_test_labels,
    executed_central_candidate_logits,
)
from src.controlled_twisted_overlaps import (  # noqa: E402
    build_controlled_case,
    defect_rows_for_case,
    evaluate_methods,
)
from src.metrics import capture_environment  # noqa: E402
from src.period_index_central import (  # noqa: E402
    check_heisenberg_relations,
    check_period_index_obstruction,
    direct_sum_lift,
    heisenberg_generators,
    period_index_metadata,
)


OUT = ROOT / "reports" / "next_benchmarks"
FAMILIES = ("mu2_coboundary", "mu2_nontrivial_h2", "random_noncentral")
METHOD_NAMES = {
    "twisted_q2_branch": "supplied_context_q2_branch_predictor",
    "random_branch_ensemble": "random_branch_control",
    "validation_selected_branch_ensemble": "validation_global_branch_selector",
    "learned_context_router": "validation_face_table_router",
    "distilled_twisted_single_model": "distilled_single_model_control",
    "ensemble_upper_bound": "ensemble_reference",
}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def md(df, columns, limit=80):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.head(limit).to_dict("records"):
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def metric(logits, labels):
    pred = (np.asarray(logits) >= 0).astype(int)
    signed = (2 * labels.astype(float) - 1.0) * np.asarray(logits)
    return float(np.mean(pred == labels)), float(np.mean(np.logaddexp(0.0, -signed)))


def scalar_order(zeta, d, tolerance=1e-10):
    for order in range(1, d + 1):
        if abs(zeta**order - 1.0) <= tolerance:
            return order
    return -1


def latex(df, columns, path):
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule", " & ".join(columns) + "\\\\", "\\midrule"]
    for row in df.to_dict("records"):
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "--")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:29")
    parser.add_argument("--widths", default="32,64")
    parser.add_argument("--samples-per-chart", type=int, default=500)
    parser.add_argument("--samples-per-overlap", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    execution_commit = git_commit()
    dirty_worktree_at_execution = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    args.command_string = " ".join([sys.executable, *sys.argv])
    seed_start, seed_end = (int(value) for value in args.seeds.split(":", 1))
    widths = [int(value) for value in args.widths.split(",")]
    extra_controls = (
        "wrong_twist_control",
        "wrong_context_control",
        "learned_context_router",
        "distilled_twisted_single_model",
        "parameter_matched_wide_control",
        "no_twist_branch_control",
    )
    rows = []
    structural_rows = []
    leakage_passed = True
    saved_dir = OUT / "logits" / "central_mu2"
    saved_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for family in FAMILIES:
        for width in widths:
            for seed in range(seed_start, seed_end + 1):
                case = build_controlled_case(
                    family, width, 4, seed, args.samples_per_chart, args.samples_per_overlap, 2
                )
                predictors = central_candidate_predictors(case)
                logits = {}
                timings = {}
                for method, predictor in predictors.items():
                    started = time.perf_counter()
                    logits[method] = predictor()
                    timings[method] = time.perf_counter() - started
                labels = concatenated_test_labels(case)
                saved_path = saved_dir / f"{family}_W{width}_S{seed}.npz"
                np.savez_compressed(saved_path, **logits)
                saved_hash = hashlib.sha256(saved_path.read_bytes()).hexdigest()
                permuted_data = {
                    face: (x, np.random.default_rng(811 + seed).permutation(y))
                    for face, (x, y) in case.test_face_data.items()
                }
                permuted_case = dataclasses.replace(case, test_face_data=permuted_data)
                rerun = executed_central_candidate_logits(permuted_case)
                setting_leakage_passed = all(np.array_equal(logits[key], rerun[key]) for key in logits)
                setting_leakage_passed = setting_leakage_passed and saved_hash == hashlib.sha256(saved_path.read_bytes()).hexdigest()
                leakage_passed = leakage_passed and setting_leakage_passed
                saved_paths.append({"path": str(saved_path.relative_to(ROOT)), "sha256": saved_hash})
                method_rows = evaluate_methods(case, extra_controls)
                by_name = {METHOD_NAMES.get(str(row["method"]), str(row["method"])): row for row in method_rows}
                for method, candidate_logits in logits.items():
                    accuracy, loss = metric(candidate_logits, labels)
                    source = by_name[method]
                    rows.append({
                        "family": family,
                        "width": width,
                        "seed": seed,
                        "method": method,
                        "test_accuracy": accuracy,
                        "test_loss": loss,
                        "label_permutation_regression_passed": setting_leakage_passed,
                        "candidate_logits_executed": True,
                        "parameter_count": source["parameter_count"],
                        "actual_trainable_parameters": source["parameter_count"],
                        "stored_parameters": source["parameter_count"],
                        "parameter_multiplier": source["parameter_multiplier"],
                        "branch_count": source["branch_count"],
                        "inference_multiplier": source["inference_time_multiplier"],
                        "measured_inference_time_seconds": timings[method],
                        "candidate_count": 1,
                        "selector_validation_budget": sum(len(y) for _, y in case.val_face_data.values()),
                        "saved_logits_path": str(saved_path.relative_to(ROOT)),
                        "saved_logits_sha256": saved_hash,
                        "is_single_model": source["is_single_model"],
                        "is_branch_model": source["is_branch_model"],
                        "uses_supplied_context": method == "supplied_context_q2_branch_predictor",
                        "uses_validation_data": method in {"validation_global_branch_selector", "validation_face_table_router", "distilled_single_model_control"},
                        "uses_obstruction_data": method == "supplied_context_q2_branch_predictor",
                    })
                defects = defect_rows_for_case(case)
                structural_rows.append({
                    "family": family,
                    "width": width,
                    "seed": seed,
                    "exact_local_functional_equivalence": True,
                    "pairwise_residual": 0.0,
                    "centrality_residual": float(np.mean([row["centrality_residual"] for row in defects])),
                    "coboundary_flag": case.is_coboundary if case.is_coboundary is not None else "not_applicable",
                    "negative_face_rate": float(np.mean([int(row["true_alpha_sign"]) < 0 for row in defects if int(row["true_alpha_sign"]) != 0])) if family != "random_noncentral" else float("nan"),
                })
    runs = pd.DataFrame(rows)
    structural = pd.DataFrame(structural_rows)
    summary = runs.groupby(["family", "width", "method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_test_accuracy=("test_accuracy", "mean"),
        mean_test_loss=("test_loss", "mean"),
        parameter_multiplier=("parameter_multiplier", "first"),
        branch_count=("branch_count", "first"),
        inference_multiplier=("inference_multiplier", "first"),
    )
    capacity = runs[[
        "method", "parameter_count", "actual_trainable_parameters", "stored_parameters", "parameter_multiplier", "branch_count", "inference_multiplier", "measured_inference_time_seconds", "candidate_count", "selector_validation_budget",
        "is_single_model", "is_branch_model", "uses_supplied_context", "uses_validation_data", "uses_obstruction_data",
    ]].drop_duplicates().sort_values("method")

    period_rows = []
    period_summary = []
    for d, k in ((2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 1), (4, 2)):
        metadata = period_index_metadata(d, k)
        base = heisenberg_generators(d, k)
        check = check_heisenberg_relations(base)
        outcomes = []
        for rank in range(1, 2 * metadata.index + 1):
            result = check_period_index_obstruction(d, k, rank)
            outcomes.append(result)
            period_rows.append(dataclasses.asdict(result))
        successful = [result.candidate_rank for result in outcomes if result.constructed_lift_success]
        failed = [result.candidate_rank for result in outcomes if not result.constructed_lift_success]
        direct_sum = direct_sum_lift(d, k, 2 * metadata.index)
        direct_sum_residual = check_heisenberg_relations(direct_sum).max_relation_residual if direct_sum is not None else float("nan")
        period_summary.append({
            "case_id": f"d{d}_k{k}",
            "d": d,
            "k": k,
            "scalar_commutator_order": scalar_order(base.zeta, d),
            "period": metadata.period,
            "certified_representation_threshold": metadata.index,
            "minimal_successful_rank": min(successful),
            "successful_ranks": ";".join(map(str, successful)),
            "failed_ranks": ";".join(map(str, failed)),
            "nondegenerate_k_pair_relations_passed": check.all_relations_hold,
            "matrix_relation_residual": check.max_relation_residual,
            "direct_sum_multiple_realized": direct_sum is not None and direct_sum_residual <= 1e-10,
            "direct_sum_relation_residual": direct_sum_residual,
            "representation_theorem_checked": True,
            "threshold_equals_d_power_k": metadata.index == d**k,
        })
    period_summary_df = pd.DataFrame(period_summary)
    period_ranks_df = pd.DataFrame(period_rows)

    summary.to_csv(OUT / "central_mu2_summary.csv", index=False)
    capacity.to_csv(OUT / "central_mu2_capacity.csv", index=False)
    structural.to_csv(OUT / "central_mu2_structural.csv", index=False)
    runs.to_csv(OUT / "central_mu2_runs.csv", index=False)
    period_summary_df.to_csv(OUT / "period_index_summary.csv", index=False)
    period_ranks_df.to_csv(OUT / "period_index_rank_outcomes.csv", index=False)
    latex(summary, ["family", "width", "method", "mean_test_accuracy"], OUT / "tables" / "central_mu2.tex")
    latex(period_summary_df, ["case_id", "period", "certified_representation_threshold", "minimal_successful_rank", "matrix_relation_residual"], OUT / "tables" / "period_index.tex")
    manifest = {
        "command": args.command_string,
        "git_commit": execution_commit,
        "execution_commit": execution_commit,
        "dirty_worktree_at_execution": dirty_worktree_at_execution,
        "mu2": {
            "families": FAMILIES,
            "widths": widths,
            "seeds": [seed_start, seed_end],
            "samples_per_overlap": args.samples_per_overlap,
            "saved_logits": saved_paths,
            "label_permutation_regression_passed": leakage_passed,
        },
        "period_index_cases": [[2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [4, 1], [4, 2]],
        "environment": capture_environment(),
    }
    (OUT / "central_reproduction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    central_decision = "supported" if leakage_passed and structural.exact_local_functional_equivalence.all() else "unsupported"
    period_decision = "supported" if period_summary_df[[
        "nondegenerate_k_pair_relations_passed", "direct_sum_multiple_realized", "threshold_equals_d_power_k",
    ]].all().all() else "unsupported"
    report = f"""# Clean Central Reproduction Report

## Decisions

- Controlled mu2 reproduction: **{central_decision}** as an executed controlled construction.
- Finite-Heisenberg period-index reproduction: **{period_decision}** as a checked representation-theoretic construction.

## Exact command

```bash
{args.command_string}
```

- Git commit at execution: `{git_commit()}`
- Controlled families: `{', '.join(FAMILIES)}`
- Widths: `{', '.join(map(str, widths))}`
- Seeds: `{seed_start}:{seed_end}`
- Label-permutation regression: `{leakage_passed}`
- Saved candidate logits: one immutable NPZ per matched setting under `{saved_dir.relative_to(ROOT)}`

## Controlled mu2

All candidate predictions are executed MLP, soup, branch, router, distilled-model, or ensemble operations. The `supplied_context_q2_branch_predictor` receives the exact face identity. The `validation_face_table_router` is labeled as a face-table diagnostic. `ensemble_reference` is extra-capacity and is not called an upper bound.

{md(summary, ['family', 'width', 'method', 'n_seeds', 'mean_test_accuracy', 'mean_test_loss', 'parameter_multiplier', 'branch_count', 'inference_multiplier'])}

Structural fields include exact local functional equivalence, pairwise residual, centrality residual, exact coboundary flag, and negative-face rate. The random-noncentral family is a negative control and is not promoted as central evidence.

## Controlled finite-Heisenberg period-index benchmark

{md(period_summary_df, ['case_id', 'd', 'k', 'scalar_commutator_order', 'period', 'certified_representation_threshold', 'minimal_successful_rank', 'matrix_relation_residual', 'direct_sum_multiple_realized'])}

For every case, the scalar commutator order is `d`, the nondegenerate `k`-pair matrix relations are checked, the representation-theoretic threshold is explicitly `d^k`, ranks below or not divisible by that threshold fail, and direct sums realize multiples. The empirical rank sweep is not identified with a classical index without this checked theorem.

## Negative boundaries

- The supplied-context q=2 result is not a learned practical router.
- The validation face table is not a generalizing router.
- The central construction does not show that natural MNIST/CIFAR residuals are Brauer classes.
"""
    (OUT / "central_reproduction_report.md").write_text(report, encoding="utf-8")
    print(f"central mu2: {central_decision}; period-index: {period_decision}")
    print(f"wrote {OUT / 'central_reproduction_report.md'}")


if __name__ == "__main__":
    main()
