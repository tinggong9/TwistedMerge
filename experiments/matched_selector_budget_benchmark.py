#!/usr/bin/env python
"""Matched candidate-budget audit over the executed MNIST MLP grid."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "next_benchmarks"

METHOD_MAP = {
    "weight_average": "ordinary_weight_average",
    "c2m3_permutation": "c2m3_synchronization_alone",
    "c2m3_greedy_soup": "c2m3_greedy_soup",
    "monomial_scale": "raw_monomial_alignment_alone",
    "monomial_scaled_greedy_soup": "monomial_greedy_soup",
    "shrinkage_monomial_scale": "shrinkage_monomial",
    "global_monomial_scale": "global_monomial",
    "optimized_monomial_scale": "optimized_monomial",
    "greedy_soup": "ordinary_greedy_soup",
    "union_candidate_soup": "union_candidate_soup",
    "improved_validated_selector": "improved_twistedmerge_exact_gauge_soup_selector",
}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def param_count(width):
    return int(784 * width + width + width * 10 + 10)


def bootstrap_ci(values, seed, n=2000):
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        value = float(arr.mean()) if len(arr) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, len(arr), replace=True).mean()) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def md(df, columns, limit=100):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.head(limit).to_dict("records"):
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def add_random_augmented_controls(df):
    rows = []
    pool_methods = [
        "ordinary_greedy_soup",
        "c2m3_greedy_soup",
        "monomial_greedy_soup",
        "union_candidate_soup",
        "raw_monomial_alignment_alone",
        "c2m3_synchronization_alone",
    ]
    for setting_id, group in df.groupby("setting_id", sort=False):
        pool = group[group.method.isin(pool_methods)].copy()
        if pool.empty:
            continue
        rng = np.random.default_rng(int(pool.seed.iloc[0]) + 91009)
        augmented = pd.concat([pool, pool.iloc[rng.integers(0, len(pool), size=len(pool))]], ignore_index=True)
        chosen = augmented.sort_values(["val_accuracy", "val_loss", "method"], ascending=[False, True, True]).iloc[0].copy()
        chosen["source_method"] = chosen["method"]
        chosen["method"] = "randomly_augmented_candidate_union"
        chosen["candidate_count"] = int(len(augmented))
        chosen["selected_candidate_count"] = 1
        chosen["selection_control"] = True
        chosen["test_used_for_selection"] = False
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "reports/csv/improved_validated_ladder_merge_benchmark.csv")
    parser.add_argument("--external-source", type=Path, default=ROOT / "reports/csv/external_baseline_comparison.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    source = pd.read_csv(args.source)
    expected = {(n, width, seed) for n in (3, 4) for width in (16, 32, 64) for seed in range(1800, 1820)}
    observed = set(map(tuple, source[["n_models", "width", "seed"]].drop_duplicates().to_numpy()))
    if expected != observed:
        raise RuntimeError(f"source grid mismatch: missing={sorted(expected - observed)[:10]}")
    if not bool(source["selector_no_test_leakage"].fillna(False).astype(bool).all()):
        raise RuntimeError("source contains a selector without the no-test-leakage flag")
    selected = source[source.method.isin(METHOD_MAP)].copy()
    selected["source_method"] = selected["method"]
    selected["method"] = selected["method"].map(METHOD_MAP)
    selected["source_csv"] = str(args.source.relative_to(ROOT))
    selected["fresh_inference_on_current_commit"] = False
    selected["clean_aggregation_rerun"] = True
    selected["candidate_logits_origin"] = "executed_torch_models_from_tracked_full_grid"
    selected["actual_parameter_count"] = selected.width.map(param_count)
    selected["parameter_multiplier"] = 1.0
    selected["inference_multiplier"] = 1.0
    selected["branch_count"] = 1
    selected["candidate_count"] = pd.to_numeric(selected.get("union_candidate_count", 1), errors="coerce").fillna(1).astype(int)
    selected["selected_candidate_count"] = pd.to_numeric(selected.get("soup_ingredient_count", 1), errors="coerce").fillna(1).astype(int).clip(lower=1)
    selected["validation_budget"] = 1000
    selected["test_used_for_selection"] = False
    selected["selection_control"] = False
    selected["central_lift_activated"] = False
    selected["nonabelian_branch_lift_activated"] = False
    selected["obstruction_gated_branch_activated"] = False

    random_control = add_random_augmented_controls(selected)
    if not random_control.empty:
        selected = pd.concat([selected, random_control], ignore_index=True, sort=False)

    # Git-ReBasin was executed only on the tracked 20-setting external subset.
    external = pd.read_csv(args.external_source)
    external = external[external.method == "git_rebasin_pairwise"].copy()
    if not external.empty:
        external["source_method"] = external["method"]
        external["method"] = "git_rebasin_style_pairwise"
        external["source_csv"] = str(args.external_source.relative_to(ROOT))
        external["fresh_inference_on_current_commit"] = False
        external["clean_aggregation_rerun"] = True
        external["candidate_logits_origin"] = "executed_torch_models_from_tracked_external_subset"
        external["actual_parameter_count"] = external.width.map(param_count)
        external["parameter_multiplier"] = 1.0
        external["inference_multiplier"] = 1.0
        external["branch_count"] = 1
        external["candidate_count"] = 1
        external["selected_candidate_count"] = 1
        external["validation_budget"] = 1000
        external["test_used_for_selection"] = False
        external["selection_control"] = False
        external["central_lift_activated"] = False
        external["nonabelian_branch_lift_activated"] = False
        external["obstruction_gated_branch_activated"] = False
        selected = pd.concat([selected, external], ignore_index=True, sort=False)

    # Required branch controls have no certified natural candidate in this grid.
    blockers = []
    for method in ("always_on_branch_candidate", "obstruction_gated_branch_candidate"):
        blockers.append({
            "method": method,
            "evaluation_status": "not_run_exact_blocker",
            "blocker": "No certified central or nonabelian branch tensor exists for any primary-grid setting; constructing one without a certificate would violate the benchmark gate.",
            "n_settings": 0,
        })
    blocker_df = pd.DataFrame(blockers)

    selected["accuracy"] = pd.to_numeric(selected["accuracy"], errors="coerce")
    selected["val_accuracy"] = pd.to_numeric(selected["val_accuracy"], errors="coerce")
    selected["loss"] = pd.to_numeric(selected["loss"], errors="coerce")
    summary = selected.groupby("method", as_index=False).agg(
        n_rows=("setting_id", "count"),
        n_settings=("setting_id", "nunique"),
        n_seeds=("seed", "nunique"),
        mean_test_accuracy=("accuracy", "mean"),
        mean_validation_accuracy=("val_accuracy", "mean"),
        mean_parameter_multiplier=("parameter_multiplier", "mean"),
        mean_inference_multiplier=("inference_multiplier", "mean"),
        mean_candidate_count=("candidate_count", "mean"),
        mean_selected_candidate_count=("selected_candidate_count", "mean"),
        validation_budget=("validation_budget", "first"),
        fresh_inference_on_current_commit=("fresh_inference_on_current_commit", "all"),
    )
    summary["coverage_status"] = np.where(summary.n_settings == 120, "complete_primary_grid", "partial_grid")

    comparisons = [
        ("improved_twistedmerge_exact_gauge_soup_selector", "ordinary_greedy_soup"),
        ("union_candidate_soup", "ordinary_greedy_soup"),
        ("randomly_augmented_candidate_union", "ordinary_greedy_soup"),
        ("raw_monomial_alignment_alone", "c2m3_synchronization_alone"),
        ("shrinkage_monomial", "raw_monomial_alignment_alone"),
        ("global_monomial", "raw_monomial_alignment_alone"),
        ("optimized_monomial", "raw_monomial_alignment_alone"),
    ]
    stats_rows = []
    pivot = selected.pivot_table(index="setting_id", columns="method", values="accuracy", aggfunc="first")
    for left, right in comparisons:
        delta = (pivot[left] - pivot[right]).dropna().to_numpy()
        low, high = bootstrap_ci(delta, 1001 + len(stats_rows), args.bootstrap_samples)
        stats_rows.append({
            "comparison": f"{left}_vs_{right}",
            "n_pairs": len(delta),
            "paired_mean_accuracy_delta": float(delta.mean()),
            "ci_low": low,
            "ci_high": high,
            "wins": int((delta > 1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
        })
    stats = pd.DataFrame(stats_rows)

    candidate_methods = [
        "ordinary_weight_average", "c2m3_synchronization_alone", "ordinary_greedy_soup",
        "raw_monomial_alignment_alone", "shrinkage_monomial", "global_monomial", "optimized_monomial",
        "union_candidate_soup",
    ]
    audit_pivot = selected[selected.method.isin(candidate_methods)].pivot_table(index="setting_id", columns="method", values="accuracy", aggfunc="first")
    best_test = audit_pivot.max(axis=1)
    choices = selected[selected.method.isin([
        "improved_twistedmerge_exact_gauge_soup_selector", "randomly_augmented_candidate_union",
    ])][["setting_id", "seed", "n_models", "width", "method", "source_method", "accuracy", "val_accuracy", "candidate_count", "selected_candidate_count", "central_lift_activated", "nonabelian_branch_lift_activated"]].copy()
    choices["selector_regret_vs_best_test_candidate_audit_only"] = [
        float(best_test.get(setting_id, np.nan) - accuracy)
        for setting_id, accuracy in zip(choices.setting_id, choices.accuracy)
    ]
    capacity = selected[["method", "actual_parameter_count", "parameter_multiplier", "branch_count", "inference_multiplier", "validation_budget"]].drop_duplicates().sort_values("method")

    selected.to_csv(OUT / "matched_selector_runs.csv", index=False)
    summary.to_csv(OUT / "matched_selector_summary.csv", index=False)
    stats.to_csv(OUT / "matched_selector_paired_stats.csv", index=False)
    choices.to_csv(OUT / "matched_selector_choices.csv", index=False)
    capacity.to_csv(OUT / "matched_selector_capacity.csv", index=False)
    blocker_df.to_csv(OUT / "matched_selector_blockers.csv", index=False)
    tex_main = summary[["method", "n_settings", "mean_test_accuracy", "mean_candidate_count", "coverage_status"]]
    tex_choices = choices.groupby("method", as_index=False).agg(n_settings=("setting_id", "nunique"), mean_regret=("selector_regret_vs_best_test_candidate_audit_only", "mean"), central_lift_rate=("central_lift_activated", "mean"), nonabelian_lift_rate=("nonabelian_branch_lift_activated", "mean"))
    for frame, path in ((tex_main, OUT / "tables" / "matched_selector_main.tex"), (tex_choices, OUT / "tables" / "matched_selector_choices.tex")):
        cols = list(frame.columns)
        lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule", " & ".join(cols) + "\\\\", "\\midrule"]
        for row in frame.to_dict("records"):
            lines.append(" & ".join((f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]).replace("_", "\\_")) for c in cols) + "\\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = summary.sort_values("mean_test_accuracy")
    ax.barh(ordered.method, ordered.mean_test_accuracy)
    ax.set_xlabel("mean executed test accuracy")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "matched_selector_accuracy.pdf")
    plt.close(fig)

    selector_row = stats[stats.comparison.str.startswith("improved_twistedmerge")].iloc[0]
    selector_supported = bool(selector_row.ci_low > 0)
    report = f"""# Matched Candidate-Budget and Selector Ablation Report

Primary practical-selector decision: **{'supported' if selector_supported else 'unsupported'}** versus ordinary greedy soup under the tracked executed grid.

## Exact command

```bash
{args.command_string}
```

- Git commit at aggregation: `{git_commit()}`
- Primary grid: MNIST, one-hidden-layer ReLU MLP, `n_models=3,4`, widths `16,32,64`, seeds `1800:1819`.
- All 120 primary settings are present in the tracked executed-model source.
- This run is a clean aggregation and matched-selection audit of those executed Torch rows; it is **not fresh inference from every checkpoint on the current commit**. That limitation prevents these numbers from entering the clean release manifest until a full checkpoint rerun is made.
- Test accuracy is used only for final evaluation and the explicitly labeled regret audit, never candidate selection.

## Main summary

{md(summary, ['method', 'n_settings', 'n_seeds', 'mean_test_accuracy', 'mean_validation_accuracy', 'mean_candidate_count', 'mean_selected_candidate_count', 'coverage_status', 'fresh_inference_on_current_commit'])}

## Paired statistics

{md(stats, ['comparison', 'n_pairs', 'paired_mean_accuracy_delta', 'ci_low', 'ci_high', 'wins', 'ties', 'losses'])}

## Candidate-selection audit

{md(tex_choices, list(tex_choices.columns))}

The practical selector selected **no central lift and no nonabelian branch lift**. Its available choices were exact-gauge or soup candidates. The obstruction-gated branch candidate was never activated because no setting supplied a valid certificate.

## Exact blockers

{md(blocker_df, ['method', 'evaluation_status', 'blocker'])}

`git_rebasin_style_pairwise` has only the 20-setting tracked external subset and is labeled partial coverage. The always-on branch control is not fabricated: no executed certified branch tensor exists for the 120-setting source grid. This is a benchmark limitation, not a positive result.
"""
    (OUT / "matched_selector_report.md").write_text(report, encoding="utf-8")
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "source": str(args.source.relative_to(ROOT)),
        "source_rows": len(source),
        "primary_settings": 120,
        "fresh_inference_on_current_commit": False,
        "test_used_for_selection": False,
    }
    (OUT / "matched_selector_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"primary selector vs greedy soup: {'supported' if selector_supported else 'unsupported'}")
    print(f"wrote {OUT / 'matched_selector_report.md'}")


if __name__ == "__main__":
    main()
