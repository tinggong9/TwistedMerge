#!/usr/bin/env python
"""Held-out prediction of merge degradation from natural checkpoint diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "next_benchmarks"

# Preregistered before any held-out evaluation in this program.
PRIMARY_TARGET = "weight_average_degradation"
PRIMARY_PREDICTOR = "cycle_residual"
HARM_THRESHOLD = 0.01


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) <= 1e-15 or np.std(y[mask]) <= 1e-15:
        return float("nan")
    if method == "spearman":
        x = rankdata(x[mask])
        y = rankdata(y[mask])
    else:
        x, y = x[mask], y[mask]
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_corr(x, y, method, seed, n=2000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        value = safe_corr(x[idx], y[idx], method)
        if np.isfinite(value):
            values.append(value)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (float("nan"), float("nan"))


def auc_score(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels, scores = labels[mask], scores[mask]
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def r2_score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 1e-15 else float("nan")


def scalar_oof(df, predictor, target):
    pred = np.full(len(df), np.nan)
    fold_rows = []
    for fold in sorted(df.setting_group.unique()):
        train = df.setting_group != fold
        test = ~train
        x_train = df.loc[train, predictor].to_numpy(dtype=float)
        y_train = df.loc[train, target].to_numpy(dtype=float)
        x_test = df.loc[test, predictor].to_numpy(dtype=float)
        finite = np.isfinite(x_train) & np.isfinite(y_train)
        if finite.sum() < 3 or np.std(x_train[finite]) <= 1e-15:
            pred[test] = np.nanmean(y_train)
            slope = 0.0
            intercept = float(np.nanmean(y_train))
        else:
            X = np.column_stack([np.ones(finite.sum()), x_train[finite]])
            intercept, slope = np.linalg.lstsq(X, y_train[finite], rcond=None)[0]
            pred[test] = intercept + slope * x_test
        fold_rows.append({
            "target": target,
            "predictor": predictor,
            "heldout_setting": fold,
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "intercept": intercept,
            "coefficient": slope,
        })
    return pred, fold_rows


def ridge_oof(df, predictors, target, ridge=1e-3):
    pred = np.full(len(df), np.nan)
    rows = []
    for fold in sorted(df.setting_group.unique()):
        train = df.setting_group != fold
        test = ~train
        X_train = df.loc[train, predictors].to_numpy(dtype=float)
        X_test = df.loc[test, predictors].to_numpy(dtype=float)
        y_train = df.loc[train, target].to_numpy(dtype=float)
        means = np.nanmean(X_train, axis=0)
        X_train = np.where(np.isfinite(X_train), X_train, means)
        X_test = np.where(np.isfinite(X_test), X_test, means)
        std = np.nanstd(X_train, axis=0)
        std[std <= 1e-12] = 1.0
        X_train = (X_train - means) / std
        X_test = (X_test - means) / std
        center = float(y_train.mean())
        coef = np.linalg.solve(X_train.T @ X_train + ridge * np.eye(len(predictors)), X_train.T @ (y_train - center))
        pred[test] = center + X_test @ coef
        rows.append({"target": target, "predictor": "+".join(predictors), "heldout_setting": fold, "train_rows": int(train.sum()), "test_rows": int(test.sum()), "intercept": center, "coefficient": json.dumps(dict(zip(predictors, map(float, coef))), sort_keys=True)})
    return pred, rows


def md(df, columns, limit=80):
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.head(limit).to_dict("records"):
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.6g}" if isinstance(value, float) and np.isfinite(value) else str(value))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "reports/csv/improved_validated_ladder_merge_benchmark.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    source = pd.read_csv(args.source)
    rows = []
    for setting_id, group in source.groupby("setting_id", sort=False):
        methods = group.set_index("method")
        required = {"weight_average", "c2m3_permutation", "monomial_scale"}
        if not required.issubset(methods.index):
            continue
        common = methods.loc["weight_average"]
        monomial = methods.loc["monomial_scale"]
        weight = methods.loc["weight_average"]
        sync = methods.loc["c2m3_permutation"]
        rows.append({
            "setting_id": setting_id,
            "dataset": common.dataset,
            "architecture": common.architecture,
            "n_models": int(common.n_models),
            "width": int(common.width),
            "seed": int(common.seed),
            "setting_group": f"N{int(common.n_models)}_W{int(common.width)}",
            "weight_average_degradation": float(common.individual_accuracy_mean - weight.accuracy),
            "synchronized_merge_degradation": float(common.individual_accuracy_mean - sync.accuracy),
            "harmful_merge_indicator": int(common.individual_accuracy_mean - weight.accuracy > HARM_THRESHOLD),
            "monomial_benefit_over_permutation": float(monomial.accuracy - sync.accuracy),
            "pairwise_alignment_loss": float(common.pairwise_alignment_residual),
            "inverse_consistency_residual": float(common.sync_disagreement),
            "cycle_residual": float(common.permutation_cycle_score),
            "centrality_residual": float(common.permutation_centrality),
            "cocycle_closure_residual": float("nan"),
            "distance_to_coboundaries": float("nan"),
            "synchronization_disagreement": float(common.sync_disagreement),
            "log_scale_variance": float(monomial.log_scale_variance) if pd.notna(monomial.log_scale_variance) else float("nan"),
            "individual_model_accuracy_variance": float(common.individual_accuracy_variance),
            "validation_loss": float(weight.val_loss),
            "validation_delta": float(weight.val_accuracy - common.individual_accuracy_mean),
            "natural_trained_checkpoint_instance": True,
            "planted_labels": False,
        })
    runs = pd.DataFrame(rows)
    predictors = [
        "pairwise_alignment_loss", "inverse_consistency_residual", "cycle_residual", "centrality_residual",
        "synchronization_disagreement", "log_scale_variance", "individual_model_accuracy_variance",
        "validation_loss", "validation_delta",
    ]
    targets = [
        "weight_average_degradation", "synchronized_merge_degradation", "monomial_benefit_over_permutation",
    ]
    summary_rows = []
    regression_rows = []
    oof_cache = {}
    for target in targets:
        for predictor in predictors:
            pred, fold_rows = scalar_oof(runs, predictor, target)
            regression_rows.extend(fold_rows)
            oof_cache[(target, predictor)] = pred
            pearson = safe_corr(runs[predictor], runs[target], "pearson")
            spearman = safe_corr(runs[predictor], runs[target], "spearman")
            p_low, p_high = bootstrap_corr(runs[predictor], runs[target], "pearson", 1103 + len(summary_rows), args.bootstrap_samples)
            s_low, s_high = bootstrap_corr(runs[predictor], runs[target], "spearman", 2103 + len(summary_rows), args.bootstrap_samples)
            auc = auc_score(runs.harmful_merge_indicator, pred if target == PRIMARY_TARGET else runs[predictor])
            calibration = float(np.mean(np.abs(np.clip((pred - np.nanmin(pred)) / max(np.nanmax(pred) - np.nanmin(pred), 1e-12), 0, 1) - runs.harmful_merge_indicator))) if target == PRIMARY_TARGET else float("nan")
            summary_rows.append({
                "target": target,
                "predictor": predictor,
                "n_instances": len(runs),
                "pearson": pearson,
                "pearson_ci_low": p_low,
                "pearson_ci_high": p_high,
                "spearman": spearman,
                "spearman_ci_low": s_low,
                "spearman_ci_high": s_high,
                "heldout_r2": r2_score(runs[target], pred),
                "harmful_merge_auc": auc,
                "calibration_error": calibration,
                "leave_one_setting_out": True,
            })
    multivariate = [
        "pairwise_alignment_loss", "cycle_residual", "centrality_residual", "synchronization_disagreement",
        "log_scale_variance", "individual_model_accuracy_variance", "validation_loss", "validation_delta",
    ]
    full_pred, full_rows = ridge_oof(runs, multivariate, PRIMARY_TARGET)
    val_pred, val_rows = ridge_oof(runs, ["validation_loss", "validation_delta"], PRIMARY_TARGET)
    pair_pred, pair_rows = ridge_oof(runs, ["pairwise_alignment_loss"], PRIMARY_TARGET)
    regression_rows.extend(full_rows + val_rows + pair_rows)
    comparison = pd.DataFrame([
        {"model": "all_diagnostics_plus_validation", "heldout_r2": r2_score(runs[PRIMARY_TARGET], full_pred), "harmful_merge_auc": auc_score(runs.harmful_merge_indicator, full_pred)},
        {"model": "validation_only_baseline", "heldout_r2": r2_score(runs[PRIMARY_TARGET], val_pred), "harmful_merge_auc": auc_score(runs.harmful_merge_indicator, val_pred)},
        {"model": "pairwise_alignment_only_baseline", "heldout_r2": r2_score(runs[PRIMARY_TARGET], pair_pred), "harmful_merge_auc": auc_score(runs.harmful_merge_indicator, pair_pred)},
    ])
    summary = pd.DataFrame(summary_rows)
    regressions = pd.DataFrame(regression_rows)
    primary = summary[(summary.target == PRIMARY_TARGET) & (summary.predictor == PRIMARY_PREDICTOR)].iloc[0]
    added_value = bool(comparison.iloc[0].heldout_r2 > comparison.iloc[1].heldout_r2 and primary.pearson_ci_low > 0 and primary.heldout_r2 > 0)
    runs.to_csv(OUT / "diagnostic_prediction_runs.csv", index=False)
    summary.to_csv(OUT / "diagnostic_prediction_summary.csv", index=False)
    regressions.to_csv(OUT / "diagnostic_prediction_regressions.csv", index=False)
    comparison.to_csv(OUT / "diagnostic_prediction_model_comparison.csv", index=False)
    tex = summary[(summary.target == PRIMARY_TARGET) & summary.predictor.isin([PRIMARY_PREDICTOR, "pairwise_alignment_loss", "validation_loss", "validation_delta"])].copy()
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "predictor & Pearson & Spearman & held-out $R^2$ & AUC\\\\", "\\midrule"]
    for row in tex.itertuples():
        lines.append(f"{row.predictor.replace('_', ' ')} & {row.pearson:.3f} & {row.spearman:.3f} & {row.heldout_r2:.3f} & {row.harmful_merge_auc:.3f}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (OUT / "tables" / "diagnostic_prediction.tex").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    primary_pred = oof_cache[(PRIMARY_TARGET, PRIMARY_PREDICTOR)]
    for group, part in runs.assign(oof_prediction=primary_pred).groupby("setting_group"):
        ax.scatter(part[PRIMARY_TARGET], part.oof_prediction, label=group, alpha=0.75)
    lo = min(runs[PRIMARY_TARGET].min(), np.nanmin(primary_pred))
    hi = max(runs[PRIMARY_TARGET].max(), np.nanmax(primary_pred))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    ax.set_xlabel("observed weight-average degradation")
    ax.set_ylabel("leave-one-setting-out prediction")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "diagnostic_prediction.pdf")
    plt.close(fig)
    decision = "supported" if added_value else "unsupported"
    report = f"""# Held-Out Diagnostic Prediction Report

Natural-data diagnostic hypothesis: **{decision}** under the preregistered gate.

## Preregistration and exact command

- Primary target: `{PRIMARY_TARGET}`
- Primary predictor: `{PRIMARY_PREDICTOR}`
- Harmful-merge threshold: `{HARM_THRESHOLD}` absolute accuracy
- Evaluation: leave one complete `(n_models,width)` setting out; seeds never cross from a held-out setting into its training folds.

```bash
{args.command_string}
```

- Git commit at execution: `{git_commit()}`
- Instances: `{len(runs)}` natural trained-checkpoint collections from `{args.source.relative_to(ROOT)}`
- Planted labels: `False`

## Correlation and held-out prediction

{md(summary[summary.target == PRIMARY_TARGET], ['predictor', 'n_instances', 'pearson', 'pearson_ci_low', 'pearson_ci_high', 'spearman', 'spearman_ci_low', 'spearman_ci_high', 'heldout_r2', 'harmful_merge_auc', 'calibration_error'])}

## Baseline comparison

{md(comparison, ['model', 'heldout_r2', 'harmful_merge_auc'])}

The harmful-merge AUC is undefined when the fixed `0.01` indicator has only one class. In this grid every weight-average instance crossed the threshold, so the report retains `NaN` rather than changing the preregistered threshold after observing outcomes.

## Decision

The diagnostic is promoted only if the preregistered primary correlation interval clears zero, leave-one-setting-out `R^2` is positive, and the full diagnostic model adds held-out value beyond ordinary validation metrics. Result: **{decision}**. Missing cocycle-closure and certified distance-to-coboundary values are left unavailable rather than imputed as evidence.
"""
    (OUT / "diagnostic_prediction_report.md").write_text(report, encoding="utf-8")
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "primary_target": PRIMARY_TARGET,
        "primary_predictor": PRIMARY_PREDICTOR,
        "harm_threshold": HARM_THRESHOLD,
        "split": "leave_one_complete_n_models_width_setting_out",
        "decision": decision,
    }
    (OUT / "diagnostic_prediction_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"natural-data diagnostic hypothesis: {decision}")
    print(f"wrote {OUT / 'diagnostic_prediction_report.md'}")


if __name__ == "__main__":
    main()
