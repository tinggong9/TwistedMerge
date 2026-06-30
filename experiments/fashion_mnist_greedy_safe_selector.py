#!/usr/bin/env python
"""Greedy-safe selector analysis for the Fashion-MNIST MLP ladder rows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.greedy_safe_selector import (  # noqa: E402
    DEFAULT_GREEDY_SAFE_POOL,
    nested_validation_selector,
    regret_bound_selector,
    tau_bootstrap_selector,
    tau_fixed_selector,
    tau_loss_aware_selector,
)
from src.metrics import capture_environment  # noqa: E402


BASELINES = ("greedy_soup", "c2m3_permutation", "weight_average")
BASE_METHODS = (
    "weight_average",
    "c2m3_permutation",
    "monomial_scale",
    "shrinkage_monomial_scale",
    "global_monomial_scale",
    "optimized_monomial_scale",
    "greedy_soup",
    "c2m3_greedy_soup",
    "monomial_scaled_greedy_soup",
    "shrinkage_monomial_greedy_soup",
    "global_monomial_greedy_soup",
    "optimized_monomial_greedy_soup",
    "union_candidate_soup",
    "improved_validated_selector",
)
INT_COLUMNS = {
    "n_rows",
    "n_settings",
    "n_pairs",
    "accuracy_wins",
    "accuracy_ties",
    "accuracy_losses",
    "left_greedy_count",
    "beneficial_challenger_count",
    "false_challenger_count",
    "tie_count",
}


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | str:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return "unknown"


def bootstrap_mean_ci(values, n_bootstrap: int, seed: int) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_two_sided(wins: int, losses: int) -> float:
    import math

    n = wins + losses
    if n <= 0:
        return float("nan")
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def setting_metrics(group: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = {}
    for method, item in group.groupby("method", dropna=False):
        row = item.iloc[0]
        metrics[str(method)] = {
            "accuracy": float(row["val_accuracy"]),
            "loss": float(row["val_loss"]),
        }
    return metrics


def selected_row(group: pd.DataFrame, method: str) -> pd.Series:
    rows = group[group["method"] == method]
    if rows.empty:
        raise KeyError(f"method {method!r} missing from setting {group['setting_id'].iloc[0]}")
    return rows.iloc[0]


def make_selector_row(group: pd.DataFrame, choice, selector_mode: str, extra: dict) -> dict:
    selected = selected_row(group, choice.selected)
    greedy = selected_row(group, "greedy_soup")
    c2m3 = selected_row(group, "c2m3_permutation")
    weight = selected_row(group, "weight_average")
    row = selected.to_dict()
    row.update(
        {
            "method": "greedy_safe_selector",
            "source_method": choice.selected,
            "selector_mode": selector_mode,
            "selector_challenger": choice.challenger,
            "selector_baseline": choice.baseline,
            "selector_tau_accuracy": choice.tau_accuracy,
            "selector_tau_loss": choice.tau_loss,
            "selector_confidence": choice.confidence,
            "selector_lcb": choice.lower_confidence_bound,
            "selector_predicted_regret_bound": choice.predicted_regret_bound,
            "selector_val_margin": choice.validation_accuracy_delta,
            "selector_val_loss_delta": choice.validation_loss_delta,
            "selector_left_greedy": choice.selected != "greedy_soup",
            "selector_no_test_leakage": not choice.used_test_metrics,
            "symmetry_status": "validation_selected_greedy_safe_single_model_or_soup",
            "method_notes": "Greedy soup fallback; challenger accepted only by validation-only greedy-safe gate.",
            "selection_used_validation_only": not choice.used_test_metrics,
            "validation_protocol": "aggregate_replay_from_5m",
            "nested_sample_disjoint": False,
        }
    )
    row.update(extra)
    row["accuracy_delta_vs_greedy_soup"] = float(row["accuracy"]) - float(greedy["accuracy"])
    row["loss_delta_vs_greedy_soup"] = float(row["loss"]) - float(greedy["loss"])
    row["accuracy_delta_vs_c2m3"] = float(row["accuracy"]) - float(c2m3["accuracy"])
    row["loss_delta_vs_c2m3"] = float(row["loss"]) - float(c2m3["loss"])
    row["accuracy_delta_vs_weight_average"] = float(row["accuracy"]) - float(weight["accuracy"])
    row["loss_delta_vs_weight_average"] = float(row["loss"]) - float(weight["loss"])
    row["regret_vs_greedy_soup"] = max(0.0, float(greedy["accuracy"]) - float(row["accuracy"]))
    return row


def add_greedy_safe_rows(df: pd.DataFrame, args) -> pd.DataFrame:
    tau_accuracy_grid = parse_csv(args.tau_accuracy_grid, float)
    tau_loss_grid = parse_csv(args.tau_loss_grid, float)
    confidence_grid = parse_csv(args.bootstrap_confidence_grid, float)
    regret_grid = parse_csv(args.regret_threshold_grid, float)
    rows = [row for row in df.to_dict("records") if row["method"] in BASE_METHODS]
    n_validation = int(round(float(df["max_train_samples"].dropna().iloc[0]) * float(df["val_fraction"].dropna().iloc[0])))
    pool = [name for name in DEFAULT_GREEDY_SAFE_POOL if name in set(df["method"])]
    for setting_id, group in df.groupby("setting_id", sort=True):
        metrics = setting_metrics(group)
        for tau in tau_accuracy_grid:
            choice = tau_fixed_selector(metrics, challenger_pool=pool, tau_accuracy=tau)
            rows.append(make_selector_row(group, choice, "tau_fixed", {"selector_grid_label": f"tau={tau:g}"}))
        for tau_acc in tau_accuracy_grid:
            for tau_loss in tau_loss_grid:
                choice = tau_loss_aware_selector(metrics, challenger_pool=pool, tau_accuracy=tau_acc, tau_loss=tau_loss)
                rows.append(
                    make_selector_row(
                        group,
                        choice,
                        "tau_loss_aware",
                        {"selector_grid_label": f"tau_acc={tau_acc:g};tau_loss={tau_loss:g}"},
                    )
                )
        for confidence in confidence_grid:
            choice = tau_bootstrap_selector(metrics, challenger_pool=pool, n_validation=n_validation, confidence=confidence)
            rows.append(make_selector_row(group, choice, "tau_bootstrap", {"selector_grid_label": f"confidence={confidence:g}"}))
        for tau_acc in tau_accuracy_grid:
            choice = nested_validation_selector(
                metrics,
                metrics,
                challenger_pool=pool,
                tau_accuracy=tau_acc,
                tau_loss=0.0,
            )
            rows.append(
                make_selector_row(
                    group,
                    choice,
                    "nested_validation",
                    {
                        "selector_grid_label": f"tau_acc={tau_acc:g}",
                        "nested_sample_disjoint": False,
                        "validation_protocol": "aggregate_replay_nested_proxy",
                    },
                )
            )
        for threshold in regret_grid:
            choice = regret_bound_selector(
                metrics,
                challenger_pool=pool,
                regret_threshold=threshold,
                confidence=args.regret_confidence,
                n_validation=n_validation,
            )
            rows.append(
                make_selector_row(
                    group,
                    choice,
                    "regret_bound",
                    {"selector_grid_label": f"regret={threshold:g};confidence={args.regret_confidence:g}"},
                )
            )
    out = pd.DataFrame(rows)
    for column in ["selector_left_greedy", "selection_used_validation_only", "nested_sample_disjoint"]:
        if column in out.columns:
            out[column] = out[column].fillna(False).astype(bool)
    out["selection_used_validation_only"] = out["selection_used_validation_only"].where(
        out["method"] == "greedy_safe_selector",
        out.get("selector_no_test_leakage", True),
    )
    return out


def method_summary(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "selector_mode", "selector_grid_label"]
    for keys, group in df.groupby(group_cols, dropna=False):
        method, selector_mode, grid_label = keys
        deltas = pd.to_numeric(group["accuracy_delta_vs_greedy_soup"], errors="coerce")
        ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap, seed=9300 + len(rows))
        left = group.get("selector_left_greedy", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        wins = deltas > 0
        ties = deltas == 0
        losses = deltas < 0
        rows.append(
            {
                "summary_type": "method_summary",
                "method": method,
                "selector_mode": "" if pd.isna(selector_mode) else selector_mode,
                "selector_grid_label": "" if pd.isna(grid_label) else grid_label,
                "n_rows": int(len(group)),
                "n_settings": int(group["setting_id"].nunique()),
                "mean_val_accuracy": float(pd.to_numeric(group["val_accuracy"], errors="coerce").mean()),
                "mean_val_loss": float(pd.to_numeric(group["val_loss"], errors="coerce").mean()),
                "mean_test_accuracy": float(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
                "mean_delta_vs_greedy_soup": float(deltas.mean()),
                "delta_vs_greedy_ci_low": ci_low,
                "delta_vs_greedy_ci_high": ci_high,
                "mean_delta_vs_c2m3": float(pd.to_numeric(group["accuracy_delta_vs_c2m3"], errors="coerce").mean()),
                "mean_delta_vs_weight_average": float(pd.to_numeric(group["accuracy_delta_vs_weight_average"], errors="coerce").mean()),
                "mean_regret_vs_greedy_soup": float(pd.to_numeric(group.get("regret_vs_greedy_soup", np.nan), errors="coerce").mean()),
                "left_greedy_count": int(left.sum()),
                "left_greedy_rate": float(left.mean()),
                "beneficial_challenger_count": int((left & wins).sum()),
                "beneficial_challenger_rate": float((left & wins).mean()),
                "false_challenger_count": int((left & losses).sum()),
                "false_challenger_rate": float((left & losses).mean()),
                "tie_count": int(ties.sum()),
                "tie_rate": float(ties.mean()),
                "accuracy_wins": int(wins.sum()),
                "accuracy_ties": int(ties.sum()),
                "accuracy_losses": int(losses.sum()),
                "selection_used_validation_only": bool(group.get("selection_used_validation_only", True).fillna(True).astype(bool).all()),
            }
        )
    return pd.DataFrame(rows)


def paired_summary(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    fixed = df.pivot_table(
        index=["setting_id", "method", "selector_mode", "selector_grid_label"],
        values=["accuracy", "loss"],
        aggfunc="first",
    ).reset_index()
    baseline = df[df["method"].isin(BASELINES)].pivot_table(index="setting_id", columns="method", values=["accuracy", "loss"], aggfunc="first")
    baseline.columns = [f"{metric}__{method}" for metric, method in baseline.columns]
    baseline = baseline.reset_index()
    merged = fixed.merge(baseline, on="setting_id", how="left")
    for baseline_name in BASELINES:
        clean = merged.dropna(subset=[f"accuracy__{baseline_name}"])
        delta = clean["accuracy"] - clean[f"accuracy__{baseline_name}"]
        loss_delta = clean["loss"] - clean[f"loss__{baseline_name}"]
        for keys, group in clean.groupby(["method", "selector_mode", "selector_grid_label"], dropna=False):
            method, mode, label = keys
            d = group["accuracy"] - group[f"accuracy__{baseline_name}"]
            l = group["loss"] - group[f"loss__{baseline_name}"]
            wins = int((d > 0).sum())
            ties = int((d == 0).sum())
            losses = int((d < 0).sum())
            ci_low, ci_high = bootstrap_mean_ci(d, n_bootstrap, seed=10200 + len(rows))
            rows.append(
                {
                    "summary_type": "paired_comparison",
                    "comparison": f"{method}_vs_{baseline_name}",
                    "method": method,
                    "selector_mode": "" if pd.isna(mode) else mode,
                    "selector_grid_label": "" if pd.isna(label) else label,
                    "baseline": baseline_name,
                    "n_pairs": int(len(group)),
                    "paired_mean_accuracy_delta": float(d.mean()),
                    "paired_accuracy_delta_ci_low": ci_low,
                    "paired_accuracy_delta_ci_high": ci_high,
                    "paired_mean_loss_delta": float(l.mean()),
                    "accuracy_wins": wins,
                    "accuracy_ties": ties,
                    "accuracy_losses": losses,
                    "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                }
            )
        _ = delta, loss_delta
    return pd.DataFrame(rows)


def selector_choice_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sel = df[df["method"] == "greedy_safe_selector"].copy()
    for keys, group in sel.groupby(["selector_mode", "selector_grid_label"], dropna=False):
        mode, label = keys
        choices = group["source_method"].value_counts(dropna=False).to_dict()
        challengers = group["selector_challenger"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "summary_type": "selector_choice_counts",
                "method": "greedy_safe_selector",
                "selector_mode": mode,
                "selector_grid_label": label,
                "n_rows": int(len(group)),
                "selector_choice_counts": json.dumps({str(k): int(v) for k, v in choices.items()}),
                "selector_challenger_counts": json.dumps({str(k): int(v) for k, v in challengers.items()}),
            }
        )
    return pd.DataFrame(rows)


def claim_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = summary[summary["summary_type"] == "method_summary"].copy()
    selectors = methods[methods["method"] == "greedy_safe_selector"].copy()
    if selectors.empty:
        return pd.DataFrame()
    safe = selectors.sort_values(
        ["false_challenger_rate", "mean_regret_vs_greedy_soup", "left_greedy_rate"],
        ascending=[True, True, False],
    ).iloc[0]
    best_delta = selectors.sort_values("mean_delta_vs_greedy_soup", ascending=False).iloc[0]
    low_false = float(safe["false_challenger_rate"]) == 0.0
    matched_greedy = float(safe["mean_delta_vs_greedy_soup"]) >= -1e-12
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "greedy_safe_selector_avoids_harmful_departures",
            "claim_decision": "Supported limited" if low_false else "Supported negative result",
            "claim_reason": f"best safety row {safe['selector_mode']} {safe['selector_grid_label']} false challenger rate={safe['false_challenger_rate']:.4f}",
        }
    )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "greedy_safe_selector_matches_greedy_while_preserving_c2m3_gain",
            "claim_decision": "Supported limited" if matched_greedy and float(safe["mean_delta_vs_c2m3"]) > 0 else "Supported descriptive",
            "claim_reason": f"safe row delta vs greedy={safe['mean_delta_vs_greedy_soup']:.6f}, delta vs C2M3={safe['mean_delta_vs_c2m3']:.6f}",
        }
    )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "greedy_safe_selector_beats_greedy_soup_overall",
            "claim_decision": "Supported limited" if float(best_delta["delta_vs_greedy_ci_low"]) > 0 else "Not yet supported",
            "claim_reason": f"best selector mean delta vs greedy={best_delta['mean_delta_vs_greedy_soup']:.6f}, CI=[{best_delta['delta_vs_greedy_ci_low']:.6f},{best_delta['delta_vs_greedy_ci_high']:.6f}]",
        }
    )
    rows.append(
        {
            "summary_type": "claim_decision",
            "claim": "external_model_soups_win",
            "claim_decision": "Not yet supported",
            "claim_reason": "this replay uses the in-repo Fashion-MNIST greedy soup baseline, not official external Model Soups code",
        }
    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    parts = [
        method_summary(df, n_bootstrap),
        paired_summary(df, n_bootstrap),
        selector_choice_summary(df),
    ]
    first = pd.concat(parts, ignore_index=True, sort=False)
    claims = claim_rows(first)
    return pd.concat([first, claims], ignore_index=True, sort=False)


def format_value(value, column: str) -> str:
    if value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if column in INT_COLUMNS:
        return str(int(round(float(value))))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    rows = []
    view = df.copy()
    for column in columns:
        if column not in view:
            view[column] = ""
    rows.extend(view[columns].head(max_rows).to_dict("records"))
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(col, ""), col) for col in columns) + " |")
    return "\n".join(lines)


def write_plots(df: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    sel = df[df["method"] == "greedy_safe_selector"].copy()
    plt.figure(figsize=(7.2, 4.2))
    for mode, group in sel.groupby("selector_mode"):
        plt.scatter(group["selector_val_margin"], group["accuracy_delta_vs_greedy_soup"], s=16, alpha=0.55, label=mode)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("validation accuracy margin of challenger vs greedy")
    plt.ylabel("test accuracy delta vs greedy soup")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir / "fashion_greedy_safe_delta_vs_greedy_soup.pdf")
    plt.close()

    methods = summary[(summary["summary_type"] == "method_summary") & (summary["method"] == "greedy_safe_selector")].copy()
    top = methods.sort_values(["false_challenger_rate", "mean_regret_vs_greedy_soup"]).head(18)
    labels = top["selector_mode"].astype(str) + "\n" + top["selector_grid_label"].astype(str)
    plt.figure(figsize=(8.2, 4.4))
    plt.bar(np.arange(len(top)), top["mean_regret_vs_greedy_soup"].astype(float))
    plt.xticks(np.arange(len(top)), labels, rotation=45, ha="right", fontsize=6)
    plt.ylabel("mean regret vs greedy soup")
    plt.tight_layout()
    plt.savefig(plot_dir / "fashion_greedy_safe_regret.pdf")
    plt.close()


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = summary[(summary["summary_type"] == "method_summary") & (summary["method"] == "greedy_safe_selector")].copy()
    rows = rows.sort_values(["false_challenger_rate", "mean_regret_vs_greedy_soup", "selector_mode"]).head(12)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Mode & Grid & $\\Delta$ greedy & regret & false & beneficial \\\\",
        "\\midrule",
    ]
    for _idx, row in rows.iterrows():
        lines.append(
            f"{str(row['selector_mode']).replace('_', '\\_')} & "
            f"{str(row['selector_grid_label']).replace('_', '\\_')} & "
            f"{float(row['mean_delta_vs_greedy_soup']):+.5f} & "
            f"{float(row['mean_regret_vs_greedy_soup']):.5f} & "
            f"{float(row['false_challenger_rate']):.3f} & "
            f"{float(row['beneficial_challenger_rate']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    method_rows = summary[summary["summary_type"] == "method_summary"].copy()
    selector_rows = method_rows[method_rows["method"] == "greedy_safe_selector"].copy()
    safe_rows = selector_rows.sort_values(["false_challenger_rate", "mean_regret_vs_greedy_soup", "left_greedy_rate"]).head(15)
    best_rows = selector_rows.sort_values("mean_delta_vs_greedy_soup", ascending=False).head(15)
    paired = summary[
        (summary["summary_type"] == "paired_comparison")
        & (summary["method"] == "greedy_safe_selector")
        & (summary["baseline"].isin(["greedy_soup", "c2m3_permutation", "weight_average"]))
    ].copy()
    claims = summary[summary["summary_type"] == "claim_decision"].copy()
    choice_counts = summary[summary["summary_type"] == "selector_choice_counts"].copy().head(30)
    settings = df[df["method"] == "greedy_soup"].drop_duplicates("setting_id")
    report = f"""# Fashion-MNIST Greedy-Safe Selector Report

This report is generated by `experiments/fashion_mnist_greedy_safe_selector.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- Base candidate table: `{args.base_csv}`
- Dataset: Fashion-MNIST, one-hidden-layer ReLU MLP
- Settings: `{settings.shape[0]}` total; N=3,4 and widths 32,64,128 inherited from 5(m)
- Main setting N=4,width=64 has 10 seeds; secondary settings have 5 seeds
- Test split: inherited from 5(m), `max_test_samples=5000`; the full Fashion-MNIST test set was not rerun in this selector replay
- Selection data: validation metrics only; test metrics are read after selection for reporting
- Nested-validation rows are aggregate replay proxies because the 5(m) artifact does not contain sample-level selector/accept predictions. The source selector API supports disjoint selector and accept metrics, and the unit tests cover that behavior.

## Git State

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_dirty()}`

## Greedy-Safe Rows With Lowest False-Challenger Rate

{md_table(safe_rows, ["selector_mode", "selector_grid_label", "n_rows", "mean_test_accuracy", "mean_delta_vs_greedy_soup", "mean_delta_vs_c2m3", "mean_regret_vs_greedy_soup", "left_greedy_rate", "false_challenger_rate", "beneficial_challenger_rate", "tie_rate", "selection_used_validation_only"], 20)}

## Greedy-Safe Rows With Best Mean Delta Versus Greedy

{md_table(best_rows, ["selector_mode", "selector_grid_label", "n_rows", "mean_test_accuracy", "mean_delta_vs_greedy_soup", "delta_vs_greedy_ci_low", "delta_vs_greedy_ci_high", "mean_regret_vs_greedy_soup", "false_challenger_rate", "beneficial_challenger_rate"], 20)}

## Paired Selector Comparisons

{md_table(paired, ["selector_mode", "selector_grid_label", "baseline", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "accuracy_wins", "accuracy_ties", "accuracy_losses", "sign_test_two_sided_p"], 60)}

## Selector Choice Counts

{md_table(choice_counts, ["selector_mode", "selector_grid_label", "n_rows", "selector_choice_counts", "selector_challenger_counts"], 30)}

## Claim Decisions

{md_table(claims, ["claim", "claim_decision", "claim_reason"], 20)}

## Negative Boundaries

- No claim is made that TwistedMerge++ beats greedy soup overall.
- No external Model Soups, external C2M3, or SOTA comparison is made.
- This selector replay inherits the 5(m) candidate metrics; it does not retrain the MLP models or rerun the full Fashion-MNIST test set.
- Nested-validation rows are validation-only aggregate proxies here, not sample-disjoint nested model evaluations.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_config(args, path: Path) -> None:
    config = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "dirty_worktree": git_dirty(),
        "base_csv": str(args.base_csv),
        "tau_accuracy_grid": args.tau_accuracy_grid,
        "tau_loss_grid": args.tau_loss_grid,
        "bootstrap_confidence_grid": args.bootstrap_confidence_grid,
        "regret_threshold_grid": args.regret_threshold_grid,
        "regret_confidence": args.regret_confidence,
        "environment": capture_environment(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=ROOT / "reports/csv/fashion_mnist_improved_ladder.csv")
    parser.add_argument("--tau-accuracy-grid", default="0,0.0005,0.001,0.002,0.005")
    parser.add_argument("--tau-loss-grid", default="0,0.001,0.002,0.005")
    parser.add_argument("--bootstrap-confidence-grid", default="0.80,0.90,0.95")
    parser.add_argument("--regret-threshold-grid", default="0,0.0005,0.001,0.002")
    parser.add_argument("--regret-confidence", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    base = pd.read_csv(args.base_csv)
    df = add_greedy_safe_rows(base, args)
    summary = summarize(df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    config_dir = args.reports_dir / "configs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "fashion_mnist_greedy_safe_selector.csv"
    summary_path = csv_dir / "fashion_mnist_greedy_safe_selector_summary.csv"
    table_path = table_dir / "fashion_greedy_safe_selector_table.tex"
    report_path = args.reports_dir / "fashion_mnist_greedy_safe_selector_report.md"
    config_path = config_dir / "fashion_mnist_greedy_safe_selector_config.json"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_plots(df, summary, plot_dir)
    write_latex_table(summary, table_path)
    write_report(args, df, summary, report_path)
    write_config(args, config_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {table_path}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
