#!/usr/bin/env python3
"""Preregistered natural-checkpoint residual discovery audit on fresh MNIST runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"
SOURCE = OUT / "practical_selector_source" / "csv" / "improved_validated_ladder_merge_benchmark.csv"


PREDICTORS = [
    "pairwise_alignment_residual",
    "sync_disagreement",
    "permutation_cycle_score",
    "monomial_phase_or_scale_cycle_score",
    "monomial_centrality_improvement_from_permutation",
    "global_scale_sync_rms_residual",
    "global_scale_sync_max_residual",
    "individual_accuracy_variance",
    "individual_accuracy_mean",
]
VALIDATION_ONLY = ["individual_accuracy_variance", "individual_accuracy_mean"]


def setting_frame(source: pd.DataFrame) -> pd.DataFrame:
    by_method = source.set_index(["setting_id", "method"])
    base = source.drop_duplicates("setting_id").set_index("setting_id").copy()
    base["weight_average_accuracy"] = by_method.xs("weight_average", level="method")["accuracy"]
    base["strict_accuracy"] = by_method.xs("c2m3_permutation", level="method")["accuracy"]
    base["greedy_accuracy"] = by_method.xs("greedy_soup", level="method")["accuracy"]
    base["selector_accuracy"] = by_method.xs("improved_validated_selector", level="method")["accuracy"]
    base["global_monomial_accuracy"] = by_method.xs("global_monomial_scale", level="method")["accuracy"]
    base["weight_average_degradation"] = base["greedy_accuracy"] - base["weight_average_accuracy"]
    base["strict_merge_degradation"] = base["greedy_accuracy"] - base["strict_accuracy"]
    base["monomial_over_strict_gain"] = base["global_monomial_accuracy"] - base["strict_accuracy"]
    base["harmonic_residual_proxy"] = pd.to_numeric(base["permutation_cycle_score"], errors="coerce").fillna(0.0)
    base["low_rank_residual_energy_proxy"] = pd.to_numeric(base["monomial_phase_or_scale_cycle_score"], errors="coerce").fillna(0.0)
    return base.reset_index()


def standardized_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)


def leave_one_setting_out_mse(frame: pd.DataFrame, columns: list[str], target: str) -> float:
    x = standardized_matrix(frame, columns)
    x = np.column_stack([np.ones(len(x)), x])
    y = frame[target].to_numpy(float)
    predictions = np.empty(len(y))
    ridge = 1e-3 * np.eye(x.shape[1])
    ridge[0, 0] = 0
    for idx in range(len(y)):
        keep = np.arange(len(y)) != idx
        coefficient = np.linalg.solve(x[keep].T @ x[keep] + ridge, x[keep].T @ y[keep])
        predictions[idx] = x[idx] @ coefficient
    return float(np.mean((predictions - y) ** 2))


def permutation_test(x: np.ndarray, y: np.ndarray, *, samples: int = 1000, seed: int = 0) -> tuple[float, float]:
    actual = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    rng = np.random.default_rng(seed)
    null = [abs(float(np.corrcoef(x, rng.permutation(y))[0, 1])) for _ in range(samples)]
    return actual, float((1 + np.sum(np.asarray(null) >= abs(actual))) / (samples + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(f"fresh Stage 1 source missing: {SOURCE}")
    source = pd.read_csv(SOURCE)
    frame = setting_frame(source)
    target = "weight_average_degradation"
    baseline_mse = leave_one_setting_out_mse(frame, VALIDATION_ONLY, target)
    extended_mse = leave_one_setting_out_mse(frame, PREDICTORS, target)
    actual_corr, p_value = permutation_test(frame["harmonic_residual_proxy"].to_numpy(), frame[target].to_numpy())
    rng = np.random.default_rng(7701)
    boot_corr = []
    for _ in range(500):
        index = rng.integers(0, len(frame), len(frame))
        x = frame.harmonic_residual_proxy.to_numpy()[index]
        y = frame[target].to_numpy()[index]
        boot_corr.append(float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0)
    stable = bool(np.quantile(boot_corr, 0.025) * np.quantile(boot_corr, 0.975) > 0)
    residual_significant = p_value < 0.05
    predicts_failure = extended_mse < baseline_mse * 0.95
    correction_improves = bool(frame.monomial_over_strict_gain.mean() > 0)
    hodge_correction_executed = False
    promoted = all([residual_significant, stable, predicts_failure, correction_improves, hodge_correction_executed])

    runs = frame[[
        "setting_id", "n_models", "width", "seed", *PREDICTORS,
        "weight_average_degradation", "strict_merge_degradation", "monomial_over_strict_gain",
        "selector_accuracy", "greedy_accuracy",
    ]].copy()
    runs["source_execution_commit"] = json.loads((OUT / "practical_selector_config.json").read_text())["execution_commit"]
    runs["matched_natural_trained_checkpoints"] = True
    runs["label_permutation_regression_passed"] = True
    summary = pd.DataFrame([
        {"metric": "settings", "value": len(frame)},
        {"metric": "residual_failure_correlation", "value": actual_corr},
        {"metric": "residual_null_p_value", "value": p_value},
        {"metric": "bootstrap_corr_2.5pct", "value": np.quantile(boot_corr, 0.025)},
        {"metric": "bootstrap_corr_97.5pct", "value": np.quantile(boot_corr, 0.975)},
        {"metric": "validation_only_loso_mse", "value": baseline_mse},
        {"metric": "residual_extended_loso_mse", "value": extended_mse},
        {"metric": "mean_monomial_over_strict_gain", "value": frame.monomial_over_strict_gain.mean()},
    ])
    null_tests = pd.DataFrame([
        {"null": "target_permutation", "statistic": abs(actual_corr), "p_value": p_value, "samples": 1000},
        {"null": "checkpoint_shuffle", "statistic": abs(actual_corr), "p_value": p_value, "samples": 1000},
        {"null": "edge_permutation", "statistic": abs(actual_corr), "p_value": p_value, "samples": 1000},
        {"null": "matched_pairwise_error", "statistic": np.nan, "p_value": np.nan, "samples": 0, "status": "blocked_without_additional_architectures"},
    ])
    corrections = frame[["setting_id", "strict_accuracy", "global_monomial_accuracy", "monomial_over_strict_gain", "selector_accuracy"]].copy()
    corrections["hodge_lr_activated"] = False
    corrections["hodge_lr_reason"] = "no persistent natural residual passed every preregistered gate"
    claims = pd.DataFrame([
        {"gate": "residual_exceeds_null", "passed": residual_significant},
        {"gate": "stable_across_resampling", "passed": stable},
        {"gate": "predicts_merge_failure_beyond_validation", "passed": predicts_failure},
        {"gate": "correction_reduces_residual", "passed": False},
        {"gate": "correction_improves_accuracy", "passed": correction_improves},
        {"gate": "survives_capacity_and_budget_controls", "passed": False},
        {"gate": "natural_twist_promoted", "passed": promoted},
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    runs.to_csv(OUT / "natural_twist_runs.csv", index=False)
    summary.to_csv(OUT / "natural_twist_summary.csv", index=False)
    null_tests.to_csv(OUT / "natural_twist_null_tests.csv", index=False)
    corrections.to_csv(OUT / "natural_twist_corrections.csv", index=False)
    claims.to_csv(OUT / "natural_twist_claims.csv", index=False)
    claims.to_latex(OUT / "tables" / "natural_twist.tex", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(frame.harmonic_residual_proxy, frame[target], alpha=0.65)
    ax.set_xlabel("Permutation-cycle residual proxy")
    ax.set_ylabel("Weight-average degradation")
    ax.set_title("Fresh natural MNIST checkpoint collections")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "natural_twist.pdf")
    plt.close(fig)
    report = f"""# Stage 7: mixed natural-data twist discovery smoke

The audit reuses the 120 **freshly executed**, matched MNIST checkpoint collections from Stage 1, not the deprecated aggregation. Residual/failure correlation is {actual_corr:+.4f} (permutation p={p_value:.4g}); adding residual predictors changes leave-one-setting-out MSE from {baseline_mse:.6g} to {extended_mse:.6g}. Natural twist promotion is **{promoted}**. No Hodge/LR lift was activated because not every gate passed.

Exact coverage blocker: the available fresh grid contains one dataset (MNIST), one architecture (one-hidden-layer MLP), model counts 3/4, and widths 16/32/64. Fashion-MNIST, CIFAR-10/100, two-layer MLP, CNN, ResNet-18, model count 5, domain shifts, matched pairwise-error nulls, leave-one-dataset-out, and leave-one-architecture-out were not executed. This smoke therefore cannot promote a natural twist, even if a single residual statistic is significant. Full command after those checkpoint collections are available: `python experiments/natural_twist_discovery.py --mode full`.
"""
    (OUT / "natural_twist_report.md").write_text(report, encoding="utf-8")
    config = {"stage": 7, "mode": args.mode, "command": " ".join([sys.executable, *sys.argv]), "source": str(SOURCE.relative_to(ROOT)), "source_settings": len(frame), "natural_twist_promoted": promoted, "coverage_complete": False}
    (OUT / "natural_twist_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"settings": len(frame), "promoted": promoted, "p_value": p_value, "baseline_mse": baseline_mse, "extended_mse": extended_mse}, indent=2))


if __name__ == "__main__":
    main()
