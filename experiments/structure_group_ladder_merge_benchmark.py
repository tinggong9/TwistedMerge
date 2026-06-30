#!/usr/bin/env python
"""Actionable merge benchmark for the structure-group ladder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ladder_merge_methods import (  # noqa: E402
    METHOD_METADATA,
    estimate_signs_and_positive_scales,
    transform_mlp_positive_scale,
    transform_mlp_signed,
)
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    collect_features,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    format_markdown_table,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.structure_group_ladder import (  # noqa: E402
    StructureGroupLadderMerge,
    estimate_pairwise_permutations_from_activations,
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_worktree_dirty() -> bool | str:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(status.strip())
    except Exception:
        return "unknown"


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def bootstrap_mean_ci(values, n_bootstrap: int = 500, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def standard_error(values) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def diag_by_level(ladder_result) -> dict[str, object]:
    out: dict[str, object] = {
        "ladder_final_decision": ladder_result.final_decision,
        "ladder_selected_level": ladder_result.selected_level,
        "supports_brauer_projective_interpretation": any(
            diag.supports_brauer_projective_interpretation
            for diag in ladder_result.diagnostics
        ),
        "has_finite_index_candidate": any(
            diag.is_finite_index_candidate
            for diag in ladder_result.diagnostics
        ),
    }
    for diag in ladder_result.diagnostics:
        prefix = diag.level
        out[f"{prefix}_centrality"] = diag.centrality_score
        out[f"{prefix}_cycle_score"] = diag.cycle_score
        out[f"{prefix}_residual_type"] = diag.residual_type
        out[f"{prefix}_phase_residual"] = diag.phase_residual
        out[f"{prefix}_detected_order_d"] = diag.detected_order_d
    perm = out.get("permutation_centrality", float("nan"))
    mono = out.get("monomial_phase_or_scale_centrality", float("nan"))
    gl = out.get("low_rank_GL_centrality", float("nan"))
    out["monomial_centrality_improvement_from_permutation"] = (
        float(perm - mono) if np.isfinite(perm) and np.isfinite(mono) else float("nan")
    )
    out["gl_centrality_improvement_from_permutation"] = (
        float(perm - gl) if np.isfinite(perm) and np.isfinite(gl) else float("nan")
    )
    return out


def add_method_row(
    rows: list[dict],
    *,
    base: dict,
    method: str,
    metrics: dict[str, float] | None,
    single_best_accuracy: float,
    extra: dict | None = None,
) -> None:
    meta = METHOD_METADATA[method]
    accuracy = float(metrics["accuracy"]) if metrics is not None else float("nan")
    loss = float(metrics["loss"]) if metrics is not None else float("nan")
    row = {
        **base,
        "method": method,
        "loss": loss,
        "accuracy": accuracy,
        "single_best_accuracy": single_best_accuracy,
        "merge_degradation": single_best_accuracy - accuracy if np.isfinite(accuracy) else float("nan"),
        "symmetry_status": meta.symmetry_status,
        "is_single_model": meta.is_single_model,
        "capacity_matched_to_weight_average": meta.capacity_matched_to_weight_average,
        "method_notes": meta.notes,
        "evaluation_status": "evaluated" if metrics is not None else "not_evaluated",
    }
    if extra:
        row.update(extra)
    rows.append(row)


def run_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int) -> list[dict]:
    device = device_from_arg(args.device)
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 999)
    match_loader = make_loader(train_data, args.batch_size, shuffle=False, seed=seed + 501)

    models = []
    individual_accuracies = []
    individual_losses = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        model = make_model("mlp", spec, width)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed + 11)
        train_model(model, train_loader, args.epochs, args.lr, device)
        metrics = evaluate_model(model, test_loader, device)
        individual_accuracies.append(metrics["accuracy"])
        individual_losses.append(metrics["loss"])
        model.to("cpu")
        models.append(model)

    features = {
        idx: collect_features(model, match_loader, device)
        for idx, model in enumerate(models)
    }
    pairwise = estimate_pairwise_permutations_from_activations(features, n_models, width)
    ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    ladder_result = StructureGroupLadderMerge(max_order=args.max_order).run(
        {"permutation": pairwise},
        n_models=n_models,
        width=width,
        activations=features,
        candidate_lift_rank=width,
    )
    diagnostics = diag_by_level(ladder_result)
    setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
    single_best_accuracy = float(max(individual_accuracies))
    base = {
        "setting_id": setting_id,
        "dataset": "mnist",
        "architecture": "mlp_relu",
        "n_models": n_models,
        "width": width,
        "seed": seed,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "matching": "activation",
        "sync_reference": ref,
        "sync_disagreement": sync_disagreement,
        "individual_accuracy_mean": float(np.mean(individual_accuracies)),
        "individual_accuracy_min": float(np.min(individual_accuracies)),
        "individual_accuracy_max": single_best_accuracy,
        "individual_loss_mean": float(np.mean(individual_losses)),
        **diagnostics,
    }
    rows: list[dict] = []

    weight_avg = average_models(models, "mlp", spec, width)
    add_method_row(
        rows,
        base=base,
        method="weight_average",
        metrics=evaluate_model(weight_avg, test_loader, device),
        single_best_accuracy=single_best_accuracy,
    )

    soup, soup_indices, soup_metrics = greedy_soup(models, val_loader, test_loader, device, "mlp", spec, width)
    add_method_row(
        rows,
        base=base,
        method="greedy_soup",
        metrics=soup_metrics,
        single_best_accuracy=single_best_accuracy,
        extra={"soup_indices": json.dumps(soup_indices)},
    )

    aligned_c2m3 = [
        permute_model_to_reference(model, "mlp", spec, width, synced[idx])
        for idx, model in enumerate(models)
    ]
    c2m3_model = average_models(aligned_c2m3, "mlp", spec, width)
    add_method_row(
        rows,
        base=base,
        method="c2m3_permutation",
        metrics=evaluate_model(c2m3_model, test_loader, device),
        single_best_accuracy=single_best_accuracy,
    )

    signed_models = []
    scaled_models = []
    scale_stats = []
    sign_flip_fractions = []
    for idx, model in enumerate(models):
        perm = synced[idx]
        if idx == ref:
            signs = np.ones(width, dtype=float)
            scales = np.ones(width, dtype=float)
        else:
            signs, scales = estimate_signs_and_positive_scales(features[ref], features[idx], perm)
        sign_flip_fractions.append(float(np.mean(signs < 0.0)))
        scale_stats.append(float(np.mean(scales)))
        signed_models.append(transform_mlp_signed(model, spec, width, perm, signs))
        scaled_models.append(transform_mlp_positive_scale(model, spec, width, perm, scales))

    signed_model = average_models(signed_models, "mlp", spec, width)
    add_method_row(
        rows,
        base=base,
        method="signed_permutation",
        metrics=evaluate_model(signed_model, test_loader, device),
        single_best_accuracy=single_best_accuracy,
        extra={
            "mean_sign_flip_fraction": float(np.mean(sign_flip_fractions)),
            "symmetry_warning": "negative sign flips are not exact ReLU symmetries",
        },
    )

    scale_model = average_models(scaled_models, "mlp", spec, width)
    add_method_row(
        rows,
        base=base,
        method="monomial_scale",
        metrics=evaluate_model(scale_model, test_loader, device),
        single_best_accuracy=single_best_accuracy,
        extra={
            "mean_positive_scale": float(np.mean(scale_stats)),
            "symmetry_warning": "positive ReLU scaling is exact before averaging",
        },
    )

    add_method_row(
        rows,
        base=base,
        method="low_rank_GL_diagnostic",
        metrics=None,
        single_best_accuracy=single_best_accuracy,
        extra={
            "symmetry_warning": "not evaluated as a same-architecture ReLU merged model",
        },
    )

    add_method_row(
        rows,
        base=base,
        method="ensemble_upper_bound",
        metrics=evaluate_ensemble(models, test_loader, device),
        single_best_accuracy=single_best_accuracy,
    )

    by_method = {row["method"]: row for row in rows}
    c2m3_acc = by_method["c2m3_permutation"]["accuracy"]
    weight_acc = by_method["weight_average"]["accuracy"]
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    c2m3_loss = by_method["c2m3_permutation"]["loss"]
    weight_loss = by_method["weight_average"]["loss"]
    greedy_loss = by_method["greedy_soup"]["loss"]
    for row in rows:
        acc = row["accuracy"]
        loss = row["loss"]
        row["accuracy_delta_vs_c2m3"] = acc - c2m3_acc if np.isfinite(acc) else float("nan")
        row["accuracy_delta_vs_weight_average"] = acc - weight_acc if np.isfinite(acc) else float("nan")
        row["accuracy_delta_vs_greedy_soup"] = acc - greedy_acc if np.isfinite(acc) else float("nan")
        row["loss_delta_vs_c2m3"] = loss - c2m3_loss if np.isfinite(loss) else float("nan")
        row["loss_delta_vs_weight_average"] = loss - weight_loss if np.isfinite(loss) else float("nan")
        row["loss_delta_vs_greedy_soup"] = loss - greedy_loss if np.isfinite(loss) else float("nan")
    return rows


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for (n_models, width, method), group in df.groupby(["n_models", "width", "method"], dropna=False):
        evaluated = group[group["evaluation_status"] == "evaluated"]
        accuracy = pd.to_numeric(evaluated["accuracy"], errors="coerce")
        degradation = pd.to_numeric(evaluated["merge_degradation"], errors="coerce")
        delta_c2m3 = pd.to_numeric(evaluated["accuracy_delta_vs_c2m3"], errors="coerce")
        delta_weight = pd.to_numeric(evaluated["accuracy_delta_vs_weight_average"], errors="coerce")
        delta_greedy = pd.to_numeric(evaluated["accuracy_delta_vs_greedy_soup"], errors="coerce")
        ci_low, ci_high = bootstrap_mean_ci(accuracy, n_bootstrap=n_bootstrap, seed=17 + int(n_models) * 100 + int(width))
        status = "not_evaluated"
        if len(evaluated):
            mean_delta = float(delta_c2m3.mean())
            if method == "c2m3_permutation":
                status = "baseline"
            elif mean_delta > 0:
                status = "descriptive_mean_gain_vs_c2m3"
            elif mean_delta < 0:
                status = "descriptive_mean_loss_vs_c2m3"
            else:
                status = "no_mean_delta_vs_c2m3"
        rows.append(
            {
                "n_models": n_models,
                "width": width,
                "method": method,
                "n_rows": int(len(group)),
                "n_evaluated": int(len(evaluated)),
                "n_seeds": int(group["seed"].nunique()),
                "mean_accuracy": float(accuracy.mean()) if len(evaluated) else float("nan"),
                "accuracy_standard_error": standard_error(accuracy),
                "accuracy_bootstrap_ci_low": ci_low,
                "accuracy_bootstrap_ci_high": ci_high,
                "mean_merge_degradation": float(degradation.mean()) if len(evaluated) else float("nan"),
                "mean_accuracy_delta_vs_c2m3": float(delta_c2m3.mean()) if len(evaluated) else float("nan"),
                "mean_accuracy_delta_vs_weight_average": float(delta_weight.mean()) if len(evaluated) else float("nan"),
                "mean_accuracy_delta_vs_greedy_soup": float(delta_greedy.mean()) if len(evaluated) else float("nan"),
                "mean_monomial_centrality_improvement": float(
                    pd.to_numeric(group["monomial_centrality_improvement_from_permutation"], errors="coerce").mean()
                ),
                "method_status_vs_c2m3": status,
                "symmetry_status": str(group["symmetry_status"].iloc[0]),
                "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    return format_markdown_table(rows, columns)


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    method_rows = []
    for method, meta in METHOD_METADATA.items():
        method_rows.append(
            {
                "method": method,
                "symmetry_status": meta.symmetry_status,
                "is_single_model": meta.is_single_model,
                "capacity_matched": meta.capacity_matched_to_weight_average,
            }
        )
    perf_cols = [
        "n_models",
        "width",
        "method",
        "n_seeds",
        "mean_accuracy",
        "accuracy_standard_error",
        "mean_merge_degradation",
        "mean_accuracy_delta_vs_c2m3",
        "mean_accuracy_delta_vs_weight_average",
        "mean_accuracy_delta_vs_greedy_soup",
        "method_status_vs_c2m3",
    ]
    residual_rows = (
        df.groupby(["n_models", "width"], dropna=False)
        .agg(
            mean_permutation_centrality=("permutation_centrality", "mean"),
            mean_monomial_centrality=("monomial_phase_or_scale_centrality", "mean"),
            mean_gl_centrality=("low_rank_GL_centrality", "mean"),
            mean_monomial_improvement=("monomial_centrality_improvement_from_permutation", "mean"),
            fraction_projective_candidates=("supports_brauer_projective_interpretation", lambda s: float(pd.Series(s).astype(bool).mean()) if len(s) else 0.0),
        )
        .reset_index()
        .to_dict("records")
    )
    residual_cols = [
        "n_models",
        "width",
        "mean_permutation_centrality",
        "mean_monomial_centrality",
        "mean_gl_centrality",
        "mean_monomial_improvement",
        "fraction_projective_candidates",
    ]
    mono_summary = summary[summary["method"] == "monomial_scale"].copy()
    c2m3_beaters = summary[
        (summary["method"] != "ensemble_upper_bound")
        & (summary["method"] != "low_rank_GL_diagnostic")
        & (pd.to_numeric(summary["mean_accuracy_delta_vs_c2m3"], errors="coerce") > 0)
    ]
    mean_mono_delta = float(pd.to_numeric(mono_summary["mean_accuracy_delta_vs_c2m3"], errors="coerce").mean()) if not mono_summary.empty else float("nan")
    mean_mono_delta_greedy = float(pd.to_numeric(mono_summary["mean_accuracy_delta_vs_greedy_soup"], errors="coerce").mean()) if not mono_summary.empty else float("nan")
    mean_mono_improvement = float(pd.to_numeric(summary["mean_monomial_centrality_improvement"], errors="coerce").mean())
    if np.isfinite(mean_mono_delta) and mean_mono_delta > 0:
        mono_perf = "Monomial scaling has a descriptive positive mean accuracy delta versus C2M3 in this run."
    elif np.isfinite(mean_mono_delta) and mean_mono_delta < 0:
        mono_perf = "Monomial scaling has a descriptive negative mean accuracy delta versus C2M3 in this run."
    else:
        mono_perf = "Monomial scaling has no mean accuracy gain versus C2M3 in this run."
    if c2m3_beaters.empty:
        beat_text = "No evaluated single-model ladder method beats C2M3 on mean accuracy across every fixed setting."
    else:
        beat_text = (
            "Some evaluated methods have descriptive positive mean deltas versus C2M3 in at least one fixed setting. "
            "These are descriptive only unless replicated with stronger statistics."
        )
    report = f"""# Structure-Group Ladder Merge Report

This report is generated by `experiments/structure_group_ladder_merge_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State At Report Generation

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_worktree_dirty()}`

## Grid Settings

- Dataset: MNIST
- Architecture: one-hidden-layer ReLU MLP
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples: `{args.max_train_samples}`
- Test samples: `{args.max_test_samples}`
- Batch size: `{args.batch_size}`
- Matching: activation

## Method Labels

{table(method_rows, ["method", "symmetry_status", "is_single_model", "capacity_matched"])}

## Per-Method Performance Summary

{format_markdown_table(summary.to_dict("records"), perf_cols)}

## Residual Diagnostics

{table(residual_rows, residual_cols)}

## Interpretation

Mean monomial centrality improvement from the permutation level is `{mean_mono_improvement:.4f}`.
{mono_perf}
Mean monomial accuracy delta versus greedy soup is `{mean_mono_delta_greedy:.4f}`.
{beat_text}

Positive monomial scaling is an exact ReLU reparameterization before averaging.
Signed permutation is heuristic for ReLU because negative sign flips are not exact
hidden-unit symmetries.  Low-rank GL is diagnostic only and is not evaluated as
a same-architecture ReLU merged model.

## Negative Boundaries

- This does not prove TwistedMerge++ beats C2M3 on real MNIST/CIFAR.
- This does not make signed or GL transforms exact single-model ReLU merges.
- This does not make noncentral residuals Brauer/projective classes.
- Ensemble and GL diagnostic rows are not capacity-matched single merged models.
- All fixed-setting comparisons have five seeds and are descriptive.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1600,1601,1602,1603,1604")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    spec, train_data, test_data = load_dataset(
        "mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
    )
    rows = []
    for seed in parse_csv(args.seeds, int):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                rows.extend(run_setting(args, spec, train_data, test_data, seed, n_models, width))

    df = pd.DataFrame(rows)
    summary = summarize(df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "structure_group_ladder_merge_benchmark.csv"
    summary_path = csv_dir / "structure_group_ladder_merge_summary.csv"
    report_path = args.reports_dir / "structure_group_ladder_merge_report.md"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(args, df, summary, report_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
