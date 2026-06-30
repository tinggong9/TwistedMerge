#!/usr/bin/env python
"""Mine MNIST model-merging residuals for finite-index projective structure."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.finite_index_twists import clock_matrix, root_of_unity, shift_matrix, torsion_order  # noqa: E402
from src.metrics import capture_environment, pearsonr, save_json  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    compose_perm,
    compute_pairwise_permutations,
    device_from_arg,
    evaluate_model,
    format_markdown_table,
    load_dataset,
    make_loader,
    make_model,
    permutation_matrix,
    permute_model_to_reference,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.twisted_merge_plus import TwistedMergePlus  # noqa: E402


THRESHOLDS = {
    "strict": 1e-6,
    "medium": 1e-3,
    "loose": 1e-2,
}


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def safe_spearman(x, y) -> float:
    xv = pd.Series(list(x), dtype=float)
    yv = pd.Series(list(y), dtype=float)
    if len(xv) < 2:
        return float("nan")
    return pearsonr(xv.rank(method="average"), yv.rank(method="average"))


def nearest_root_of_unity(phase: complex | None, max_order: int) -> dict[str, object]:
    if phase is None or not np.isfinite(phase.real) or not np.isfinite(phase.imag):
        return {
            "nearest_root_q": np.nan,
            "nearest_root_exponent": np.nan,
            "detected_order_d": np.nan,
            "nearest_root_phase_real": np.nan,
            "nearest_root_phase_imag": np.nan,
            "phase_residual": np.nan,
        }
    unit = complex(phase) / abs(phase)
    best = (abs(unit - 1.0), 1, 0, 1, complex(1.0))
    for q in range(2, max_order + 1):
        for exponent in range(q):
            root = root_of_unity(q, exponent)
            order = torsion_order(q, exponent)
            residual = abs(unit - root)
            if residual < best[0] - 1e-15 or (abs(residual - best[0]) <= 1e-15 and order < best[3]):
                best = (residual, q, exponent, order, root)
    residual, q, exponent, order, root = best
    return {
        "nearest_root_q": q,
        "nearest_root_exponent": exponent,
        "detected_order_d": order,
        "nearest_root_phase_real": float(root.real),
        "nearest_root_phase_imag": float(root.imag),
        "phase_residual": float(residual),
    }


def permutation_cycle_stats(perm: np.ndarray) -> dict[str, float | int]:
    visited = np.zeros(len(perm), dtype=bool)
    lengths = []
    for start in range(len(perm)):
        if visited[start]:
            continue
        cur = start
        length = 0
        while not visited[cur]:
            visited[cur] = True
            length += 1
            cur = int(perm[cur])
        lengths.append(length)
    return {
        "fixed_point_fraction": float(np.mean(perm == np.arange(len(perm)))) if len(perm) else float("nan"),
        "permutation_num_cycles": int(len(lengths)),
        "permutation_avg_cycle_length": float(np.mean(lengths)) if lengths else float("nan"),
        "permutation_max_cycle_length": int(max(lengths)) if lengths else 0,
    }


def defect_metrics(
    defect_matrix: np.ndarray,
    width: int,
    max_order: int,
    centrality_tol: float = 1e-12,
) -> dict[str, object]:
    eye = np.eye(width, dtype=complex)
    scalar = complex(np.trace(defect_matrix) / max(width, 1))
    central_target = scalar * eye
    centrality = float(np.linalg.norm(defect_matrix - central_target, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12))
    phase = scalar / abs(scalar) if abs(scalar) > centrality_tol else None
    root = nearest_root_of_unity(phase, max_order)
    phase_angle = float(np.angle(phase)) if phase is not None else float("nan")
    cycle = float(np.linalg.norm(defect_matrix - eye, ord="fro") / max(np.sqrt(2.0 * width), 1e-12))
    detected_order = root["detected_order_d"]
    phase_residual = root["phase_residual"]
    candidate_score = (
        centrality + float(phase_residual)
        if isinstance(phase_residual, float) and math.isfinite(float(phase_residual))
        else float("inf")
    )
    return {
        "cycle_score": cycle,
        "centrality_score": centrality,
        "scalar_trace_real": float(scalar.real),
        "scalar_trace_imag": float(scalar.imag),
        "phase_real": float(phase.real) if phase is not None else float("nan"),
        "phase_imag": float(phase.imag) if phase is not None else float("nan"),
        "phase_angle": phase_angle,
        "finite_index_candidate_score": candidate_score,
        **root,
        "is_scalar_central_permutation": bool(centrality <= centrality_tol and int(detected_order) == 1)
        if not (isinstance(detected_order, float) and math.isnan(detected_order))
        else False,
    }


def add_threshold_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    order = pd.to_numeric(out["detected_order_d"], errors="coerce")
    for name, tol in THRESHOLDS.items():
        out[f"finite_index_candidate_{name}"] = (
            (out["centrality_score"] <= tol)
            & (out["phase_residual"] <= tol)
            & (order > 1)
        )
    return out


def positive_control_rows(max_order: int) -> list[dict]:
    rows = []
    for order in [2, 3, 4]:
        zeta = root_of_unity(order, 1)
        U = clock_matrix(order, zeta)
        V = shift_matrix(order)
        pairwise = {
            (0, 0): np.eye(order, dtype=complex),
            (1, 1): np.eye(order, dtype=complex),
            (2, 2): np.eye(order, dtype=complex),
            (0, 1): U,
            (1, 2): V,
            (2, 0): np.linalg.inv(U) @ np.linalg.inv(V),
        }
        defect = U @ V @ np.linalg.inv(U) @ np.linalg.inv(V)
        metrics = defect_metrics(defect, order, max_order)
        tmpp = TwistedMergePlus().run(
            pairwise,
            n_models=3,
            width=order,
            candidate_lift_rank=order,
            max_root_order=max_order,
        )
        rows.append(
            {
                "source": "positive_control",
                "setting_id": f"clock_shift_order_{order}",
                "dataset": "synthetic_clock_shift",
                "architecture": "projective_pair",
                "n_models": 3,
                "width": order,
                "seed": -1,
                "triangle": "0-1-2",
                "i": 0,
                "j": 1,
                "k": 2,
                "matching": "exact_clock_shift",
                "individual_accuracy_mean": np.nan,
                "single_best_accuracy": np.nan,
                "weight_average_accuracy": np.nan,
                "weight_merge_degradation": np.nan,
                "c2m3_accuracy": np.nan,
                "c2m3_merge_degradation": np.nan,
                "sync_disagreement": np.nan,
                "tmpp_classification": tmpp.diagnostics.classification,
                "tmpp_selected_method": tmpp.selected_method,
                "candidate_rank_width": order,
                "rank_divisible_by_detected_order": True,
                **metrics,
                "fixed_point_fraction": np.nan,
                "permutation_num_cycles": np.nan,
                "permutation_avg_cycle_length": np.nan,
                "permutation_max_cycle_length": np.nan,
                "interpretation": "positive control: exact scalar root-of-unity projective defect",
            }
        )
    return rows


def run_real_setting(args, spec, train_data, test_loader, seed: int, n_models: int, width: int) -> list[dict]:
    device = device_from_arg(args.device)
    models = []
    individual_accuracies = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        model = make_model("mlp", spec, width)
        train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=model_seed + 11)
        train_model(model, train_loader, args.epochs, args.lr, device)
        metrics = evaluate_model(model, test_loader, device)
        individual_accuracies.append(metrics["accuracy"])
        model.to("cpu")
        models.append(model)

    match_loader = make_loader(train_data, args.batch_size, shuffle=False, seed=args.dataset_seed + 501)
    pairwise = compute_pairwise_permutations(models, "mlp", match_loader, device, "activation")
    sync_ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    aligned_synced = [
        permute_model_to_reference(model, "mlp", spec, width, synced[idx])
        for idx, model in enumerate(models)
    ]
    weight_average = average_models(models, "mlp", spec, width)
    c2m3_model = average_models(aligned_synced, "mlp", spec, width)
    weight_metrics = evaluate_model(weight_average, test_loader, device)
    c2m3_metrics = evaluate_model(c2m3_model, test_loader, device)
    single_best = float(max(individual_accuracies))
    mean_individual = float(np.mean(individual_accuracies))
    tmpp = TwistedMergePlus().run(
        pairwise,
        n_models=n_models,
        width=width,
        candidate_lift_rank=width,
        max_root_order=args.max_order,
    )

    rows = []
    setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
    for i, j, k in combinations(range(n_models), 3):
        defect_perm = compose_perm(compose_perm(pairwise[(i, j)], pairwise[(j, k)]), pairwise[(k, i)])
        defect_matrix = permutation_matrix(defect_perm)
        metrics = defect_metrics(defect_matrix, width, args.max_order)
        cycles = permutation_cycle_stats(defect_perm)
        detected_order = metrics["detected_order_d"]
        rank_divisible = (
            bool(width % int(detected_order) == 0)
            if not (isinstance(detected_order, float) and math.isnan(detected_order)) and int(detected_order) > 1
            else False
        )
        rows.append(
            {
                "source": "real_mnist",
                "setting_id": setting_id,
                "dataset": "mnist",
                "architecture": "mlp",
                "n_models": n_models,
                "width": width,
                "seed": seed,
                "triangle": f"{i}-{j}-{k}",
                "i": i,
                "j": j,
                "k": k,
                "matching": "activation",
                "individual_accuracy_mean": mean_individual,
                "single_best_accuracy": single_best,
                "weight_average_accuracy": weight_metrics["accuracy"],
                "weight_merge_degradation": single_best - weight_metrics["accuracy"],
                "c2m3_accuracy": c2m3_metrics["accuracy"],
                "c2m3_merge_degradation": single_best - c2m3_metrics["accuracy"],
                "sync_disagreement": sync_disagreement,
                "sync_reference": sync_ref,
                "tmpp_classification": tmpp.diagnostics.classification,
                "tmpp_selected_method": tmpp.selected_method,
                "candidate_rank_width": width,
                "rank_divisible_by_detected_order": rank_divisible,
                **metrics,
                **cycles,
                "interpretation": "real MNIST activation-permutation triangle residual",
            }
        )
    return rows


def most_common_order(series: pd.Series) -> str:
    clean = [int(value) for value in pd.to_numeric(series, errors="coerce").dropna().tolist()]
    nontrivial = [value for value in clean if value > 1]
    if not nontrivial:
        return "none"
    return str(Counter(nontrivial).most_common(1)[0][0])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold_name, tol in THRESHOLDS.items():
        for scope, cols in [
            ("overall_by_source", ["source"]),
            ("real_setting", ["source", "dataset", "architecture", "n_models", "width"]),
        ]:
            for keys, group in df.groupby(cols, dropna=False):
                if scope == "real_setting" and group["source"].iloc[0] != "real_mnist":
                    continue
                if not isinstance(keys, tuple):
                    keys = (keys,)
                labels = dict(zip(cols, keys, strict=True))
                candidate_col = f"finite_index_candidate_{threshold_name}"
                candidate_score = group["finite_index_candidate_score"].replace([np.inf, -np.inf], np.nan)
                row = {
                    "summary_scope": scope,
                    "threshold": threshold_name,
                    "threshold_value": tol,
                    "n_triangles": int(len(group)),
                    "mean_centrality_score": float(group["centrality_score"].mean()),
                    "min_centrality_score": float(group["centrality_score"].min()),
                    "mean_phase_residual": float(group["phase_residual"].mean()),
                    "min_phase_residual": float(group["phase_residual"].min()),
                    "fraction_finite_index_candidates": float(group[candidate_col].mean()),
                    "most_common_detected_order": most_common_order(group["detected_order_d"]),
                    "pearson_centrality_vs_weight_degradation": pearsonr(group["centrality_score"], group["weight_merge_degradation"]),
                    "spearman_centrality_vs_weight_degradation": safe_spearman(group["centrality_score"], group["weight_merge_degradation"]),
                    "pearson_candidate_score_vs_weight_degradation": pearsonr(candidate_score, group["weight_merge_degradation"]),
                    "pearson_cycle_score_vs_weight_degradation": pearsonr(group["cycle_score"], group["weight_merge_degradation"]),
                    "tmpp_classification_counts": json.dumps(group["tmpp_classification"].value_counts(dropna=False).to_dict(), sort_keys=True),
                }
                row.update(labels)
                rows.append(row)
    return pd.DataFrame(rows)


def plot_phase_histogram(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    real = df[(df["source"] == "real_mnist") & np.isfinite(df["phase_angle"])]
    positive = df[(df["source"] == "positive_control") & np.isfinite(df["phase_angle"])]
    if not real.empty:
        ax.hist(real["phase_angle"], bins=24, alpha=0.7, label="real MNIST residuals")
    if not positive.empty:
        ax.scatter(positive["phase_angle"], np.zeros(len(positive)), marker="x", color="black", label="clock-shift controls")
    ax.set_xlabel("phase angle of trace-normalized scalar")
    ax.set_ylabel("count")
    ax.set_title("Finite-index residual phase mining")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    real = df[df["source"] == "real_mnist"].copy()
    controls = df[df["source"] == "positive_control"].copy()
    threshold_rows = summary[summary["summary_scope"] == "overall_by_source"].to_dict("records")
    real_summary = summary[
        (summary["summary_scope"] == "real_setting")
        & (summary["threshold"] == "medium")
    ].to_dict("records")
    control_columns = [
        "setting_id",
        "detected_order_d",
        "centrality_score",
        "phase_residual",
        "tmpp_classification",
        "tmpp_selected_method",
    ]
    summary_columns = [
        "source",
        "threshold",
        "n_triangles",
        "mean_centrality_score",
        "min_centrality_score",
        "fraction_finite_index_candidates",
        "most_common_detected_order",
        "pearson_cycle_score_vs_weight_degradation",
    ]
    real_columns = [
        "dataset",
        "architecture",
        "n_models",
        "width",
        "threshold",
        "n_triangles",
        "mean_centrality_score",
        "min_centrality_score",
        "fraction_finite_index_candidates",
        "most_common_detected_order",
    ]
    example_columns = [
        "setting_id",
        "triangle",
        "width",
        "cycle_score",
        "centrality_score",
        "phase_residual",
        "detected_order_d",
        "finite_index_candidate_score",
        "tmpp_classification",
        "weight_merge_degradation",
    ]
    examples = real.sort_values("finite_index_candidate_score").head(10).to_dict("records")
    strict_real = float(real["finite_index_candidate_strict"].mean()) if not real.empty else 0.0
    medium_real = float(real["finite_index_candidate_medium"].mean()) if not real.empty else 0.0
    if strict_real == 0.0 and medium_real == 0.0:
        conclusion = (
            "No real MNIST activation-permutation triangle residuals pass the strict or medium "
            "finite-index scalar phase thresholds.  In this run, finite-index TwistedMerge remains "
            "a controlled/projective extension, not an observed natural MNIST permutation residual pattern."
        )
    else:
        conclusion = (
            "Some real MNIST residuals pass strict or medium thresholds.  This is descriptive evidence only; "
            "it is not a model-merging improvement claim and needs replication/capacity-matched tests."
        )
    report = f"""# Finite-Index Residual Mining Report

This report is generated by `experiments/mine_finite_index_residuals.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Grid Settings

- Dataset: MNIST
- Architecture: one-hidden-layer MLP
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples: `{args.max_train_samples}`
- Test samples: `{args.max_test_samples}`
- Matching: activation
- Max root order: `{args.max_order}`

This default run uses five seeds and widths 16/32 to keep CPU time bounded.
The script supports `--widths 16,32,64` and more seeds for a larger run.

## Outputs

- Per-triangle CSV: `reports/csv/finite_index_residual_mining.csv`
- Summary CSV: `reports/csv/finite_index_residual_mining_summary.csv`
- Phase histogram: `reports/plots/finite_index_residual_phase_histogram.pdf`
- This report: `reports/finite_index_residual_mining_report.md`

## Positive-Control Detector Table

{format_markdown_table(controls.to_dict("records"), control_columns)}

## Threshold Sensitivity

{format_markdown_table(threshold_rows, summary_columns)}

## Real MNIST Residual Mining Summary

{format_markdown_table(real_summary, real_columns)}

## Most Finite-Index-Like Real Residuals

{format_markdown_table(examples, example_columns)}

## Negative Result Status

{conclusion}

Permutation triangle defects are usually not scalar matrices.  A nontrivial
permutation cycle is not the same thing as a scalar root-of-unity projective
phase.  The mining table therefore reports fixed-point fraction and cycle
statistics alongside centrality and phase residuals.

## Interpretation

- The clock-shift positive controls verify that the detector can identify
  exact finite-index projective residuals of orders 2, 3, and 4.
- Real MNIST activation-permutation residuals were mined directly from trained
  models, but this run should be treated as descriptive.
- If no strict/medium real candidates appear, the honest conclusion is that
  finite-index TwistedMerge has not yet been observed in natural MNIST
  permutation residuals.
- Loose-threshold candidates, if any, are not strong evidence.
- No practical model-merging improvement is claimed here.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1400,1401,1402,1403,1404")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--max-order", type=int, default=12)
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
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=args.dataset_seed + 999)
    rows = positive_control_rows(args.max_order)
    for seed in parse_csv(args.seeds, int):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                rows.extend(run_real_setting(args, spec, train_data, test_loader, seed, n_models, width))

    df = add_threshold_columns(pd.DataFrame(rows))
    summary = summarize(df)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "finite_index_residual_mining.csv"
    summary_path = csv_dir / "finite_index_residual_mining_summary.csv"
    plot_path = plot_dir / "finite_index_residual_phase_histogram.pdf"
    report_path = args.reports_dir / "finite_index_residual_mining_report.md"
    config_path = args.reports_dir / "configs" / "finite_index_residual_mining_config.json"

    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_phase_histogram(df, plot_path)
    write_report(args, df, summary, report_path)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "environment": capture_environment(),
            "commit": git_commit(),
        },
    )
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
