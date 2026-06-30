#!/usr/bin/env python
"""Evaluate real block-orthogonal ladder diagnostics on MNIST MLPs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.block_gauge_alignment import (  # noqa: E402
    estimate_block_orthogonal_alignments,
    summarize_block_alignment_stats,
)
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    collect_features,
    device_from_arg,
    evaluate_model,
    format_markdown_table,
    load_dataset,
    make_loader,
    make_model,
    set_seed,
    train_model,
)
from src.structure_group_ladder import (  # noqa: E402
    LadderDiagnostics,
    LadderResult,
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


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


def diag_row(
    *,
    source: str,
    setting_id: str,
    n_models: int,
    width: int,
    seed: int,
    block_size: int,
    triangle: tuple[int, int, int],
    result: LadderResult,
    diag: LadderDiagnostics,
    block_stats: dict[str, float | bool],
    individual_accuracy_mean: float | None = None,
    individual_accuracy_max: float | None = None,
) -> dict:
    by_level = {item.level: item for item in result.diagnostics}
    perm = by_level.get("permutation")
    block = by_level.get("block_orthogonal")
    perm_centrality = perm.centrality_score if perm is not None else float("nan")
    block_centrality = block.centrality_score if block is not None else float("nan")
    improvement = (
        float(perm_centrality - block_centrality)
        if np.isfinite(perm_centrality) and np.isfinite(block_centrality)
        else float("nan")
    )
    return {
        "setting_id": setting_id,
        "source": source,
        "level": diag.level,
        "n_models": n_models,
        "width": width,
        "seed": seed,
        "block_size": block_size,
        "triangle": "-".join(str(item) for item in triangle),
        "cycle_score": diag.cycle_score,
        "centrality_score": diag.centrality_score,
        "phase_residual": diag.phase_residual,
        "detected_order_d": diag.detected_order_d,
        "rank_allowed": diag.rank_allowed,
        "residual_type": diag.residual_type,
        "selected_resolution": diag.selected_resolution,
        "centrality_improvement_from_previous_level": diag.centrality_improvement_from_previous_level,
        "centrality_improvement_from_permutation_to_block": improvement,
        "supports_brauer_projective_interpretation": diag.supports_brauer_projective_interpretation,
        "is_finite_index_candidate": diag.is_finite_index_candidate,
        "final_decision": result.final_decision,
        "selected_level": result.selected_level,
        "mean_pairwise_block_residual": block_stats.get("mean_pairwise_block_residual", float("nan")),
        "max_pairwise_block_residual": block_stats.get("max_pairwise_block_residual", float("nan")),
        "used_remainder_block": block_stats.get("used_remainder_block", False),
        "individual_accuracy_mean": float(individual_accuracy_mean) if individual_accuracy_mean is not None else float("nan"),
        "individual_accuracy_max": float(individual_accuracy_max) if individual_accuracy_max is not None else float("nan"),
        "merge_evaluated": False,
        "merge_accuracy": float("nan"),
        "merge_notes": "not evaluated: block-orthogonal rotations are feature-space diagnostics for ReLU MLPs",
        "notes": " ".join(diag.notes),
    }


def rows_from_result(
    *,
    source: str,
    setting_id: str,
    n_models: int,
    width: int,
    seed: int,
    block_size: int,
    triangle: tuple[int, int, int],
    result: LadderResult,
    block_stats: dict[str, float | bool],
    individual_accuracy_mean: float | None = None,
    individual_accuracy_max: float | None = None,
) -> list[dict]:
    return [
        diag_row(
            source=source,
            setting_id=setting_id,
            n_models=n_models,
            width=width,
            seed=seed,
            block_size=block_size,
            triangle=triangle,
            result=result,
            diag=diag,
            block_stats=block_stats,
            individual_accuracy_mean=individual_accuracy_mean,
            individual_accuracy_max=individual_accuracy_max,
        )
        for diag in result.diagnostics
    ]


def controlled_rows(max_order: int) -> list[dict]:
    rows: list[dict] = []

    rng = np.random.default_rng(2029)
    base = rng.normal(size=(256, 4))
    gauges = {
        0: np.eye(4),
        1: block_diag([rotation(0.35), rotation(-0.2)]),
        2: block_diag([rotation(-0.15), rotation(0.4)]),
    }
    activations = {idx: base @ gauge for idx, gauge in gauges.items()}
    perms = {(i, j): np.arange(4) for i in range(3) for j in range(3)}
    block_maps, stats = estimate_block_orthogonal_alignments(perms, activations, 3, 4, 2)
    result = StructureGroupLadderMerge(max_order=max_order).run(
        {"block_orthogonal": block_maps},
        n_models=3,
        width=4,
        candidate_lift_rank=4,
        triples=[(0, 1, 2)],
    )
    rows.extend(
        rows_from_result(
            source="synthetic",
            setting_id="block_rotation_recovered",
            n_models=3,
            width=4,
            seed=-1,
            block_size=2,
            triangle=(0, 1, 2),
            result=result,
            block_stats=summarize_block_alignment_stats(stats),
        )
    )

    reflection = np.array([[0.0, 1.0], [1.0, 0.0]])
    rot = rotation(0.4)
    noncentral = {
        (0, 0): np.eye(2),
        (1, 1): np.eye(2),
        (2, 2): np.eye(2),
        (0, 1): reflection,
        (1, 2): rot,
        (2, 0): np.linalg.inv(reflection) @ np.linalg.inv(rot),
    }
    result = StructureGroupLadderMerge(max_order=max_order).run(
        {"block_orthogonal": noncentral},
        n_models=3,
        width=2,
        candidate_lift_rank=2,
        triples=[(0, 1, 2)],
    )
    rows.extend(
        rows_from_result(
            source="synthetic",
            setting_id="block_noncentral_not_brauer",
            n_models=3,
            width=2,
            seed=-1,
            block_size=2,
            triangle=(0, 1, 2),
            result=result,
            block_stats={"mean_pairwise_block_residual": 0.0, "max_pairwise_block_residual": 0.0, "used_remainder_block": False},
        )
    )

    scalar = {
        (0, 0): np.eye(4),
        (1, 1): np.eye(4),
        (2, 2): np.eye(4),
        (0, 1): np.eye(4),
        (1, 2): np.eye(4),
        (2, 0): -np.eye(4),
    }
    result = StructureGroupLadderMerge(max_order=max_order).run(
        {"block_orthogonal": scalar},
        n_models=3,
        width=4,
        candidate_lift_rank=4,
        triples=[(0, 1, 2)],
    )
    rows.extend(
        rows_from_result(
            source="synthetic",
            setting_id="scalar_block_phase_detected",
            n_models=3,
            width=4,
            seed=-1,
            block_size=2,
            triangle=(0, 1, 2),
            result=result,
            block_stats={"mean_pairwise_block_residual": 0.0, "max_pairwise_block_residual": 0.0, "used_remainder_block": False},
        )
    )
    return rows


def train_real_models(args, spec, train_data, test_data, seed: int, n_models: int, width: int):
    device = device_from_arg(args.device)
    train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=seed + 311)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 997)
    models = []
    accuracies = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        model = make_model("mlp", spec, width)
        train_model(model, train_loader, args.epochs, args.lr, device)
        metrics = evaluate_model(model, test_loader, device)
        accuracies.append(float(metrics["accuracy"]))
        model.to("cpu")
        models.append(model)
    match_loader = make_loader(train_data, args.batch_size, shuffle=False, seed=seed + 501)
    activations = {
        idx: collect_features(model, match_loader, device)
        for idx, model in enumerate(models)
    }
    for model in models:
        model.to("cpu")
    return activations, float(np.mean(accuracies)), float(np.max(accuracies))


def run_real_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int) -> list[dict]:
    activations, mean_accuracy, max_accuracy = train_real_models(args, spec, train_data, test_data, seed, n_models, width)
    pairwise = estimate_pairwise_permutations_from_activations(activations, n_models, width)
    rows: list[dict] = []
    for block_size in parse_csv(args.block_sizes, int):
        if block_size > width:
            continue
        block_maps, block_stats_raw = estimate_block_orthogonal_alignments(
            pairwise,
            activations,
            n_models,
            width,
            block_size,
            allow_remainder=args.allow_remainder_block,
        )
        block_stats = summarize_block_alignment_stats(block_stats_raw)
        ladder = StructureGroupLadderMerge(max_order=args.max_order)
        setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}_B{block_size}"
        for triangle in combinations(range(n_models), 3):
            result = ladder.run(
                {"permutation": pairwise, "block_orthogonal": block_maps},
                n_models=n_models,
                width=width,
                activations=activations,
                candidate_lift_rank=width,
                triples=[tuple(triangle)],
            )
            rows.extend(
                rows_from_result(
                    source="real_mnist",
                    setting_id=setting_id,
                    n_models=n_models,
                    width=width,
                    seed=seed,
                    block_size=block_size,
                    triangle=tuple(triangle),
                    result=result,
                    block_stats=block_stats,
                    individual_accuracy_mean=mean_accuracy,
                    individual_accuracy_max=max_accuracy,
                )
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, block_size, level), group in df.groupby(["source", "block_size", "level"], dropna=False):
        central = group["supports_brauer_projective_interpretation"].fillna(False).astype(bool)
        noncentral = group["residual_type"].astype(str).str.contains("noncentral")
        not_eval = group["residual_type"].eq("not_evaluated")
        reductions = group["residual_type"].eq("block_gauge_reduces_residual")
        block_improvement = pd.to_numeric(group["centrality_improvement_from_permutation_to_block"], errors="coerce")
        residual_counts = group["residual_type"].value_counts(dropna=False)
        most_common = str(residual_counts.index[0]) if not residual_counts.empty else "none"
        if level == "block_orthogonal" and not central.any() and block_improvement.mean() > 0:
            interpretation = "block gauges reduce centrality but do not produce scalar/projective candidates"
        elif level == "block_orthogonal" and central.any():
            interpretation = "central/projective block candidates found; descriptive only"
        elif not_eval.all():
            interpretation = "not evaluated"
        elif noncentral.mean() >= 0.5:
            interpretation = "mostly noncentral holonomy"
        else:
            interpretation = "diagnostic baseline/control"
        rows.append(
            {
                "source": source,
                "block_size": int(block_size),
                "level": level,
                "n_rows": int(len(group)),
                "mean_centrality_score": float(pd.to_numeric(group["centrality_score"], errors="coerce").mean()),
                "min_centrality_score": float(pd.to_numeric(group["centrality_score"], errors="coerce").min()),
                "mean_cycle_score": float(pd.to_numeric(group["cycle_score"], errors="coerce").mean()),
                "fraction_block_reduces_residual": float(reductions.mean()),
                "fraction_central_projective_candidates": float(central.mean()),
                "fraction_noncentral_holonomy": float(noncentral.mean()),
                "mean_centrality_improvement_from_permutation_to_block": float(block_improvement.mean()),
                "mean_pairwise_block_residual": float(pd.to_numeric(group["mean_pairwise_block_residual"], errors="coerce").mean()),
                "max_pairwise_block_residual": float(pd.to_numeric(group["max_pairwise_block_residual"], errors="coerce").max()),
                "fraction_merge_evaluated": float(group["merge_evaluated"].fillna(False).astype(bool).mean()),
                "most_common_residual_type": most_common,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    return format_markdown_table(rows, columns)


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    synthetic_block = df[(df["source"] == "synthetic") & (df["level"] == "block_orthogonal")].to_dict("records")
    real_summary = summary[summary["source"] == "real_mnist"].to_dict("records")
    real_block_summary = summary[(summary["source"] == "real_mnist") & (summary["level"] == "block_orthogonal")].to_dict("records")
    real_block = df[(df["source"] == "real_mnist") & (df["level"] == "block_orthogonal")].copy()
    central_fraction = (
        float(real_block["supports_brauer_projective_interpretation"].fillna(False).astype(bool).mean())
        if not real_block.empty
        else 0.0
    )
    mean_improvement = (
        float(pd.to_numeric(real_block["centrality_improvement_from_permutation_to_block"], errors="coerce").mean())
        if not real_block.empty
        else float("nan")
    )
    if real_block.empty:
        interpretation = "Real MNIST block diagnostics were skipped."
    elif central_fraction > 0:
        interpretation = (
            "Some real MNIST block rows passed scalar/projective thresholds. These rows are descriptive and require independent verification."
        )
    elif np.isfinite(mean_improvement) and mean_improvement > 0:
        interpretation = (
            "Real block-orthogonal gauges reduce residual centrality on average, but no real MNIST block row passes scalar/root-of-unity thresholds."
        )
    else:
        interpretation = (
            "Real block-orthogonal gauges do not reduce residual centrality on average and produce no scalar/projective candidates."
        )

    control_cols = [
        "setting_id",
        "block_size",
        "residual_type",
        "centrality_score",
        "cycle_score",
        "phase_residual",
        "detected_order_d",
        "selected_resolution",
    ]
    summary_cols = [
        "block_size",
        "level",
        "n_rows",
        "mean_centrality_score",
        "min_centrality_score",
        "mean_cycle_score",
        "fraction_block_reduces_residual",
        "fraction_central_projective_candidates",
        "mean_centrality_improvement_from_permutation_to_block",
        "mean_pairwise_block_residual",
        "most_common_residual_type",
    ]
    block_cols = [
        "block_size",
        "n_rows",
        "mean_centrality_score",
        "min_centrality_score",
        "mean_cycle_score",
        "fraction_block_reduces_residual",
        "fraction_central_projective_candidates",
        "mean_centrality_improvement_from_permutation_to_block",
        "mean_pairwise_block_residual",
        "max_pairwise_block_residual",
        "fraction_merge_evaluated",
    ]

    report = f"""# Block-Orthogonal Ladder Report

This report is generated by `experiments/block_orthogonal_ladder_experiment.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State At Report Generation

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_worktree_dirty()}`

## Settings

- Dataset: MNIST
- Architecture: one-hidden-layer ReLU MLP
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Block sizes: `{args.block_sizes}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples: `{args.max_train_samples}`
- Test samples: `{args.max_test_samples}`
- Matching: activation
- Block partition: contiguous; remainder block allowed: `{args.allow_remainder_block}`

## Synthetic Controls

{table(synthetic_block, control_cols)}

## Real MNIST Block Summary

{table(real_block_summary, block_cols)}

## Real MNIST Summary By Level And Block Size

{table(real_summary, summary_cols)}

## Interpretation

{interpretation}

The block-orthogonal level is a feature-space diagnostic for ReLU MLPs.  A
general orthogonal rotation of ReLU hidden units is not an exact parameter
symmetry, so this run does not evaluate a block-orthogonal same-architecture
merged model.  C2M3/permutation alignment remains the merge baseline.

## Merge Performance

Block-orthogonal merge performance was not evaluated in this ReLU MLP run.
The CSVs preserve `merge_evaluated = False` for these rows.  This is deliberate:
without a linear/tanh architecture or a tested exact parameter transform, block
rotations should not be reported as single-model ReLU merges.

## Negative Boundaries

- This does not prove block-orthogonal ReLU transformations are exact model symmetries.
- This does not prove real neural residuals are Brauer/projective classes.
- This does not prove TwistedMerge++ beats C2M3 on real MNIST/CIFAR.
- This does not provide a capacity-matched block/projective lift.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1700,1701,1702,1703,1704")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--block-sizes", default="2,4,8")
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
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--allow-remainder-block", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    rows = controlled_rows(args.max_order)
    if not args.skip_real:
        spec, train_data, test_data = load_dataset(
            "mnist",
            args.data_dir,
            args.max_train_samples,
            args.max_test_samples,
            args.dataset_seed,
        )
        for seed in parse_csv(args.seeds, int):
            for n_models in parse_csv(args.model_counts, int):
                for width in parse_csv(args.widths, int):
                    rows.extend(run_real_setting(args, spec, train_data, test_data, seed, n_models, width))

    df = pd.DataFrame(rows)
    summary = summarize(df)
    csv_dir = args.reports_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "block_orthogonal_ladder.csv"
    summary_path = csv_dir / "block_orthogonal_ladder_summary.csv"
    report_path = args.reports_dir / "block_orthogonal_ladder_report.md"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(args, df, summary, report_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
