#!/usr/bin/env python
"""Run the synthetic U(1) phase-cocycle obstruction experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alignment import wrap_angle  # noqa: E402
from src.cocycles import (  # noqa: E402
    estimate_u1_phases_spectral,
    sample_u1_cocycle,
    u1_edge_residual,
    u1_triangle_obstruction,
)
from src.metrics import capture_environment, mean_task_accuracy, pearsonr, save_json, summarize_by_level, flatten_columns  # noqa: E402
from src.plotting import plot_accuracy_vs_obstruction, save_latex_table  # noqa: E402
from src.synthetic_tasks import make_base_weight, make_local_datasets, make_u1_local_weights  # noqa: E402
from src.twisted_merge import descended_u1_merge, rank_lift_u1_merge  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def run_once(args: argparse.Namespace, seed: int, noise_std: float) -> dict:
    rng = np.random.default_rng(args.seed_offset + seed)
    cocycle = sample_u1_cocycle(args.nodes, noise_std, rng)
    obstruction = u1_triangle_obstruction(args.nodes, cocycle.phases)
    base_weight = make_base_weight(args.dim, rng)
    local_weights = make_u1_local_weights(base_weight, cocycle.true_phases, rng, args.model_noise)
    datasets = make_local_datasets(
        local_weights,
        rng,
        n_val=args.val_samples,
        n_test=args.test_samples,
        label_noise=args.label_noise,
    )

    phases_hat = estimate_u1_phases_spectral(args.nodes, cocycle.phases)
    phase_error = np.mean(np.abs(wrap_angle(phases_hat - cocycle.true_phases)))
    naive = descended_u1_merge(local_weights, phases_hat)
    lifted = rank_lift_u1_merge(local_weights, phases_hat, datasets.x_val, datasets.y_val, n_branches=args.branches)

    oracle_accuracy = mean_task_accuracy(datasets.x_test, datasets.y_test, local_weights)
    naive_accuracy = mean_task_accuracy(datasets.x_test, datasets.y_test, naive.node_weights)
    rank_lift_accuracy = mean_task_accuracy(datasets.x_test, datasets.y_test, lifted.node_weights)

    return {
        "experiment": "u1",
        "seed": seed,
        "noise_std": noise_std,
        "nodes": args.nodes,
        "dim": args.dim,
        "branches": args.branches,
        "model_noise": args.model_noise,
        "label_noise": args.label_noise,
        "edge_residual": u1_edge_residual(phases_hat, cocycle.phases),
        "phase_error": float(phase_error),
        **obstruction,
        "oracle_accuracy": oracle_accuracy,
        "naive_accuracy": naive_accuracy,
        "rank_lift_accuracy": rank_lift_accuracy,
        "naive_failure": oracle_accuracy - naive_accuracy,
        "rank_lift_gain": rank_lift_accuracy - naive_accuracy,
        "rank_lift_branch_count": int(len(np.unique(lifted.assignments))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-offset", type=int, default=2603)
    parser.add_argument("--noise-stds", default="0.0,0.05,0.10,0.20,0.40,0.80,1.20")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--model-noise", type=float, default=0.03)
    parser.add_argument("--label-noise", type=float, default=0.02)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    rows = []
    for noise_std in parse_float_list(args.noise_stds):
        for seed in range(args.seeds):
            rows.append(run_once(args, seed, noise_std))
    df = pd.DataFrame(rows)

    csv_path = args.reports_dir / "csv" / "synthetic_u1_results.csv"
    table_path = args.reports_dir / "tables" / "synthetic_u1_summary.tex"
    plot_path = args.reports_dir / "plots" / "synthetic_u1_obstruction.png"
    config_path = args.reports_dir / "configs" / "synthetic_u1_config.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    summary = flatten_columns(summarize_by_level(df, "noise_std"))
    summary_csv = args.reports_dir / "csv" / "synthetic_u1_summary.csv"
    summary.to_csv(summary_csv, index=False)
    save_latex_table(summary, table_path)
    plot_accuracy_vs_obstruction(csv_path, plot_path, "U(1) obstruction vs merge accuracy")

    save_json(
        config_path,
        {
            "argv": sys.argv,
            "args": vars(args) | {"reports_dir": str(args.reports_dir)},
            "environment": capture_environment(),
            "correlations": {
                "obstruction_vs_naive_failure": pearsonr(df["obstruction_score"], df["naive_failure"]),
                "obstruction_vs_rank_lift_gain": pearsonr(df["obstruction_score"], df["rank_lift_gain"]),
            },
            "outputs": {
                "csv": str(csv_path),
                "summary_csv": str(summary_csv),
                "table": str(table_path),
                "plot": str(plot_path),
            },
        },
    )
    print(f"wrote {csv_path}")
    print(f"wrote {summary_csv}")
    print(f"wrote {table_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
