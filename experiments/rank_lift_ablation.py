#!/usr/bin/env python
"""Ablate the number of rank-lift branches in synthetic experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cocycles import estimate_mu2_gauges_spectral, estimate_u1_phases_spectral, sample_mu2_cocycle, sample_u1_cocycle  # noqa: E402
from src.metrics import capture_environment, mean_task_accuracy, save_json  # noqa: E402
from src.plotting import plot_rank_ablation, save_latex_table  # noqa: E402
from src.synthetic_tasks import make_base_weight, make_local_datasets, make_mu2_local_weights, make_u1_local_weights  # noqa: E402
from src.twisted_merge import descended_mu2_merge, descended_u1_merge, rank_lift_mu2_merge, rank_lift_u1_merge  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def run_mu2(args: argparse.Namespace, seed: int, rank: int) -> dict:
    rng = np.random.default_rng(args.seed_offset + 1000 + seed)
    cocycle = sample_mu2_cocycle(args.nodes, args.mu2_flip_prob, rng)
    base_weight = make_base_weight(args.dim, rng)
    local_weights = make_mu2_local_weights(base_weight, cocycle.true_gauges, rng, args.model_noise)
    datasets = make_local_datasets(local_weights, rng, args.val_samples, args.test_samples, args.label_noise)
    gauges_hat = estimate_mu2_gauges_spectral(args.nodes, cocycle.signs)
    if rank <= 1:
        merge = descended_mu2_merge(local_weights, gauges_hat)
        effective_rank = 1
    else:
        merge = rank_lift_mu2_merge(local_weights, gauges_hat, datasets.x_val, datasets.y_val)
        effective_rank = 2
    return {
        "experiment": "mu2",
        "seed": seed,
        "rank": rank,
        "effective_rank": effective_rank,
        "accuracy": mean_task_accuracy(datasets.x_test, datasets.y_test, merge.node_weights),
    }


def run_u1(args: argparse.Namespace, seed: int, rank: int) -> dict:
    rng = np.random.default_rng(args.seed_offset + 2000 + seed)
    cocycle = sample_u1_cocycle(args.nodes, args.u1_noise_std, rng)
    base_weight = make_base_weight(args.dim, rng)
    local_weights = make_u1_local_weights(base_weight, cocycle.true_phases, rng, args.model_noise)
    datasets = make_local_datasets(local_weights, rng, args.val_samples, args.test_samples, args.label_noise)
    phases_hat = estimate_u1_phases_spectral(args.nodes, cocycle.phases)
    if rank <= 1:
        merge = descended_u1_merge(local_weights, phases_hat)
    else:
        merge = rank_lift_u1_merge(local_weights, phases_hat, datasets.x_val, datasets.y_val, n_branches=rank)
    return {
        "experiment": "u1",
        "seed": seed,
        "rank": rank,
        "effective_rank": rank,
        "accuracy": mean_task_accuracy(datasets.x_test, datasets.y_test, merge.node_weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=3901)
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--mu2-flip-prob", type=float, default=0.25)
    parser.add_argument("--u1-noise-std", type=float, default=0.8)
    parser.add_argument("--model-noise", type=float, default=0.03)
    parser.add_argument("--label-noise", type=float, default=0.02)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    rows = []
    for rank in parse_int_list(args.ranks):
        for seed in range(args.seeds):
            rows.append(run_mu2(args, seed, rank))
            rows.append(run_u1(args, seed, rank))
    df = pd.DataFrame(rows)
    csv_path = args.reports_dir / "csv" / "rank_lift_ablation.csv"
    table_path = args.reports_dir / "tables" / "rank_lift_ablation.tex"
    plot_path = args.reports_dir / "plots" / "rank_lift_ablation.png"
    config_path = args.reports_dir / "configs" / "rank_lift_ablation_config.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    summary = df.groupby(["experiment", "rank", "effective_rank"])["accuracy"].agg(["mean", "std"]).reset_index()
    summary_csv = args.reports_dir / "csv" / "rank_lift_ablation_summary.csv"
    summary.to_csv(summary_csv, index=False)
    save_latex_table(summary, table_path)
    plot_rank_ablation(csv_path, plot_path)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "args": vars(args) | {"reports_dir": str(args.reports_dir)},
            "environment": capture_environment(),
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
