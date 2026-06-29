#!/usr/bin/env python
"""Run the synthetic MU(2) sign-cocycle obstruction experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cocycles import (  # noqa: E402
    estimate_mu2_gauges_spectral,
    mu2_edge_agreement,
    mu2_triangle_obstruction,
    sample_mu2_cocycle,
)
from src.metrics import capture_environment, mean_task_accuracy, pearsonr, save_json, summarize_by_level, flatten_columns  # noqa: E402
from src.plotting import plot_accuracy_vs_obstruction, save_latex_table  # noqa: E402
from src.synthetic_tasks import make_base_weight, make_local_datasets, make_mu2_local_weights  # noqa: E402
from src.twisted_merge import descended_mu2_merge, rank_lift_mu2_merge  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def run_once(args: argparse.Namespace, seed: int, flip_prob: float) -> dict:
    rng = np.random.default_rng(args.seed_offset + seed)
    cocycle = sample_mu2_cocycle(args.nodes, flip_prob, rng)
    obstruction = mu2_triangle_obstruction(args.nodes, cocycle.signs)
    base_weight = make_base_weight(args.dim, rng)
    local_weights = make_mu2_local_weights(base_weight, cocycle.true_gauges, rng, args.model_noise)
    datasets = make_local_datasets(
        local_weights,
        rng,
        n_val=args.val_samples,
        n_test=args.test_samples,
        label_noise=args.label_noise,
    )

    gauges_hat = estimate_mu2_gauges_spectral(args.nodes, cocycle.signs)
    gauge_accuracy = max(
        np.mean(gauges_hat == cocycle.true_gauges),
        np.mean(-gauges_hat == cocycle.true_gauges),
    )
    naive = descended_mu2_merge(local_weights, gauges_hat)
    lifted = rank_lift_mu2_merge(local_weights, gauges_hat, datasets.x_val, datasets.y_val)

    oracle_accuracy = mean_task_accuracy(datasets.x_test, datasets.y_test, local_weights)
    naive_accuracy = mean_task_accuracy(datasets.x_test, datasets.y_test, naive.node_weights)
    rank_lift_accuracy = mean_task_accuracy(datasets.x_test, datasets.y_test, lifted.node_weights)

    return {
        "experiment": "mu2",
        "seed": seed,
        "flip_prob": flip_prob,
        "nodes": args.nodes,
        "dim": args.dim,
        "model_noise": args.model_noise,
        "label_noise": args.label_noise,
        "n_flipped_edges": len(cocycle.flipped_edges),
        "edge_agreement": mu2_edge_agreement(gauges_hat, cocycle.signs),
        "gauge_accuracy": float(gauge_accuracy),
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
    parser.add_argument("--seed-offset", type=int, default=1701)
    parser.add_argument("--flip-probs", default="0.0,0.02,0.05,0.10,0.20,0.30,0.40")
    parser.add_argument("--model-noise", type=float, default=0.03)
    parser.add_argument("--label-noise", type=float, default=0.02)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    rows = []
    for flip_prob in parse_float_list(args.flip_probs):
        for seed in range(args.seeds):
            rows.append(run_once(args, seed, flip_prob))
    df = pd.DataFrame(rows)

    csv_path = args.reports_dir / "csv" / "synthetic_mu2_results.csv"
    table_path = args.reports_dir / "tables" / "synthetic_mu2_summary.tex"
    plot_path = args.reports_dir / "plots" / "synthetic_mu2_obstruction.png"
    config_path = args.reports_dir / "configs" / "synthetic_mu2_config.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    summary = flatten_columns(summarize_by_level(df, "flip_prob"))
    summary_csv = args.reports_dir / "csv" / "synthetic_mu2_summary.csv"
    summary.to_csv(summary_csv, index=False)
    save_latex_table(summary, table_path)
    plot_accuracy_vs_obstruction(csv_path, plot_path, "mu_2 obstruction vs merge accuracy")

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
