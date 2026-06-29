#!/usr/bin/env python
"""Planted cycle-obstruction benchmark for MNIST MLP model merging.

This experiment starts from one trained MLP per seed, creates functionally
equivalent hidden-permutation copies, then corrupts the observed pairwise
alignment maps.  The trained functions are fixed within a seed; only the
alignment observations change across inconsistency levels.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    compose_perm,
    cycle_score,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    format_markdown_table,
    invert_perm,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    rank_lifted_branch_models,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)


LEVEL_FRACTIONS = {
    "zero": 0.0,
    "low": 0.0625,
    "medium": 0.1875,
    "high": 0.375,
}


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def safe_corr(x, y, rank: bool = False) -> float:
    xv = pd.Series(list(x), dtype=float)
    yv = pd.Series(list(y), dtype=float)
    if rank:
        xv = xv.rank(method="average")
        yv = yv.rank(method="average")
    if len(xv) < 2 or float(xv.std(ddof=0)) == 0.0 or float(yv.std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv.to_numpy(), yv.to_numpy())[0, 1])


def monotone_non_decreasing(values: list[float], tolerance: float = 1e-9) -> bool:
    return all(values[idx + 1] + tolerance >= values[idx] for idx in range(len(values) - 1))


def disjoint_swap_involution(width: int, n_swaps: int) -> np.ndarray:
    perm = np.arange(width)
    for swap_idx in range(min(n_swaps, width // 2)):
        a = 2 * swap_idx
        b = a + 1
        perm[a], perm[b] = perm[b], perm[a]
    return perm


def random_swap_permutation(width: int, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(width)
    for _ in range(n_swaps):
        a, b = rng.choice(width, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    return perm


def true_pairwise_from_copy_perms(copy_perms: list[np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    inverses = [invert_perm(perm) for perm in copy_perms]
    pairwise = {}
    for i, j in product(range(len(copy_perms)), repeat=2):
        pairwise[(i, j)] = inverses[j][copy_perms[i]]
    return pairwise


def planted_pairwise(
    true_pairwise: dict[tuple[int, int], np.ndarray],
    n_models: int,
    width: int,
    defect_family: str,
    level: str,
    seed: int,
) -> tuple[dict[tuple[int, int], np.ndarray], int, str]:
    out = {pair: perm.copy() for pair, perm in true_pairwise.items()}
    fraction = LEVEL_FRACTIONS[level]
    n_swaps = int(round(fraction * width))
    if level == "zero" or n_swaps == 0:
        return out, 0, "none"

    if defect_family == "central_mu2":
        error = disjoint_swap_involution(width, n_swaps)
    elif defect_family == "random_noncentral":
        rng = np.random.default_rng(seed)
        error = random_swap_permutation(width, n_swaps, rng)
    else:
        raise ValueError(f"unknown defect family: {defect_family}")

    corrupted_edge = (0, 1)
    out[corrupted_edge] = compose_perm(error, true_pairwise[corrupted_edge])
    out[(1, 0)] = invert_perm(out[corrupted_edge])
    return out, n_swaps, "0-1"


def max_logit_disagreement(base_model, copies: list, loader, device, max_batches: int = 2) -> float:
    torch, _, _ = require_torch()
    base_model.to(device)
    base_model.eval()
    for model in copies:
        model.to(device)
        model.eval()
    max_diff = 0.0
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            base_logits = base_model(x)
            for model in copies:
                diff = (base_logits - model(x)).abs().max().detach().cpu()
                max_diff = max(max_diff, float(diff))
    return max_diff


def add_row(rows: list[dict], base: dict, baseline: str, metrics: dict, uses_alignment: bool, note: str) -> None:
    rows.append(
        {
            **base,
            "baseline": baseline,
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "merge_degradation": base["base_accuracy"] - metrics["accuracy"],
            "uses_planted_alignment": uses_alignment,
            "is_single_model": baseline not in {"ensemble_upper_bound"} and not baseline.startswith("twisted_rank_lift_"),
            "capacity_matched_to_weight_average": baseline not in {"ensemble_upper_bound"} and not baseline.startswith("twisted_rank_lift_"),
            "method_note": note,
        }
    )


def run_seed(args, seed: int) -> list[dict]:
    torch, _, _ = require_torch()
    device = device_from_arg(args.device)
    set_seed(seed)
    spec, train_data, test_data = load_dataset(
        "mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
    )
    train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=seed + 11)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=args.dataset_seed + 999)
    model = make_model("mlp", spec, args.width)
    train_model(model, train_loader, args.epochs, args.lr, device)
    base_metrics = evaluate_model(model, test_loader, device)
    model.to("cpu")

    rng = np.random.default_rng(seed + 100003)
    copy_perms = [rng.permutation(args.width) for _ in range(args.n_models)]
    copies = [
        permute_model_to_reference(model, "mlp", spec, args.width, perm)
        for perm in copy_perms
    ]
    copy_metrics = [evaluate_model(copy, test_loader, device) for copy in copies]
    copy_accuracy_std = float(np.std([metric["accuracy"] for metric in copy_metrics], ddof=0))
    logit_diff = max_logit_disagreement(model, copies, test_loader, device)
    true_pairwise = true_pairwise_from_copy_perms(copy_perms)
    true_score, _ = cycle_score(true_pairwise, args.n_models, args.width)

    rows = []
    for defect_family in parse_csv(args.defect_families, str):
        for level_idx, level in enumerate(parse_csv(args.levels, str)):
            planted, n_swaps, corrupted_edge = planted_pairwise(
                true_pairwise,
                args.n_models,
                args.width,
                defect_family,
                level,
                seed + level_idx * 1009,
            )
            planted_score, _cycle_rows = cycle_score(planted, args.n_models, args.width)
            _ref, synced, sync_disagreement = synchronize_permutations(planted, args.n_models)
            aligned_to_zero = [
                permute_model_to_reference(copy, "mlp", spec, args.width, planted[(0, idx)])
                for idx, copy in enumerate(copies)
            ]
            aligned_synced = [
                permute_model_to_reference(copy, "mlp", spec, args.width, synced[idx])
                for idx, copy in enumerate(copies)
            ]
            oracle_aligned = [
                permute_model_to_reference(copy, "mlp", spec, args.width, true_pairwise[(0, idx)])
                for idx, copy in enumerate(copies)
            ]
            weight_avg = average_models(copies, "mlp", spec, args.width)
            git_rebasin = average_models(aligned_to_zero, "mlp", spec, args.width)
            c2m3 = average_models(aligned_synced, "mlp", spec, args.width)
            oracle = average_models(oracle_aligned, "mlp", spec, args.width)
            branches = rank_lifted_branch_models(
                aligned_synced,
                planted,
                args.rank_lift_branches,
                "mlp",
                spec,
                args.width,
            )
            base = {
                "seed": seed,
                "dataset": "mnist",
                "architecture": "mlp",
                "n_models": args.n_models,
                "width": args.width,
                "defect_family": defect_family,
                "inconsistency_level": level,
                "level_order": level_idx,
                "n_planted_swaps": n_swaps,
                "corrupted_edge": corrupted_edge,
                "base_accuracy": base_metrics["accuracy"],
                "base_loss": base_metrics["loss"],
                "copy_accuracy_std": copy_accuracy_std,
                "max_logit_disagreement": logit_diff,
                "true_cycle_score": true_score,
                "planted_cycle_score": planted_score,
                "sync_disagreement": sync_disagreement,
            }
            add_row(rows, base, "weight_average", evaluate_model(weight_avg, test_loader, device), False, "single model, no alignment observations")
            add_row(rows, base, "git_rebasin_pairwise", evaluate_model(git_rebasin, test_loader, device), True, "single model, aligns to model 0 using planted pairwise maps")
            add_row(rows, base, "c2m3_cycle_consistent", evaluate_model(c2m3, test_loader, device), True, "single model, synchronized pairwise maps")
            add_row(
                rows,
                base,
                f"twisted_rank_lift_{len(branches)}",
                evaluate_ensemble(branches, test_loader, device),
                True,
                "branch ensemble / extra capacity",
            )
            add_row(rows, base, "ensemble_upper_bound", evaluate_ensemble(copies, test_loader, device), False, "ensemble of equivalent copies")
            add_row(rows, base, "oracle_true_alignment", evaluate_model(oracle, test_loader, device), False, "single model, uses uncorrupted true alignments")
    return rows


def summarize_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    levels = ["zero", "low", "medium", "high"]
    for (family, baseline), group in df.groupby(["defect_family", "baseline"]):
        mean_by_level = (
            group.groupby("inconsistency_level")["merge_degradation"]
            .mean()
            .reindex(levels)
        )
        values = [float(mean_by_level[level]) for level in levels]
        rows.append(
            {
                "stat_type": "baseline_trend",
                "defect_family": family,
                "baseline": baseline,
                "n_rows": int(len(group)),
                "pearson_cycle_vs_degradation": safe_corr(group["planted_cycle_score"], group["merge_degradation"]),
                "spearman_cycle_vs_degradation": safe_corr(group["planted_cycle_score"], group["merge_degradation"], rank=True),
                "monotone_mean_degradation": monotone_non_decreasing(values),
                "mean_degradation_zero": values[0],
                "mean_degradation_low": values[1],
                "mean_degradation_medium": values[2],
                "mean_degradation_high": values[3],
                "mean_accuracy": float(group["accuracy"].mean()),
                "note": "",
            }
        )

    wide = df.pivot_table(
        index=["seed", "defect_family", "inconsistency_level", "level_order"],
        columns="baseline",
        values=["accuracy", "merge_degradation", "planted_cycle_score"],
        aggfunc="first",
    )
    wide.columns = ["_".join(col).strip("_") for col in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    twist_cols = [col for col in wide.columns if col.startswith("accuracy_twisted_rank_lift_")]
    twist_acc_col = twist_cols[0] if twist_cols else None
    for family, group in wide.groupby("defect_family"):
        row = {
            "stat_type": "method_delta",
            "defect_family": family,
            "baseline": "method_delta_summary",
            "n_rows": int(len(group)),
            "pearson_cycle_vs_degradation": float("nan"),
            "spearman_cycle_vs_degradation": float("nan"),
            "monotone_mean_degradation": "",
            "mean_degradation_zero": float("nan"),
            "mean_degradation_low": float("nan"),
            "mean_degradation_medium": float("nan"),
            "mean_degradation_high": float("nan"),
            "mean_accuracy": float("nan"),
            "c2m3_accuracy_delta_vs_git": float((group["accuracy_c2m3_cycle_consistent"] - group["accuracy_git_rebasin_pairwise"]).mean()),
            "rank_lift_accuracy_delta_vs_git": float((group[twist_acc_col] - group["accuracy_git_rebasin_pairwise"]).mean()) if twist_acc_col else float("nan"),
            "rank_lift_accuracy_delta_vs_c2m3": float((group[twist_acc_col] - group["accuracy_c2m3_cycle_consistent"]).mean()) if twist_acc_col else float("nan"),
            "weight_average_cycle_slope_is_control": "weight_average does not use planted alignments",
            "note": "",
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in [
        "c2m3_accuracy_delta_vs_git",
        "rank_lift_accuracy_delta_vs_git",
        "rank_lift_accuracy_delta_vs_c2m3",
        "weight_average_cycle_slope_is_control",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def plot_results(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    baselines = ["weight_average", "git_rebasin_pairwise", "c2m3_cycle_consistent", "twisted_rank_lift_2", "oracle_true_alignment"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, (family, panel) in zip(axes, df[df["baseline"].isin(baselines)].groupby("defect_family")):
        for baseline, group in panel.groupby("baseline"):
            summary = group.groupby("inconsistency_level").agg(
                planted_cycle_score=("planted_cycle_score", "mean"),
                merge_degradation=("merge_degradation", "mean"),
            )
            summary = summary.reindex(["zero", "low", "medium", "high"])
            ax.plot(summary["planted_cycle_score"], summary["merge_degradation"], marker="o", label=baseline)
        ax.set_title(family)
        ax.set_xlabel("planted cycle obstruction score")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("base accuracy minus merged accuracy")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def claim_table(stats: pd.DataFrame) -> list[dict]:
    trend = stats[stats["stat_type"] == "baseline_trend"]
    deltas = stats[stats["stat_type"] == "method_delta"].set_index("defect_family")
    git = trend[trend["baseline"] == "git_rebasin_pairwise"].set_index("defect_family")
    c2m3 = trend[trend["baseline"] == "c2m3_cycle_consistent"].set_index("defect_family")
    weight = trend[trend["baseline"] == "weight_average"].set_index("defect_family")

    rows = []
    for family in sorted(git.index):
        git_row = git.loc[family]
        c2m3_delta = float(deltas.loc[family, "c2m3_accuracy_delta_vs_git"])
        rank_delta_git = float(deltas.loc[family, "rank_lift_accuracy_delta_vs_git"])
        rank_delta_c2m3 = float(deltas.loc[family, "rank_lift_accuracy_delta_vs_c2m3"])
        git_high = float(git_row["mean_degradation_high"])
        c2m3_high = float(c2m3.loc[family, "mean_degradation_high"])
        rows.append(
            {
                "claim": f"{family}: planted score monotonically predicts Git-ReBasin degradation",
                "status": "supported_descriptive" if bool(git_row["monotone_mean_degradation"]) and git_row["spearman_cycle_vs_degradation"] > 0.8 else "unsupported",
                "evidence": f"spearman={git_row['spearman_cycle_vs_degradation']:.4f}, monotone={git_row['monotone_mean_degradation']}",
            }
        )
        rows.append(
            {
                "claim": f"{family}: cycle-consistent synchronization fixes planted inconsistency",
                "status": "supported_descriptive" if c2m3_high <= 0.005 and git_high - c2m3_high > 0.01 else "unsupported",
                "evidence": f"high-defect Git degradation={git_high:.4f}, C2M3 degradation={c2m3_high:.4f}, mean delta={c2m3_delta:.4f}",
            }
        )
        rows.append(
            {
                "claim": f"{family}: rank-lift branch adds benefit beyond C2M3",
                "status": "supported_descriptive" if rank_delta_c2m3 > 0.02 else "unsupported",
                "evidence": f"rank-lift delta vs Git={rank_delta_git:.4f}, vs C2M3={rank_delta_c2m3:.4f}",
            }
        )
    constant_weight = True
    for _family, row in weight.iterrows():
        vals = [
            float(row["mean_degradation_zero"]),
            float(row["mean_degradation_low"]),
            float(row["mean_degradation_medium"]),
            float(row["mean_degradation_high"]),
        ]
        constant_weight = constant_weight and max(vals) - min(vals) <= 1e-9
    rows.append(
        {
            "claim": "weight averaging is a negative control for planted alignment inconsistency",
            "status": "supported" if constant_weight else "mixed",
            "evidence": "mean degradation is constant across planted levels; weight averaging does not read planted pairwise alignments",
        }
    )
    if {"central_mu2", "random_noncentral"}.issubset(deltas.index):
        central_rank = float(deltas.loc["central_mu2", "rank_lift_accuracy_delta_vs_c2m3"])
        random_rank = float(deltas.loc["random_noncentral", "rank_lift_accuracy_delta_vs_c2m3"])
        rows.append(
            {
                "claim": "rank-lift branch helps only for central/twist-like planted defects",
                "status": "supported_descriptive" if central_rank > 0.02 and random_rank <= 0.005 else "unsupported",
                "evidence": f"delta vs C2M3: central={central_rank:.4f}, random={random_rank:.4f}",
            }
        )
        rows.append(
            {
                "claim": "non-central/random defects are not evidence for a central-twist-specific rank-lift mechanism",
                "status": "supported",
                "evidence": f"random rank-lift delta vs C2M3={random_rank:.4f}",
            }
        )
    return rows


def write_report(args, df: pd.DataFrame, stats: pd.DataFrame, path: Path) -> None:
    claims = claim_table(stats)
    base_unique = df[["seed", "base_accuracy", "max_logit_disagreement", "copy_accuracy_std"]].drop_duplicates()
    base_rows = [
        {"metric": "base_accuracy_mean", "value": float(base_unique["base_accuracy"].mean())},
        {"metric": "base_accuracy_min", "value": float(base_unique["base_accuracy"].min())},
        {"metric": "base_accuracy_max", "value": float(base_unique["base_accuracy"].max())},
        {"metric": "max_logit_disagreement_max", "value": float(base_unique["max_logit_disagreement"].max())},
        {"metric": "copy_accuracy_std_max", "value": float(base_unique["copy_accuracy_std"].max())},
    ]
    trend_rows = stats[stats["stat_type"] == "baseline_trend"].to_dict("records")
    trend_columns = [
        "defect_family",
        "baseline",
        "spearman_cycle_vs_degradation",
        "monotone_mean_degradation",
        "mean_degradation_zero",
        "mean_degradation_low",
        "mean_degradation_medium",
        "mean_degradation_high",
    ]
    delta_rows = stats[stats["stat_type"] == "method_delta"].to_dict("records")
    delta_columns = [
        "defect_family",
        "c2m3_accuracy_delta_vs_git",
        "rank_lift_accuracy_delta_vs_git",
        "rank_lift_accuracy_delta_vs_c2m3",
    ]
    report = f"""# Planted Obstruction Model-Merging Report

This report is generated by `experiments/planted_obstruction_model_merging.py`.

## Exact Command

```bash
{args.command_string}
```

## Construction

- Train one MNIST one-hidden-layer MLP per seed.
- Create `{args.n_models}` functionally equivalent copies by permuting hidden units and classifier input columns.
- Use the exact true pairwise permutations as the zero-inconsistency condition.
- Plant one corrupted observed edge `(0, 1)` at low, medium, and high levels.
- `central_mu2` uses a fixed disjoint-swap involution, a controlled `mu_2`-like subgroup for this benchmark.
- `random_noncentral` uses random transpositions of the same approximate size and is treated as the non-central negative-control family.

## Outputs

- Results CSV: `reports/csv/planted_obstruction_model_merging.csv`
- Stats CSV: `reports/csv/planted_obstruction_stats.csv`
- Plot: `reports/plots/planted_cycle_score_vs_degradation.pdf`

## Base Model And Copy Sanity

{format_markdown_table(base_rows, ["metric", "value"])}

The copy sanity rows check that hidden-permutation copies are functionally equivalent before merging.  Nonzero merge degradation after using planted observations is therefore caused by the merge rule and the planted alignment data, not by training different functions.

## Claim Table

{format_markdown_table(claims, ["claim", "status", "evidence"])}

## Trend Summary

{format_markdown_table(trend_rows, trend_columns)}

## Method Delta Summary

{format_markdown_table(delta_rows, delta_columns)}

## Interpretation

- `weight_average` is a negative control for planted pairwise alignments because it does not use them.
- `git_rebasin_pairwise` is expected to degrade with planted cycle score because it aligns all copies through model 0 and therefore trusts the corrupted `(0, 1)` edge.
- `c2m3_cycle_consistent` can fix this planted design when synchronization chooses an uncorrupted reference/gauge.
- `twisted_rank_lift_2` is a branch ensemble with extra capacity, not a single merged model.  Its results must not be described as capacity-matched single-model wins.
- If random non-central defects are also helped, that is not evidence for a central-twist-specific mechanism.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1200,1201,1202,1203,1204,1205,1206,1207,1208,1209")
    parser.add_argument("--n-models", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank-lift-branches", type=int, default=2)
    parser.add_argument("--levels", default="zero,low,medium,high")
    parser.add_argument("--defect-families", default="central_mu2,random_noncentral")
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    rows = []
    for seed in parse_csv(args.seeds, int):
        rows.extend(run_seed(args, seed))

    df = pd.DataFrame(rows)
    stats = summarize_stats(df)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "planted_obstruction_model_merging.csv"
    stats_path = csv_dir / "planted_obstruction_stats.csv"
    plot_path = plot_dir / "planted_cycle_score_vs_degradation.pdf"
    report_path = args.reports_dir / "planted_obstruction_model_merging_report.md"
    df.to_csv(results_path, index=False)
    stats.to_csv(stats_path, index=False)
    plot_results(df, plot_path)
    save_json(
        args.reports_dir / "configs" / "planted_obstruction_model_merging_config.json",
        {
            "argv": sys.argv,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "command_string"
            },
            "environment": capture_environment(),
        },
    )
    write_report(args, df, stats, report_path)
    print(f"wrote {results_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
