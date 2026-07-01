#!/usr/bin/env python
"""Prompt 21 preregistered replication of the narrow raw-obstruction row.

This script reuses the fixed-setting verifier implementation but writes
Prompt-21-specific artifacts so the Prompt 11 outputs are not overwritten.
"""

from __future__ import annotations

import argparse
import gzip
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.model_merging_fixed_setting_verification as verifier  # noqa: E402


RUN_BULKY_COLUMNS = (
    "pairwise_alignment_permutations_json",
    "layerwise_alignment_permutations_json",
)
TRIANGLE_BULKY_COLUMNS = (
    "pairwise_alignment_permutations_json",
    "layerwise_alignment_permutations_json",
    "p_ij",
    "p_jk",
    "p_ki",
    "triangle_perm",
)
KEY_COLUMNS = (
    "setting_id",
    "run_id",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "alignment_source",
    "seed",
    "method",
    "triangle",
)


@dataclass(frozen=True)
class ReplicationSetting:
    role: str
    dataset: str
    architecture: str
    n_models: int
    width: int
    domain_shift: str
    matching: str
    description: str

    @property
    def key(self) -> tuple:
        return (self.dataset, self.architecture, self.n_models, self.width, self.domain_shift, self.matching)


SETTINGS = (
    ReplicationSetting(
        "primary_replication",
        "fashion_mnist",
        "mlp2",
        3,
        128,
        "none",
        "activation",
        "Prompt 11 supported row: Fashion-MNIST, mlp2, N=3, W=128, no shift, activation matching.",
    ),
    ReplicationSetting(
        "neighbor_control",
        "fashion_mnist",
        "mlp2",
        3,
        128,
        "none",
        "weight",
        "Same primary setting with weight matching.",
    ),
    ReplicationSetting(
        "neighbor_control",
        "fashion_mnist",
        "mlp2",
        4,
        128,
        "none",
        "activation",
        "Fashion-MNIST N=4 no-shift activation control.",
    ),
    ReplicationSetting(
        "neighbor_control",
        "fashion_mnist",
        "mlp2",
        3,
        128,
        "input_noise",
        "activation",
        "Fashion-MNIST N=3 input-noise activation control.",
    ),
    ReplicationSetting(
        "neighbor_control",
        "mnist",
        "mlp2",
        3,
        128,
        "none",
        "activation",
        "MNIST N=3 no-shift activation control.",
    ),
)


def setting_role_map() -> dict[tuple, str]:
    return {setting.key: setting.role for setting in SETTINGS}


def setting_description_map() -> dict[tuple, str]:
    return {setting.key: setting.description for setting in SETTINGS}


def decorate_roles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    roles = setting_role_map()
    descriptions = setting_description_map()
    out = df.copy()
    role_values = []
    desc_values = []
    for row in out.itertuples(index=False):
        key = (
            getattr(row, "dataset", ""),
            getattr(row, "architecture", ""),
            int(getattr(row, "n_models", 0)),
            int(getattr(row, "width", 0)),
            getattr(row, "domain_shift", ""),
            getattr(row, "matching", ""),
        )
        role_values.append(roles.get(key, "outside_preregistration"))
        desc_values.append(descriptions.get(key, ""))
    out["preregistration_role"] = role_values
    out["setting_description"] = desc_values
    return out


def bootstrap_sign_p_like(x_values, y_values, corr_fn, n_boot: int, seed: int) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or n_boot <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    non_positive = 0
    finite_count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        value = corr_fn(x[idx], y[idx])
        if math.isfinite(value):
            finite_count += 1
            if value <= 0.0:
                non_positive += 1
    if finite_count == 0:
        return float("nan")
    return float((non_positive + 1) / (finite_count + 1))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    q_values = [float("nan")] * len(p_values)
    finite = [(idx, value) for idx, value in enumerate(p_values) if math.isfinite(value)]
    if not finite:
        return q_values
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running = 1.0
    for rank_from_end, (idx, value) in enumerate(reversed(finite), start=1):
        rank = m - rank_from_end + 1
        running = min(running, value * m / rank)
        q_values[idx] = float(min(running, 1.0))
    return q_values


def claim_decision(row: pd.Series) -> str:
    role = str(row.get("preregistration_role", ""))
    supported = str(row.get("claim_status", "")) == "supported_fixed_setting_observed"
    if role == "primary_replication":
        return "replicated_supported" if supported else "failed_replication"
    if role == "neighbor_control":
        return "exploratory_only" if supported else "setting_specific_only"
    return "exploratory_only"


def build_multiple_testing_table(runs: pd.DataFrame, stats: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    if runs.empty or stats.empty:
        return pd.DataFrame()
    base = runs[(runs["method"] == "weight_average") & (runs["alignment_source"] == "observed")].copy()
    group_cols = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
    rows = []
    for idx, (key, group) in enumerate(base.groupby(group_cols, dropna=False)):
        meta = dict(zip(group_cols, key))
        x = pd.to_numeric(group["cycle_score"], errors="coerce").to_numpy()
        y = pd.to_numeric(group["single_best_merge_degradation"], errors="coerce").to_numpy()
        rows.append(
            {
                **meta,
                "n_rows": int(len(group)),
                "n_unique_seeds": int(group["seed"].nunique()),
                "pearson": verifier.safe_pearson(x, y),
                "spearman": verifier.safe_spearman(x, y),
                "pearson_nonpositive_p_like": bootstrap_sign_p_like(
                    x,
                    y,
                    verifier.safe_pearson,
                    n_boot,
                    seed=91001 + idx * 17,
                ),
                "spearman_nonpositive_p_like": bootstrap_sign_p_like(
                    x,
                    y,
                    verifier.safe_spearman,
                    n_boot,
                    seed=92003 + idx * 17,
                ),
            }
        )
    table = decorate_roles(pd.DataFrame(rows))
    if table.empty:
        return table
    table["joint_sign_p_like"] = table[["pearson_nonpositive_p_like", "spearman_nonpositive_p_like"]].max(axis=1)
    table["pearson_bh_q"] = benjamini_hochberg(table["pearson_nonpositive_p_like"].astype(float).tolist())
    table["joint_sign_bh_q"] = benjamini_hochberg(table["joint_sign_p_like"].astype(float).tolist())

    corr_stats = stats[
        (stats["alignment_source"].astype(str) == "observed")
        & (stats["claim_status"].astype(str) != "method_summary_not_obstruction_correlation")
    ].copy()
    merge_cols = group_cols + [
        "fixed_setting_id",
        "mean_cycle_score",
        "mean_weight_merge_degradation",
        "pearson_ci_low",
        "pearson_ci_high",
        "spearman_ci_low",
        "spearman_ci_high",
        "claim_status",
        "claim_supported",
    ]
    for col in merge_cols:
        if col not in corr_stats:
            corr_stats[col] = np.nan
    table = table.merge(corr_stats[merge_cols], on=group_cols, how="left")
    table["claim_decision"] = table.apply(claim_decision, axis=1)
    return table


def _open_gzip_csv(path: Path) -> io.TextIOWrapper:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(gz, encoding="utf-8", newline="")


def write_compact_dataframe(
    df: pd.DataFrame,
    path: Path,
    shard_dir: Path,
    shard_prefix: str,
    bulky_columns: tuple[str, ...],
    rows_per_shard: int,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    present = [column for column in bulky_columns if column in df.columns]
    if df.empty or not present:
        df.to_csv(path, index=False, lineterminator="\n")
        return {
            "source_csv": str(path),
            "rows": int(len(df)),
            "shard_count": 0,
            "columns_moved": "",
            "max_shard_bytes": 0,
        }

    slim = df.drop(columns=present).copy()
    row_numbers = np.arange(1, len(slim) + 1)
    slim["large_field_shard"] = [f"{shard_prefix}_part_{(row - 1) // rows_per_shard:03d}.csv.gz" for row in row_numbers]
    slim["large_field_row"] = row_numbers
    slim.to_csv(path, index=False, lineterminator="\n")

    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob(f"{shard_prefix}_part_*.csv.gz"):
        old.unlink()
    key_cols = [col for col in KEY_COLUMNS if col in df.columns]
    shard_sizes = []
    for shard_idx, start in enumerate(range(0, len(df), rows_per_shard)):
        end = min(start + rows_per_shard, len(df))
        shard_path = shard_dir / f"{shard_prefix}_part_{shard_idx:03d}.csv.gz"
        shard = df.iloc[start:end][key_cols + present].copy()
        shard.insert(0, "large_field_row", row_numbers[start:end])
        with _open_gzip_csv(shard_path) as handle:
            shard.to_csv(handle, index=False, lineterminator="\n")
        shard_sizes.append(shard_path.stat().st_size)
    return {
        "source_csv": str(path),
        "rows": int(len(df)),
        "shard_count": len(shard_sizes),
        "columns_moved": ",".join(present),
        "max_shard_bytes": max(shard_sizes) if shard_sizes else 0,
    }


def plot_prompt21(runs: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    base = runs[(runs["method"] == "weight_average") & (runs["alignment_source"] == "observed")].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    if base.empty:
        ax.text(0.5, 0.5, "No observed Prompt 21 rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        group_cols = ["dataset", "n_models", "domain_shift", "matching"]
        colors = {
            ("fashion_mnist", 3, "none", "activation"): "tab:green",
            ("fashion_mnist", 3, "none", "weight"): "tab:blue",
            ("fashion_mnist", 4, "none", "activation"): "tab:orange",
            ("fashion_mnist", 3, "input_noise", "activation"): "tab:red",
            ("mnist", 3, "none", "activation"): "tab:purple",
        }
        for key, group in base.groupby(group_cols, dropna=False):
            label = f"{key[0]} N={key[1]} {key[2]} {key[3]}"
            ax.scatter(
                pd.to_numeric(group["cycle_score"], errors="coerce"),
                pd.to_numeric(group["single_best_merge_degradation"], errors="coerce"),
                s=20,
                alpha=0.72,
                color=colors.get(key, "tab:gray"),
                label=label,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Cycle obstruction score")
        ax.set_ylabel("Weight-average degradation vs best single")
        ax.set_title("Prompt 21 preregistered replication")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    args,
    stats: pd.DataFrame,
    individuals: pd.DataFrame,
    multiple: pd.DataFrame,
    manifest: pd.DataFrame,
    path: Path,
) -> None:
    observed = stats[
        (stats["alignment_source"].astype(str) == "observed")
        & (stats["claim_status"].astype(str) != "method_summary_not_obstruction_correlation")
    ].copy()
    observed = decorate_roles(observed)
    if not observed.empty:
        observed["claim_decision"] = observed.apply(claim_decision, axis=1)
    primary = observed[observed["preregistration_role"] == "primary_replication"].copy()
    controls = observed[observed["preregistration_role"] == "neighbor_control"].copy()

    quality = pd.DataFrame()
    if not individuals.empty:
        quality = (
            decorate_roles(individuals)
            .groupby(
                [
                    "preregistration_role",
                    "dataset",
                    "architecture",
                    "n_models",
                    "width",
                    "domain_shift",
                    "matching",
                ],
                dropna=False,
            )["test_accuracy"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
        )

    primary_decision = "failed_replication"
    if not primary.empty and (primary["claim_decision"].astype(str) == "replicated_supported").any():
        primary_decision = "replicated_supported"

    primary_seeds = verifier.parse_seeds(args.seeds)
    control_seeds = verifier.parse_seeds(args.control_seeds)
    report = f"""# Prompt 21 Replication Report

Generated by `experiments/prompt21_replication.py`.

## Exact Command

```bash
{args.command_string}
```

## Preregistered Design

Primary replication row: Fashion-MNIST, `mlp2`, `N=3`, width `128`, no shift, activation matching, fresh seeds `{verifier.summarize_seed_list(primary_seeds)}`.

Neighbor controls:

- Fashion-MNIST `N=3`, no shift, weight matching, using the same `{len(primary_seeds)}` trained-seed bundle as the primary row.
- Fashion-MNIST `N=4`, no shift, activation matching, seeds `{verifier.summarize_seed_list(control_seeds)}`.
- Fashion-MNIST `N=3`, input-noise shift, activation matching, seeds `{verifier.summarize_seed_list(control_seeds)}`.
- MNIST `N=3`, no shift, activation matching, seeds `{verifier.summarize_seed_list(control_seeds)}`.

No fake or smoke rows are generated. Checkpoints are not saved by default.

## Claim Gate

The same raw fixed-setting gate is used as Prompt 11: at least 20 observed rows, positive Pearson and Spearman correlations, and a positive bootstrap Pearson lower bound. The primary replication decision is separated from neighbor-control decisions. Multiple-testing quantities are reported as an audit aid, not as a way to promote exploratory controls.

Primary decision: `{primary_decision}`.

## Outputs

- `reports/csv/prompt21_replication_runs.csv`
- `reports/csv/prompt21_replication_stats.csv`
- `reports/csv/prompt21_replication_multiple_testing.csv`
- `reports/csv/prompt21_replication_individual_models.csv`
- `reports/csv/prompt21_replication_triangle_defects.csv`
- `reports/csv/prompt21_large_artifacts_manifest.csv`
- `reports/plots/prompt21_cycle_vs_degradation.pdf`
- `reports/prompt21_replication_report.md`

## Local Model Quality

{verifier.md_table(quality, ["preregistration_role", "dataset", "architecture", "n_models", "width", "domain_shift", "matching", "count", "mean", "min", "max"], 20)}

## Primary Replication Row

{verifier.md_table(primary, ["preregistration_role", "dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_rows", "n_unique_seeds", "mean_cycle_score", "mean_weight_merge_degradation", "pearson_cycle_vs_weight_degradation", "pearson_ci_low", "pearson_ci_high", "spearman_cycle_vs_weight_degradation", "spearman_ci_low", "spearman_ci_high", "claim_status", "claim_decision"], 10)}

## Neighbor Controls

{verifier.md_table(controls, ["preregistration_role", "dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_rows", "n_unique_seeds", "mean_cycle_score", "mean_weight_merge_degradation", "pearson_cycle_vs_weight_degradation", "pearson_ci_low", "pearson_ci_high", "spearman_cycle_vs_weight_degradation", "spearman_ci_low", "spearman_ci_high", "claim_status", "claim_decision"], 20)}

## Multiple-Testing And Sign-Gate Audit

`pearson_nonpositive_p_like` and `spearman_nonpositive_p_like` are bootstrap sign frequencies with add-one smoothing. `joint_sign_p_like` is the max of the two. Benjamini-Hochberg q-values are reported across the preregistered primary row plus neighbor controls.

{verifier.md_table(multiple, ["preregistration_role", "dataset", "architecture", "n_models", "width", "domain_shift", "matching", "n_unique_seeds", "pearson", "pearson_ci_low", "pearson_ci_high", "spearman", "spearman_ci_low", "spearman_ci_high", "pearson_nonpositive_p_like", "spearman_nonpositive_p_like", "joint_sign_p_like", "pearson_bh_q", "joint_sign_bh_q", "claim_decision"], 20)}

## Large-Field Manifest

The per-row scalar CSVs are compact. Bulky permutation/map fields, if present, are stored in deterministic gzip shards and linked by `large_field_shard` and `large_field_row`.

{verifier.md_table(manifest, ["source_csv", "rows", "shard_count", "columns_moved", "max_shard_bytes"], 20)}

## Boundary

This report is a replication and stability audit only. It does not update the paper, does not claim external-baseline superiority, and does not turn neighbor-control positives into a broad raw-obstruction claim.
"""
    path.write_text(report, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="4000:4059")
    parser.add_argument("--control-seeds", default="4000:4019")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--augmentation", default="none", choices=["none", "light"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--rank-lift-branches", type=int, default=2)
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--rows-per-shard", type=int, default=2000)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--monomial-scale-methods", default="raw")
    parser.add_argument("--monomial-log-scale-clip", type=float, default=2.0)
    parser.add_argument("--monomial-shrinkage", type=float, default=0.5)
    parser.add_argument("--monomial-activation-similarity-threshold", type=float, default=0.2)
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    args.alignment_noise_levels = ""
    return args


def seeds_for_setting(args, setting: ReplicationSetting) -> list[int]:
    primary_seeds = verifier.parse_seeds(args.seeds)
    control_seeds = verifier.parse_seeds(args.control_seeds)
    if setting.role == "primary_replication":
        return primary_seeds
    if (
        setting.dataset == "fashion_mnist"
        and setting.architecture == "mlp2"
        and setting.n_models == 3
        and setting.width == 128
        and setting.domain_shift == "none"
        and setting.matching == "weight"
    ):
        return primary_seeds
    return control_seeds


def main() -> None:
    args = parse_args()
    seeds = verifier.parse_seeds(args.seeds)
    if len(seeds) < 60:
        raise ValueError("Prompt 21 requires at least 60 fresh seeds for the primary replication.")

    all_runs: list[dict] = []
    all_individuals: list[dict] = []
    all_triangles: list[dict] = []

    training_groups: dict[tuple, list[ReplicationSetting]] = {}
    setting_seeds: dict[ReplicationSetting, list[int]] = {}
    for setting in SETTINGS:
        setting_seeds[setting] = seeds_for_setting(args, setting)
        key = (setting.dataset, setting.architecture, setting.n_models, setting.width, setting.domain_shift)
        training_groups.setdefault(key, []).append(setting)

    for group_key, settings in training_groups.items():
        dataset, architecture, n_models, width, domain_shift = group_key
        group_seeds = sorted({seed for setting in settings for seed in setting_seeds[setting]})
        for seed in group_seeds:
            for setting in settings:
                if seed not in setting_seeds[setting]:
                    continue
                print(
                    f"prompt21 dataset={dataset} arch={architecture} N={n_models} W={width} "
                    f"shift={domain_shift} matching={setting.matching} seed={seed}",
                    flush=True,
                )
                run_rows, individual_rows, triangle_rows = verifier.run_one_seed(
                    args,
                    dataset,
                    architecture,
                    n_models,
                    width,
                    domain_shift,
                    setting.matching,
                    seed,
                )
                all_runs.extend(run_rows)
                all_individuals.extend(individual_rows)
                all_triangles.extend(triangle_rows)
            verifier._TRAINED_SEED_CACHE.clear()

    runs = decorate_roles(pd.DataFrame(all_runs))
    individuals = decorate_roles(pd.DataFrame(all_individuals))
    triangles = decorate_roles(pd.DataFrame(all_triangles))
    stats = decorate_roles(verifier.compute_stats(runs, args.bootstrap_samples))
    if not stats.empty:
        stats["claim_decision"] = stats.apply(
            lambda row: claim_decision(row) if str(row.get("claim_status", "")) != "method_summary_not_obstruction_correlation" else "",
            axis=1,
        )
    multiple = build_multiple_testing_table(runs, stats, args.bootstrap_samples)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    shard_dir = csv_dir / "prompt21_large_artifacts"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        write_compact_dataframe(
            runs,
            csv_dir / "prompt21_replication_runs.csv",
            shard_dir,
            "prompt21_runs_maps",
            RUN_BULKY_COLUMNS,
            args.rows_per_shard,
        ),
        write_compact_dataframe(
            triangles,
            csv_dir / "prompt21_replication_triangle_defects.csv",
            shard_dir,
            "prompt21_triangle_maps",
            TRIANGLE_BULKY_COLUMNS,
            args.rows_per_shard,
        ),
    ]
    stats.to_csv(csv_dir / "prompt21_replication_stats.csv", index=False, lineterminator="\n")
    individuals.to_csv(csv_dir / "prompt21_replication_individual_models.csv", index=False, lineterminator="\n")
    multiple.to_csv(csv_dir / "prompt21_replication_multiple_testing.csv", index=False, lineterminator="\n")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(csv_dir / "prompt21_large_artifacts_manifest.csv", index=False, lineterminator="\n")
    plot_prompt21(runs, multiple, plot_dir / "prompt21_cycle_vs_degradation.pdf")
    write_report(args, stats, individuals, multiple, manifest, args.reports_dir / "prompt21_replication_report.md")
    verifier.save_json(
        args.reports_dir / "configs" / "prompt21_replication_config.json",
        {
            "argv": sys.argv,
            "parsed_seeds": verifier.summarize_seed_list(seeds),
            "settings": [setting.__dict__ for setting in SETTINGS],
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "environment": verifier.capture_environment(),
        },
    )

    print(f"wrote {csv_dir / 'prompt21_replication_runs.csv'}")
    print(f"wrote {csv_dir / 'prompt21_replication_stats.csv'}")
    print(f"wrote {csv_dir / 'prompt21_replication_multiple_testing.csv'}")
    print(f"wrote {args.reports_dir / 'prompt21_replication_report.md'}")


if __name__ == "__main__":
    main()
