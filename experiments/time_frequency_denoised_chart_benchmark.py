#!/usr/bin/env python
"""Denoised learned-chart benchmark for finite time-frequency period-index data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402
from src.period_index_detector import RobustPeriodIndexDetection  # noqa: E402
from src.time_frequency_benchmark import generate_paired_time_frequency_chart_dataset  # noqa: E402
from src.time_frequency_chart_denoising import (  # noqa: E402
    DENOISING_METHODS,
    RIDGE_GRID,
    DenoisedChartRecovery,
    fit_all_denoised_chart_recoveries,
)
from src.time_frequency_learned_charts import (  # noqa: E402
    CALIBRATED_CONFIDENCE_MARGIN,
    CALIBRATED_TOLERANCE,
    DENOISED_CHART_SCOPE_NOTE,
    LIFT_METHOD,
    detect_recovered_chart_generators,
    selected_method_for,
)


@dataclass(frozen=True)
class CaseSpec:
    d: int
    k: int
    candidate_ranks: tuple[int, ...]
    n_classes: int = 3

    @property
    def case_id(self) -> str:
        return f"time_frequency_d{self.d}_k{self.k}"

    @property
    def expected_period(self) -> int:
        return self.d

    @property
    def expected_index(self) -> int:
        return self.d**self.k


CSV_COLUMNS = [
    "case_id",
    "d",
    "k",
    "seed",
    "noise_level",
    "denoising_method",
    "chart_count",
    "real_dimension",
    "candidate_rank",
    "expected_period",
    "expected_index",
    "learned_operator_error_mean_raw",
    "learned_operator_error_mean_denoised",
    "learned_operator_error_max_raw",
    "learned_operator_error_max_denoised",
    "pair_reconstruction_residual_train_raw",
    "pair_reconstruction_residual_test_raw",
    "pair_reconstruction_residual_train_denoised",
    "pair_reconstruction_residual_test_denoised",
    "global_sync_residual",
    "unitary_projection_residual",
    "complex_structure_residual",
    "detector_status",
    "detected_period",
    "detected_index",
    "correct_period",
    "correct_index",
    "period_divides_rank",
    "index_divides_rank",
    "decision",
    "selected_method",
    "expected_decision",
    "pass_fail",
    "max_centrality_score",
    "max_phase_residual",
    "min_root_margin",
    "threshold_level",
    "notes",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def default_cases(*, include_d2k3: bool) -> tuple[CaseSpec, ...]:
    cases = [
        CaseSpec(2, 2, (2, 4)),
        CaseSpec(3, 2, (3, 6, 9)),
    ]
    if include_d2k3:
        cases.append(CaseSpec(2, 3, (2, 4, 8)))
    return tuple(cases)


def expected_rank_decision(case: CaseSpec, rank: int, detector_status: str) -> str:
    if detector_status == "candidate_uncertain":
        return "central_projective_candidate_uncertain"
    if detector_status == "rejected_noncentral":
        return "not_central_projective"
    if detector_status == "unknown_index":
        return "central_projective_index_unknown"
    if rank > 0 and rank % case.expected_index == 0:
        return "period_index_lift_success"
    if rank > 0 and rank % case.expected_period == 0:
        return "period_divisible_index_obstructed"
    return "rank_obstructed"


def row_passes(
    case: CaseSpec,
    rank: int,
    detection: RobustPeriodIndexDetection,
    selected_method: str,
) -> bool:
    expected = expected_rank_decision(case, rank, detection.status)
    if selected_method == LIFT_METHOD and expected != "period_index_lift_success":
        return False
    if detection.status != "certified":
        return selected_method == "none"
    return (
        detection.period == case.expected_period
        and detection.index == case.expected_index
        and detection.decision == expected
        and ((selected_method == LIFT_METHOD) == (expected == "period_index_lift_success"))
    )


def pack_recovery_row(
    *,
    case: CaseSpec,
    seed: int,
    noise_level: float,
    recovery: DenoisedChartRecovery,
    candidate_rank: int,
) -> dict[str, object]:
    detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank)
    selected_method = selected_method_for(detection)
    expected = expected_rank_decision(case, candidate_rank, detection.status)
    notes = list(recovery.notes)
    if recovery.selected_ridge is not None:
        notes.append(f"selected_ridge={recovery.selected_ridge:g}")
    notes.extend(detection.notes)
    return {
        "case_id": (
            f"{case.case_id}_{recovery.denoising_method}_rank{candidate_rank}"
            f"_noise{noise_level:g}_seed{seed}"
        ),
        "d": case.d,
        "k": case.k,
        "seed": seed,
        "noise_level": float(noise_level),
        "denoising_method": recovery.denoising_method,
        "chart_count": recovery.chart_count,
        "real_dimension": recovery.real_dimension,
        "candidate_rank": candidate_rank,
        "expected_period": case.expected_period,
        "expected_index": case.expected_index,
        "learned_operator_error_mean_raw": recovery.learned_operator_error_mean_raw,
        "learned_operator_error_mean_denoised": recovery.learned_operator_error_mean_denoised,
        "learned_operator_error_max_raw": recovery.learned_operator_error_max_raw,
        "learned_operator_error_max_denoised": recovery.learned_operator_error_max_denoised,
        "pair_reconstruction_residual_train_raw": recovery.pair_reconstruction_residual_train_raw,
        "pair_reconstruction_residual_test_raw": recovery.pair_reconstruction_residual_test_raw,
        "pair_reconstruction_residual_train_denoised": recovery.pair_reconstruction_residual_train_denoised,
        "pair_reconstruction_residual_test_denoised": recovery.pair_reconstruction_residual_test_denoised,
        "global_sync_residual": recovery.global_sync_residual,
        "unitary_projection_residual": recovery.unitary_projection_residual,
        "complex_structure_residual": recovery.complex_structure_residual,
        "detector_status": detection.status,
        "detected_period": detection.period,
        "detected_index": detection.index,
        "correct_period": detection.period == case.expected_period,
        "correct_index": detection.index == case.expected_index,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": detection.decision,
        "selected_method": selected_method,
        "expected_decision": expected,
        "pass_fail": "pass" if row_passes(case, candidate_rank, detection, selected_method) else "fail",
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "min_root_margin": detection.min_root_margin,
        "threshold_level": detection.threshold_level,
        "notes": "; ".join(notes),
    }


def scenario_rows(
    *,
    cases: Iterable[CaseSpec],
    seeds: int,
    seed_offset: int,
    noise_levels: Iterable[float],
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        for noise_level in noise_levels:
            for seed_delta in range(seeds):
                seed = int(seed_offset + seed_delta)
                dataset = generate_paired_time_frequency_chart_dataset(
                    case.d,
                    case.k,
                    n_classes=case.n_classes,
                    train_samples=train_samples,
                    validation_samples=validation_samples,
                    test_samples=test_samples,
                    noise_level=float(noise_level),
                    seed=seed,
                )
                recoveries = fit_all_denoised_chart_recoveries(dataset)
                for recovery in recoveries:
                    for rank in case.candidate_ranks:
                        rows.append(
                            pack_recovery_row(
                                case=case,
                                seed=seed,
                                noise_level=float(noise_level),
                                recovery=recovery,
                                candidate_rank=rank,
                            )
                        )
    return rows


def summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["is_certified"] = frame["detector_status"] == "certified"
    frame["is_uncertain"] = frame["detector_status"].isin(["candidate_uncertain", "unknown_index"])
    frame["is_rejected"] = frame["detector_status"] == "rejected_noncentral"
    frame["is_lift"] = frame["selected_method"] == LIFT_METHOD
    frame["false_lift"] = frame["is_lift"] & (frame["expected_decision"] != "period_index_lift_success")
    frame["obstructed_rejected"] = (
        (frame["expected_decision"] == "period_divisible_index_obstructed")
        & (frame["selected_method"] == "none")
    )
    grouped = (
        frame.groupby(["denoising_method", "d", "k", "noise_level", "candidate_rank"], dropna=False)
        .agg(
            n=("case_id", "count"),
            certification_rate=("is_certified", "mean"),
            uncertain_rate=("is_uncertain", "mean"),
            rejection_rate=("is_rejected", "mean"),
            correct_period_rate=("correct_period", "mean"),
            correct_index_rate=("correct_index", "mean"),
            lift_rate=("is_lift", "mean"),
            false_lift_rate=("false_lift", "mean"),
            period_divisible_index_obstructed_rejection_rate=("obstructed_rejected", "mean"),
            mean_learned_operator_error_raw=("learned_operator_error_mean_raw", "mean"),
            mean_learned_operator_error_denoised=("learned_operator_error_mean_denoised", "mean"),
            mean_pair_reconstruction_residual_raw=("pair_reconstruction_residual_test_raw", "mean"),
            mean_pair_reconstruction_residual_denoised=("pair_reconstruction_residual_test_denoised", "mean"),
            mean_global_sync_residual=("global_sync_residual", "mean"),
            mean_unitary_projection_residual=("unitary_projection_residual", "mean"),
        )
        .reset_index()
    )
    return grouped


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "nan" if pd.isna(value) else f"{float(value):.4g}")
    return out


def _method_noise_extent(summary_df: pd.DataFrame) -> pd.DataFrame:
    frame = summary_df.copy()
    success_rank = frame["candidate_rank"] == (frame["d"] ** frame["k"])
    eligible = frame[success_rank]
    rows = []
    for method, group in eligible.groupby("denoising_method"):
        certified = group[group["certification_rate"] > 0]
        rows.append(
            {
                "denoising_method": method,
                "max_certified_noise": certified["noise_level"].max() if not certified.empty else np.nan,
                "mean_certification_rate": group["certification_rate"].mean(),
                "max_false_lift_rate": group["false_lift_rate"].max(),
                "mean_operator_error_raw": group["mean_learned_operator_error_raw"].mean(),
                "mean_operator_error_denoised": group["mean_learned_operator_error_denoised"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _certification_gain_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    lift_rank = summary_df["candidate_rank"] == (summary_df["d"] ** summary_df["k"])
    lift_summary = summary_df[lift_rank].copy()
    raw = lift_summary[lift_summary["denoising_method"] == "raw_least_squares"][
        ["d", "k", "noise_level", "candidate_rank", "certification_rate"]
    ].rename(columns={"certification_rate": "raw_certification_rate"})
    merged = lift_summary.merge(raw, on=["d", "k", "noise_level", "candidate_rank"], how="left")
    merged["certification_gain_over_raw"] = merged["certification_rate"] - merged["raw_certification_rate"]
    gain = merged[
        (merged["denoising_method"] != "raw_least_squares")
        & (merged["certification_gain_over_raw"] > 0)
    ].copy()
    return gain.sort_values(
        ["certification_gain_over_raw", "noise_level", "denoising_method"],
        ascending=[False, True, True],
    )


def write_plots(df: pd.DataFrame, summary_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    success_rank = summary_df["candidate_rank"] == (summary_df["d"] ** summary_df["k"])
    lift_summary = summary_df[success_rank]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, group in lift_summary.groupby("denoising_method"):
        selected = group.groupby("noise_level", as_index=False)["certification_rate"].mean()
        ax.plot(selected["noise_level"], selected["certification_rate"], marker="o", label=method)
    ax.set_xlabel("chart observation noise")
    ax.set_ylabel("certification rate on lift ranks")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_title("Denoised learned-chart certification rate")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_denoised_certification_rate.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, group in lift_summary.groupby("denoising_method"):
        selected = group.groupby("noise_level", as_index=False)[
            ["mean_learned_operator_error_raw", "mean_learned_operator_error_denoised"]
        ].mean()
        ax.plot(
            selected["noise_level"],
            selected["mean_learned_operator_error_denoised"],
            marker="o",
            label=method,
        )
    raw = lift_summary.groupby("noise_level", as_index=False)["mean_learned_operator_error_raw"].mean()
    ax.plot(raw["noise_level"], raw["mean_learned_operator_error_raw"], linestyle="--", color="black", label="raw mean")
    ax.set_xlabel("chart observation noise")
    ax.set_ylabel("mean operator error")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_title("Denoised learned-chart operator error")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_denoised_operator_error.pdf")
    plt.close(fig)

    zero_noise = df[df["noise_level"] == 0.0].copy()
    zero_noise["lift_selected"] = zero_noise["selected_method"] == LIFT_METHOD
    lift = (
        zero_noise.groupby(["denoising_method", "d", "k", "candidate_rank"], as_index=False)["lift_selected"].mean()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (method, d, k), group in lift.groupby(["denoising_method", "d", "k"]):
        selected = group.sort_values("candidate_rank")
        ax.plot(selected["candidate_rank"], selected["lift_selected"], marker="o", label=f"{method} d={d},k={k}")
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("lift selection rate at zero noise")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Denoised learned-chart rank threshold")
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_denoised_rank_threshold.pdf")
    plt.close(fig)


def write_report(args: argparse.Namespace, df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> None:
    summary_columns = [
        "denoising_method",
        "d",
        "k",
        "noise_level",
        "candidate_rank",
        "n",
        "certification_rate",
        "uncertain_rate",
        "rejection_rate",
        "correct_period_rate",
        "correct_index_rate",
        "lift_rate",
        "false_lift_rate",
        "period_divisible_index_obstructed_rejection_rate",
    ]
    error_columns = [
        "denoising_method",
        "d",
        "k",
        "noise_level",
        "candidate_rank",
        "mean_learned_operator_error_raw",
        "mean_learned_operator_error_denoised",
        "mean_pair_reconstruction_residual_raw",
        "mean_pair_reconstruction_residual_denoised",
        "mean_global_sync_residual",
        "mean_unitary_projection_residual",
    ]
    rank_columns = [
        "denoising_method",
        "d",
        "k",
        "candidate_rank",
        "detector_status",
        "decision",
        "selected_method",
        "expected_decision",
        "pass_fail",
    ]
    false_lift = summary_df[summary_df["false_lift_rate"] > 0]
    if false_lift.empty:
        false_lift = pd.DataFrame(
            [{"denoising_method": "all_tested_methods", "false_lift_rate": 0.0, "n": int(len(df))}]
        )
    lift_ranks = summary_df[summary_df["candidate_rank"] == (summary_df["d"] ** summary_df["k"])]
    selected_summary = lift_ranks[
        lift_ranks["noise_level"].isin([0.0, 1e-4, 3e-4, 1e-3, 1e-2, 5e-2])
    ]
    operator_table = selected_summary.copy()
    rank_table = df[(df["noise_level"] == 0.0) & (df["seed"] == args.seed_offset)].copy()
    best_method = _method_noise_extent(summary_df).sort_values(
        ["max_false_lift_rate", "max_certified_noise", "mean_certification_rate"],
        ascending=[True, False, False],
    )
    certification_gain = _certification_gain_table(summary_df).head(16)
    improvement = summary_df[
        summary_df["mean_learned_operator_error_denoised"]
        < summary_df["mean_learned_operator_error_raw"]
    ].head(12)
    still_failed = selected_summary[
        (selected_summary["certification_rate"] < 1.0)
        & (selected_summary["noise_level"] > 0)
    ].head(12)

    report = f"""# Time-Frequency Denoised Chart Report

This report is generated by `experiments/time_frequency_denoised_chart_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Purpose

The previous learned-chart benchmark recovered the finite time-frequency
period-index structure from clean paired input maps, but noisy paired maps could
fail the calibrated robust detector.  This benchmark tests whether
structure-preserving denoising and projection recover the central period-index
class from noisy learned chart maps.

## Denoising Methods

- `raw_least_squares`: baseline learned identity-to-chart maps.
- `ridge_least_squares`: ridge value selected only by validation pair residual.
- `nearest_unitary_projection`: real polar projection to the nearest orthogonal map.
- `complex_unitary_projection`: convert real block maps to complex maps, project
  to the nearest complex unitary, then realify for evaluation.
- `global_chart_synchronization`: block spectral synchronization from all
  pairwise learned chart maps.
- `unitary_global_chart_synchronization`: pairwise complex-unitary projection
  followed by global synchronization.

Scope note: {DENOISED_CHART_SCOPE_NOTE}

## Main Certification Table

{format_markdown_table(_display_frame(selected_summary).to_dict("records"), summary_columns)}

## Operator-Error Table

{format_markdown_table(_display_frame(operator_table).to_dict("records"), error_columns)}

## Rank-Threshold Table

{format_markdown_table(_display_frame(rank_table).to_dict("records"), rank_columns)}

## False Lift Table

{format_markdown_table(_display_frame(false_lift).to_dict("records"), list(false_lift.columns))}

## Best Method Discussion

{format_markdown_table(_display_frame(best_method).to_dict("records"), list(best_method.columns))}

The best method is determined by the largest certified nonzero noise range on
index-divisible ranks, subject to zero false lifts.  If certification does not
extend beyond the raw baseline, lower operator error is reported only as a
diagnostic improvement.

## What Improved: Certification

{format_markdown_table(_display_frame(certification_gain).to_dict("records"), summary_columns + ["raw_certification_rate", "certification_gain_over_raw"])}

These rows are the strongest improvement signal: a denoised method certifies
more index-divisible noisy runs than raw least squares under the same calibrated
thresholds.

## What Improved: Operator Error

{format_markdown_table(_display_frame(improvement).to_dict("records"), error_columns)}

Projection and synchronization usually reduce operator error relative to the
raw least-squares maps.  Certification improvement is claimed only for rows
where the robust detector is certified and the index divides the candidate rank.

## What Failed

{format_markdown_table(_display_frame(still_failed).to_dict("records"), summary_columns)}

Noisy cases that remain rejected or uncertain keep `selected_method = none`.
No period-only rank is lifted, and lower operator error alone is not treated as
a period-index certificate.

## Algorithmic Conclusion

Denoising helps when it preserves the finite time-frequency chart algebra well
enough for the calibrated commutator detector.  The strongest supported claim is
empirical and benchmark-scoped: unitary projection and synchronization can
recover certified period/index rows at small nonzero chart noise while
preserving the zero-false-lift policy on tested ranks.

## Negative Boundaries

- No MNIST/CIFAR residual is claimed to be a Brauer or period-index class.
- No lift is selected for uncertain learned candidates.
- No lift is selected from period divisibility alone.
- Noncentral learned maps are not labeled central/Brauer classes.
- Arbitrary supervised neural hidden features are not claimed to recover this
  structure.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2],
    )
    parser.add_argument("--include-d2k3", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    cases = default_cases(include_d2k3=args.include_d2k3)
    rows = scenario_rows(
        cases=cases,
        seeds=args.seeds,
        seed_offset=args.seed_offset,
        noise_levels=args.noise_levels,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
    )
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    summary_df = summary_rows(df)

    csv_path = args.reports_dir / "csv" / "time_frequency_denoised_chart_benchmark.csv"
    summary_path = args.reports_dir / "csv" / "time_frequency_denoised_chart_summary.csv"
    report_path = args.reports_dir / "time_frequency_denoised_chart_report.md"
    config_path = args.reports_dir / "configs" / "time_frequency_denoised_chart_config.json"
    plots_dir = args.reports_dir / "plots"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_plots(df, summary_df, plots_dir)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "command": args.command_string,
            "commit": git_commit(),
            "environment": capture_environment(),
            "cases": [
                {
                    "d": case.d,
                    "k": case.k,
                    "candidate_ranks": list(case.candidate_ranks),
                    "expected_period": case.expected_period,
                    "expected_index": case.expected_index,
                }
                for case in cases
            ],
            "noise_levels": args.noise_levels,
            "seeds": args.seeds,
            "seed_offset": args.seed_offset,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "ridge_grid": list(RIDGE_GRID),
            "denoising_methods": list(DENOISING_METHODS),
            "calibrated_centrality_tolerance": CALIBRATED_TOLERANCE,
            "calibrated_phase_tolerance": CALIBRATED_TOLERANCE,
            "calibrated_confidence_margin": CALIBRATED_CONFIDENCE_MARGIN,
            "scope": {
                "mnist_cifar_claim": "not_claimed",
                "uncertain_lift_policy": "no_lift",
                "period_divisibility_policy": "index_divisibility_required_for_lift",
                "operator_error_only_policy": "diagnostic_not_certificate",
                "optional_d2k3_included": bool(args.include_d2k3),
            },
        },
    )
    write_report(args, df, summary_df, report_path)

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_dir / 'time_frequency_denoised_certification_rate.pdf'}")
    print(f"wrote {plots_dir / 'time_frequency_denoised_operator_error.pdf'}")
    print(f"wrote {plots_dir / 'time_frequency_denoised_rank_threshold.pdf'}")


if __name__ == "__main__":
    main()
