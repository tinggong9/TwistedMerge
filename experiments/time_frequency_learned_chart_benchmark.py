#!/usr/bin/env python
"""Learned chart recovery benchmark for finite time-frequency period-index data."""

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
from src.period_index_detector import RobustPeriodIndexDetection, robust_detect_commutator_matrix_period_index  # noqa: E402
from src.time_frequency_benchmark import (  # noqa: E402
    generate_paired_time_frequency_chart_dataset,
    time_frequency_generator_dict,
)
from src.time_frequency_learned_charts import (  # noqa: E402
    CALIBRATED_CONFIDENCE_MARGIN,
    CALIBRATED_TOLERANCE,
    LIFT_METHOD,
    LearnedChartRecovery,
    detect_recovered_chart_generators,
    fit_input_least_squares_chart,
    fit_linear_autoencoder_chart,
    fit_supervised_encoder_chart,
    identity_chart_ridge_accuracies,
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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def default_cases() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec(2, 2, (2, 4)),
        CaseSpec(3, 2, (3, 6, 9)),
        CaseSpec(2, 3, (2, 4, 8)),
    )


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


def row_passes(case: CaseSpec, rank: int, detection: RobustPeriodIndexDetection, selected_method: str) -> bool:
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


def known_operator_detection(case: CaseSpec, rank: int) -> RobustPeriodIndexDetection:
    return robust_detect_commutator_matrix_period_index(
        time_frequency_generator_dict(case.d, case.k),
        candidate_rank=rank,
        centrality_tol_grid=(CALIBRATED_TOLERANCE,),
        phase_tol_grid=(CALIBRATED_TOLERANCE,),
        confidence_margin=CALIBRATED_CONFIDENCE_MARGIN,
    )


def pack_known_row(
    *,
    case: CaseSpec,
    seed: int,
    noise_level: float,
    chart_count: int,
    real_dimension: int,
    candidate_rank: int,
    accuracies: tuple[float, float, float],
) -> dict[str, object]:
    detection = known_operator_detection(case, candidate_rank)
    selected_method = selected_method_for(detection)
    expected = expected_rank_decision(case, candidate_rank, detection.status)
    return {
        "case_id": f"{case.case_id}_known_rank{candidate_rank}_noise{noise_level:g}_seed{seed}",
        "level": "known_operator_chart",
        "d": case.d,
        "k": case.k,
        "seed": seed,
        "noise_level": noise_level,
        "chart_count": chart_count,
        "real_dimension": real_dimension,
        "latent_dimension": real_dimension,
        "candidate_rank": candidate_rank,
        "expected_period": case.expected_period,
        "expected_index": case.expected_index,
        "learned_operator_error_mean": 0.0,
        "learned_operator_error_max": 0.0,
        "pair_reconstruction_residual_train": 0.0,
        "pair_reconstruction_residual_test": 0.0,
        "train_accuracy": accuracies[0],
        "validation_accuracy": accuracies[1],
        "test_accuracy": accuracies[2],
        "detector_status": detection.status,
        "detected_period": detection.period,
        "detected_index": detection.index,
        "period_correct": detection.period == case.expected_period,
        "index_correct": detection.index == case.expected_index,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": detection.decision,
        "selected_method": selected_method,
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "min_root_margin": detection.min_root_margin,
        "threshold_level": detection.threshold_level,
        "generator_mining_used": False,
        "n_candidate_generators": len(detection.generator_names),
        "expected_decision": expected,
        "pass_fail": "pass" if row_passes(case, candidate_rank, detection, selected_method) else "fail",
        "notes": "oracle known-operator baseline; not learned",
    }


def pack_recovery_row(
    *,
    case: CaseSpec,
    seed: int,
    noise_level: float,
    recovery: LearnedChartRecovery,
    candidate_rank: int,
) -> dict[str, object]:
    detection = detect_recovered_chart_generators(recovery.candidate_generators, candidate_rank)
    selected_method = selected_method_for(detection)
    expected = expected_rank_decision(case, candidate_rank, detection.status)
    return {
        "case_id": f"{case.case_id}_{recovery.level}_rank{candidate_rank}_noise{noise_level:g}_seed{seed}",
        "level": recovery.level,
        "d": case.d,
        "k": case.k,
        "seed": seed,
        "noise_level": noise_level,
        "chart_count": recovery.chart_count,
        "real_dimension": recovery.real_dimension,
        "latent_dimension": recovery.latent_dimension,
        "candidate_rank": candidate_rank,
        "expected_period": case.expected_period,
        "expected_index": case.expected_index,
        "learned_operator_error_mean": recovery.learned_operator_error_mean,
        "learned_operator_error_max": recovery.learned_operator_error_max,
        "pair_reconstruction_residual_train": recovery.pair_reconstruction_residual_train,
        "pair_reconstruction_residual_test": recovery.pair_reconstruction_residual_test,
        "train_accuracy": recovery.train_accuracy,
        "validation_accuracy": recovery.validation_accuracy,
        "test_accuracy": recovery.test_accuracy,
        "detector_status": detection.status,
        "detected_period": detection.period,
        "detected_index": detection.index,
        "period_correct": detection.period == case.expected_period,
        "index_correct": detection.index == case.expected_index,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": detection.decision,
        "selected_method": selected_method,
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "min_root_margin": detection.min_root_margin,
        "threshold_level": detection.threshold_level,
        "generator_mining_used": recovery.generator_mining_used,
        "n_candidate_generators": len(recovery.candidate_generators),
        "expected_decision": expected,
        "pass_fail": "pass" if row_passes(case, candidate_rank, detection, selected_method) else "fail",
        "notes": "; ".join(recovery.notes + tuple(detection.notes)),
    }


def scenario_rows(
    *,
    cases: Iterable[CaseSpec],
    seeds: int,
    noise_levels: Iterable[float],
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    include_d2k3: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        if case.d == 2 and case.k == 3 and not include_d2k3:
            continue
        for noise_level in noise_levels:
            for seed in range(seeds):
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
                accuracies = identity_chart_ridge_accuracies(dataset)
                recoveries = [
                    fit_input_least_squares_chart(dataset, ridge=1e-8 if noise_level <= 0 else 1e-5),
                ]
                if case.k <= 2:
                    recoveries.append(fit_linear_autoencoder_chart(dataset, latent_dimension=dataset.dimension_real))
                    recoveries.append(fit_supervised_encoder_chart(dataset))
                for rank in case.candidate_ranks:
                    rows.append(
                        pack_known_row(
                            case=case,
                            seed=seed,
                            noise_level=float(noise_level),
                            chart_count=dataset.chart_count,
                            real_dimension=dataset.dimension_real,
                            candidate_rank=rank,
                            accuracies=accuracies,
                        )
                    )
                    for recovery in recoveries:
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
    frame["false_lift"] = (frame["selected_method"] == LIFT_METHOD) & (
        frame["expected_decision"] != "period_index_lift_success"
    )
    frame["obstructed_rejected"] = (
        (frame["expected_decision"] == "period_divisible_index_obstructed")
        & (frame["selected_method"] == "none")
    )
    grouped = (
        frame.groupby(["level", "d", "k", "noise_level", "candidate_rank"], dropna=False)
        .agg(
            n=("case_id", "count"),
            certification_rate=("is_certified", "mean"),
            uncertain_rate=("is_uncertain", "mean"),
            rejection_rate=("is_rejected", "mean"),
            correct_period_rate=("period_correct", "mean"),
            correct_index_rate=("index_correct", "mean"),
            false_lift_rate=("false_lift", "mean"),
            period_divisible_index_obstructed_rejection_rate=("obstructed_rejected", "mean"),
            mean_learned_operator_error=("learned_operator_error_mean", "mean"),
            mean_pair_reconstruction_residual=("pair_reconstruction_residual_test", "mean"),
            mean_test_accuracy=("test_accuracy", "mean"),
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


def write_plots(df: pd.DataFrame, summary_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    learned = summary_df[summary_df["level"] != "known_operator_chart"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (level, d, k), group in learned.groupby(["level", "d", "k"]):
        selected = group.groupby("noise_level", as_index=False)["mean_learned_operator_error"].mean()
        if selected["mean_learned_operator_error"].notna().any():
            ax.plot(selected["noise_level"], selected["mean_learned_operator_error"], marker="o", label=f"{level} d={d},k={k}")
    ax.set_xlabel("chart observation noise")
    ax.set_ylabel("mean learned operator error")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_title("Learned chart operator error")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_learned_chart_operator_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for level, group in summary_df.groupby("level"):
        selected = group.groupby("noise_level", as_index=False)["certification_rate"].mean()
        ax.plot(selected["noise_level"], selected["certification_rate"], marker="o", label=level)
    ax.set_xlabel("chart observation noise")
    ax.set_ylabel("certification rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Learned chart certification rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_learned_chart_certification_rate.pdf")
    plt.close(fig)

    zero_noise = df[df["noise_level"] == 0.0].copy()
    zero_noise["lift_selected"] = zero_noise["selected_method"] == LIFT_METHOD
    lift = (
        zero_noise.groupby(["level", "d", "k", "candidate_rank"], as_index=False)["lift_selected"].mean()
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (level, d, k), group in lift.groupby(["level", "d", "k"]):
        selected = group.sort_values("candidate_rank")
        ax.plot(selected["candidate_rank"], selected["lift_selected"], marker="o", label=f"{level} d={d},k={k}")
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("lift selection rate at zero noise")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Learned chart rank threshold")
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_learned_chart_rank_threshold.pdf")
    plt.close(fig)


def write_report(args: argparse.Namespace, df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> None:
    summary_columns = [
        "level",
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
        "false_lift_rate",
        "period_divisible_index_obstructed_rejection_rate",
        "mean_learned_operator_error",
        "mean_pair_reconstruction_residual",
        "mean_test_accuracy",
    ]
    rank_columns = ["level", "d", "k", "candidate_rank", "detector_status", "decision", "selected_method", "pass_fail", "notes"]
    selected_summary = summary_df[summary_df["noise_level"].isin([0.0, 0.01, 0.05])]
    zero_rank = df[(df["noise_level"] == 0.0) & (df["seed"] == 0)]
    worked = summary_df[
        (summary_df["level"] == "input_least_squares_chart")
        & (summary_df["noise_level"] == 0.0)
        & (summary_df["certification_rate"] == 1.0)
    ]
    did_not = summary_df[
        (summary_df["level"].isin(["linear_autoencoder_chart", "supervised_encoder_chart"]))
        & ((summary_df["certification_rate"] < 1.0) | (summary_df["false_lift_rate"] == 0.0))
    ].head(12)
    report = f"""# Time-Frequency Learned Chart Report

This report is generated by `experiments/time_frequency_learned_chart_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Purpose

Prompt 5(k)(v) established a natural known-operator chart: finite
time-frequency shifts and modulations satisfy `M T = zeta T M`, and the
commutator detector recovers period `d` and index `d^k`.  This benchmark asks
whether chart maps learned from paired finite time-frequency data or learned
representations recover the same central projective structure.

## Method Levels

- `known_operator_chart`: oracle baseline using the exact time-frequency
  operators.  This is not learned.
- `input_least_squares_chart`: input least-squares learned maps from paired
  chart observations with the same sample ids.
- `linear_autoencoder_chart`: linear autoencoder features with latent chart
  transitions.  Full-dimensional runs preserve the input coordinate system;
  compressed or chart-specific bases are exploratory.
- `supervised_encoder_chart`: supervised encoder/classifier features.  These
  are exploratory because label features can discard phase information.

## Main Result Table

{format_markdown_table(_display_frame(selected_summary).to_dict("records"), summary_columns)}

## Learned Operator Recovery

The operator error and pair reconstruction residual columns measure whether the
learned transition maps recover the actual finite time-frequency chart maps.
Input least-squares is the clean paired-data test: it does not receive the
known operators except for evaluation/error reporting.

## Rank Threshold Table

{format_markdown_table(_display_frame(zero_rank).to_dict("records"), rank_columns)}

## What Worked

{format_markdown_table(_display_frame(worked).to_dict("records"), summary_columns)}

## What Did Not Work Or Stayed Exploratory

{format_markdown_table(_display_frame(did_not).to_dict("records"), summary_columns)}

Supervised encoder features often preserve the invariant label task while
discarding the phase/projective information needed by the period-index
detector.  Those rows keep `selected_method = none` unless the robust detector
certifies period and index.

## Algorithmic Conclusion

Learned chart recovery can expose the natural finite time-frequency
period-index class when the learning problem is the paired input chart map
itself.  Representation-learned maps are method-dependent: full-dimensional
linear encoders can preserve the structure, while supervised label features may
lose it.  TwistedMerge++ still selects `period_index_projective_morita_lift`
only for certified index-divisible rows.

## Negative Boundaries

- No MNIST/CIFAR residual is claimed to be a Brauer or period-index class.
- No lift is selected for uncertain learned charts.
- Noncentral learned or random maps are not called central/Brauer classes.
- Supervised encoder hidden features are exploratory unless certified.
- Period divisibility alone is not accepted without index divisibility.

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
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    parser.add_argument("--skip-d2k3", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    cases = default_cases()
    df = pd.DataFrame(
        scenario_rows(
            cases=cases,
            seeds=args.seeds,
            noise_levels=args.noise_levels,
            train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            test_samples=args.test_samples,
            include_d2k3=not args.skip_d2k3,
        )
    )
    summary_df = summary_rows(df)
    csv_path = args.reports_dir / "csv" / "time_frequency_learned_chart_benchmark.csv"
    summary_path = args.reports_dir / "csv" / "time_frequency_learned_chart_summary.csv"
    report_path = args.reports_dir / "time_frequency_learned_chart_report.md"
    config_path = args.reports_dir / "configs" / "time_frequency_learned_chart_config.json"
    plots_dir = args.reports_dir / "plots"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_plots(df, summary_df, plots_dir)
    save_json(
        config_path,
        {
            "argv": sys.argv,
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
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "calibrated_centrality_tolerance": CALIBRATED_TOLERANCE,
            "calibrated_phase_tolerance": CALIBRATED_TOLERANCE,
            "calibrated_confidence_margin": CALIBRATED_CONFIDENCE_MARGIN,
            "scope": {
                "levels_separated": [
                    "known_operator_chart",
                    "input_least_squares_chart",
                    "linear_autoencoder_chart",
                    "supervised_encoder_chart",
                ],
                "mnist_cifar_claim": "not_claimed",
                "uncertain_lift_policy": "no_lift",
                "period_divisibility_policy": "index_divisibility_required_for_lift",
            },
        },
    )
    write_report(args, df, summary_df, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_dir / 'time_frequency_learned_chart_operator_error.pdf'}")
    print(f"wrote {plots_dir / 'time_frequency_learned_chart_certification_rate.pdf'}")
    print(f"wrote {plots_dir / 'time_frequency_learned_chart_rank_threshold.pdf'}")


if __name__ == "__main__":
    main()
