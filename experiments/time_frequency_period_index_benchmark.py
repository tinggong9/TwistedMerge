#!/usr/bin/env python
"""Natural finite time-frequency period-index benchmark."""

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
from src.time_frequency_benchmark import (  # noqa: E402
    TIME_FREQUENCY_SCOPE_NOTE,
    generate_time_frequency_dataset,
    orbit_invariant_prototype_accuracy,
    time_frequency_chart_operators,
    time_frequency_generator_dict,
)
from src.twisted_merge_plus import TwistedMergePlus  # noqa: E402


LIFT_METHOD = "period_index_projective_morita_lift"


@dataclass(frozen=True)
class TimeFrequencyCase:
    d: int
    k: int
    candidate_ranks: tuple[int, ...]
    n_classes: int = 3

    @property
    def setting_id(self) -> str:
        return f"time_frequency_d{self.d}_k{self.k}"

    @property
    def expected_period(self) -> int:
        return self.d

    @property
    def expected_index(self) -> int:
        return self.d**self.k

    @property
    def signal_dimension_complex(self) -> int:
        return self.d**self.k

    @property
    def signal_dimension_real(self) -> int:
        return 2 * self.signal_dimension_complex


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def default_cases() -> tuple[TimeFrequencyCase, ...]:
    return (
        TimeFrequencyCase(d=2, k=2, candidate_ranks=(2, 4)),
        TimeFrequencyCase(d=2, k=3, candidate_ranks=(2, 4, 8)),
        TimeFrequencyCase(d=3, k=2, candidate_ranks=(3, 6, 9)),
        TimeFrequencyCase(d=4, k=1, candidate_ranks=(2, 4)),
    )


def unresolved_pairwise(width: int) -> dict[tuple[int, int], np.ndarray]:
    diagonal = np.diag(np.linspace(1.0, 2.0, width)).astype(complex)
    return {
        (0, 0): np.eye(width, dtype=complex),
        (1, 1): np.eye(width, dtype=complex),
        (2, 2): np.eye(width, dtype=complex),
        (0, 1): diagonal,
        (1, 2): np.eye(width, dtype=complex),
        (2, 0): np.eye(width, dtype=complex),
    }


def expected_decision(period: int, index: int, rank: int) -> str:
    if rank > 0 and rank % index == 0:
        return "period_index_lift_success"
    if rank > 0 and rank % period == 0:
        return "period_divisible_index_obstructed"
    return "rank_obstructed"


def run_tmpp_known_operator_chart(case: TimeFrequencyCase, rank: int):
    generators = time_frequency_generator_dict(case.d, case.k)
    return TwistedMergePlus().run(
        unresolved_pairwise(case.signal_dimension_complex),
        n_models=3,
        width=case.signal_dimension_complex,
        period_index_generators=generators,
        candidate_lift_rank=rank,
        period_index_detection_mode="robust_only",
    )


def dataset_metrics(
    case: TimeFrequencyCase,
    *,
    noise_level: float,
    seed: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> dict[str, float]:
    dataset = generate_time_frequency_dataset(
        case.d,
        case.k,
        n_classes=case.n_classes,
        train_samples=train_samples,
        validation_samples=validation_samples,
        test_samples=test_samples,
        noise_level=noise_level,
        seed=seed,
    )
    return {
        "train_accuracy_mean": orbit_invariant_prototype_accuracy(dataset, split="train"),
        "validation_accuracy_mean": orbit_invariant_prototype_accuracy(dataset, split="validation"),
        "test_accuracy_mean": orbit_invariant_prototype_accuracy(dataset, split="test"),
    }


def row_passes(case: TimeFrequencyCase, rank: int, detection, selected_method: str) -> bool:
    expected = expected_decision(case.expected_period, case.expected_index, rank)
    if detection is None:
        return False
    if detection.status != "certified":
        return False
    if detection.period != case.expected_period or detection.index != case.expected_index:
        return False
    if detection.decision != expected:
        return False
    return (selected_method == LIFT_METHOD) == (expected == "period_index_lift_success")


def pack_row(
    *,
    case: TimeFrequencyCase,
    candidate_rank: int,
    noise_level: float,
    seed: int,
    metrics: dict[str, float],
) -> dict[str, object]:
    result = run_tmpp_known_operator_chart(case, candidate_rank)
    detection = result.diagnostics.period_index
    charts = time_frequency_chart_operators(case.d, case.k)
    selected_method = result.selected_method
    notes = [
        "known_operator_chart",
        "dataset label is prototype identity under nuisance time-frequency shifts",
        "orbit-invariant prototype classifier only; no learned model chart evaluated",
    ]
    if detection is not None:
        notes.extend(detection.notes)
    return {
        "case_id": f"{case.setting_id}_rank{candidate_rank}_noise{noise_level:g}_seed{seed}",
        "level": "known_operator_chart",
        "d": case.d,
        "k": case.k,
        "signal_dimension_complex": case.signal_dimension_complex,
        "signal_dimension_real": case.signal_dimension_real,
        "n_charts": len(charts),
        "candidate_rank": candidate_rank,
        "noise_level": noise_level,
        "seed": seed,
        "expected_period": case.expected_period,
        "expected_index": case.expected_index,
        "detected_period": None if detection is None else detection.period,
        "detected_index": None if detection is None else detection.index,
        "detector_status": "missing" if detection is None else detection.status,
        "decision": "missing" if detection is None else detection.decision,
        "selected_method": selected_method,
        "period_divides_rank": None if detection is None else detection.period_divides_rank,
        "index_divides_rank": None if detection is None else detection.index_divides_rank,
        "max_centrality_score": np.nan if detection is None else detection.max_centrality_score,
        "max_phase_residual": np.nan if detection is None else detection.max_phase_residual,
        "min_root_margin": np.nan if detection is None else detection.min_root_margin,
        "train_accuracy_mean": metrics["train_accuracy_mean"],
        "validation_accuracy_mean": metrics["validation_accuracy_mean"],
        "test_accuracy_mean": metrics["test_accuracy_mean"],
        "naive_merge_accuracy": np.nan,
        "chart_aware_merge_accuracy": np.nan,
        "period_index_lift_accuracy": np.nan,
        "pass_fail": "pass" if row_passes(case, candidate_rank, detection, selected_method) else "fail",
        "notes": "; ".join(notes),
    }


def scenario_rows(
    *,
    cases: Iterable[TimeFrequencyCase],
    noise_levels: Iterable[float],
    seeds: int,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        for noise_level in noise_levels:
            for seed in range(seeds):
                metrics = dataset_metrics(
                    case,
                    noise_level=float(noise_level),
                    seed=seed,
                    train_samples=train_samples,
                    validation_samples=validation_samples,
                    test_samples=test_samples,
                )
                for rank in case.candidate_ranks:
                    rows.append(
                        pack_row(
                            case=case,
                            candidate_rank=rank,
                            noise_level=float(noise_level),
                            seed=seed,
                            metrics=metrics,
                        )
                    )
    return rows


def summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["is_pass"] = frame["pass_fail"] == "pass"
    frame["is_certified"] = frame["detector_status"] == "certified"
    frame["lift_selected"] = frame["selected_method"] == LIFT_METHOD
    grouped = (
        frame.groupby(["level", "d", "k", "candidate_rank", "expected_period", "expected_index"], dropna=False)
        .agg(
            n=("case_id", "count"),
            certification_rate=("is_certified", "mean"),
            lift_rate=("lift_selected", "mean"),
            pass_rate=("is_pass", "mean"),
            train_accuracy_mean=("train_accuracy_mean", "mean"),
            validation_accuracy_mean=("validation_accuracy_mean", "mean"),
            test_accuracy_mean=("test_accuracy_mean", "mean"),
            max_centrality_score=("max_centrality_score", "max"),
            max_phase_residual=("max_phase_residual", "max"),
            min_root_margin=("min_root_margin", "min"),
        )
        .reset_index()
    )
    return grouped


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in [
        "train_accuracy_mean",
        "validation_accuracy_mean",
        "test_accuracy_mean",
        "certification_rate",
        "lift_rate",
        "pass_rate",
        "max_centrality_score",
        "max_phase_residual",
        "min_root_margin",
    ]:
        if col in out:
            out[col] = out[col].map(lambda value: "nan" if pd.isna(value) else f"{float(value):.4g}")
    return out


def write_plots(df: pd.DataFrame, summary_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (d, k), group in summary_df.groupby(["d", "k"]):
        selected = group.sort_values("candidate_rank")
        ax.plot(
            selected["candidate_rank"],
            selected["lift_rate"],
            marker="o",
            label=f"d={d}, k={k}",
        )
        threshold = int(selected["expected_index"].iloc[0])
        ax.axvline(threshold, color="0.85", linewidth=1)
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("period-index lift selection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Time-frequency rank threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_period_index_rank_threshold.pdf")
    plt.close(fig)

    status_counts = (
        df.groupby(["d", "k", "detector_status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    status_counts["setting"] = status_counts.apply(lambda row: f"d={int(row['d'])}, k={int(row['k'])}", axis=1)
    totals = status_counts.groupby("setting")["n"].transform("sum")
    status_counts["rate"] = status_counts["n"] / totals
    pivot = status_counts.pivot_table(index="setting", columns="detector_status", values="rate", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = np.zeros(len(pivot))
    for status in pivot.columns:
        values = pivot[status].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=status)
        bottom += values
    ax.set_xlabel("time-frequency setting")
    ax.set_ylabel("detector status rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Known-operator chart detection rates")
    ax.legend(fontsize=8)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(plots_dir / "time_frequency_period_index_detection_rates.pdf")
    plt.close(fig)


def write_report(args: argparse.Namespace, df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> None:
    summary_columns = [
        "level",
        "d",
        "k",
        "candidate_rank",
        "expected_period",
        "expected_index",
        "n",
        "certification_rate",
        "lift_rate",
        "pass_rate",
        "train_accuracy_mean",
        "test_accuracy_mean",
        "max_centrality_score",
        "max_phase_residual",
    ]
    rank_columns = [
        "d",
        "k",
        "candidate_rank",
        "expected_period",
        "expected_index",
        "lift_rate",
        "pass_rate",
        "train_accuracy_mean",
        "test_accuracy_mean",
    ]
    sample_columns = [
        "case_id",
        "level",
        "d",
        "k",
        "candidate_rank",
        "detector_status",
        "detected_period",
        "detected_index",
        "decision",
        "selected_method",
        "period_divides_rank",
        "index_divides_rank",
        "pass_fail",
        "notes",
    ]
    display_summary = _display_frame(summary_df)
    rank_table = _display_frame(summary_df.sort_values(["d", "k", "candidate_rank"]))
    obstructed_samples = _display_frame(
        df[
            (df["decision"].isin(["period_divisible_index_obstructed", "rank_obstructed"]))
            & (df["noise_level"] == 0.0)
            & (df["seed"] == 0)
        ].head(8)
    )
    lift_samples = _display_frame(
        df[
            (df["selected_method"] == LIFT_METHOD)
            & (df["noise_level"] == 0.0)
            & (df["seed"] == 0)
        ].head(8)
    )
    report = f"""# Time-Frequency Period-Index Report

This report is generated by `experiments/time_frequency_period_index_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Why This Benchmark Is Natural

Discrete signals on `Z/dZ` have canonical time shifts `T` and frequency
modulations `M`.  With the convention `(T x)[n] = x[n-1]` and
`(M x)[n] = zeta^n x[n]`, they satisfy `M T = zeta T M`.  The scalar central
phase is therefore the finite Heisenberg time-frequency symmetry of the signal
domain itself.  The benchmark feeds those known chart operators to the
commutator-matrix detector; it does not plant an unrelated obstruction after a
neural experiment.

Scope note: {TIME_FREQUENCY_SCOPE_NOTE}

## Dataset And Task

The dataset uses finite chirp/Gabor-like complex prototypes on
`(C^d)^{{tensor k}}`.  Each example applies random nuisance operators
`T_1^a M_1^b ... T_k^a M_k^b` and optional complex Gaussian noise; the label is
the prototype identity, not the shift or modulation.  The accuracy columns use
an orbit-invariant nearest-prototype classifier as a sanity check for the
signal task.  They are not learned chart-transition or model-merging results.

## Known-Operator Chart Results

{format_markdown_table(display_summary.to_dict("records"), summary_columns)}

## Rank Threshold Table

{format_markdown_table(rank_table.to_dict("records"), rank_columns)}

Period-divisible or period-failing ranks are rejected:

{format_markdown_table(obstructed_samples.to_dict("records"), sample_columns)}

Index-divisible ranks select the period-index lift:

{format_markdown_table(lift_samples.to_dict("records"), sample_columns)}

## Learned-Model Chart Results

Not evaluated in this run.  The only chart level reported here is
`known_operator_chart`, where the transition/generator maps are the natural
finite time-frequency operators.

## Comparison To Previous Central Benchmarks

The earlier central period-index reports use controlled algebraic Heisenberg
generators, synthetic loop-holonomy mining, or calibrated noisy controls.  This
benchmark keeps the same conservative detector and rank policy but sources the
central relation from finite time-frequency signal geometry.

## Algorithmic Conclusion

The commutator-matrix period-index detector recovers period `d` and index
`d^k` from known finite time-frequency chart operators.  TwistedMerge++ selects
`period_index_projective_morita_lift` only when the candidate rank is divisible
by the detected index.  Period divisibility alone remains obstructed.

## Negative Boundaries

- No MNIST/CIFAR residual is claimed to be a Brauer or period-index class.
- No learned model chart transition is certified here.
- The orbit-invariant prototype classifier is a dataset sanity check, not a
  model-merging accuracy claim.
- Period divisibility is not enough; index divisibility is required.
- Uncertain candidates would not lift, although this known-operator run is
  exact and certified.

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
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    cases = default_cases()
    df = pd.DataFrame(
        scenario_rows(
            cases=cases,
            noise_levels=args.noise_levels,
            seeds=args.seeds,
            train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            test_samples=args.test_samples,
        )
    )
    summary_df = summary_rows(df)

    csv_path = args.reports_dir / "csv" / "time_frequency_period_index_benchmark.csv"
    summary_path = args.reports_dir / "csv" / "time_frequency_period_index_summary.csv"
    report_path = args.reports_dir / "time_frequency_period_index_report.md"
    config_path = args.reports_dir / "configs" / "time_frequency_period_index_config.json"
    plots_dir = args.reports_dir / "plots"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_plots(df, summary_df, plots_dir)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "environment": capture_environment(),
            "commit": git_commit(),
            "scope_note": TIME_FREQUENCY_SCOPE_NOTE,
            "cases": [
                {
                    "d": case.d,
                    "k": case.k,
                    "candidate_ranks": list(case.candidate_ranks),
                    "expected_period": case.expected_period,
                    "expected_index": case.expected_index,
                    "n_classes": case.n_classes,
                }
                for case in cases
            ],
            "noise_levels": args.noise_levels,
            "seeds": args.seeds,
            "train_samples": args.train_samples,
            "validation_samples": args.validation_samples,
            "test_samples": args.test_samples,
            "level_2_learned_model_chart": "not_evaluated",
        },
    )
    write_report(args, df, summary_df, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {plots_dir / 'time_frequency_period_index_rank_threshold.pdf'}")
    print(f"wrote {plots_dir / 'time_frequency_period_index_detection_rates.pdf'}")


if __name__ == "__main__":
    main()
