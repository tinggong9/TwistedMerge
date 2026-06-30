#!/usr/bin/env python
"""Central period-index benchmark for k-pair Heisenberg systems."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402
from src.period_index_central import (  # noqa: E402
    check_period_index_obstruction,
    period_index_metadata,
    toy_prediction_losses,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def default_cases() -> list[tuple[str, int, int]]:
    return [
        ("d2_k1", 2, 1),
        ("d2_k2", 2, 2),
        ("d2_k3", 2, 3),
        ("d3_k1", 3, 1),
        ("d3_k2", 3, 2),
        ("d4_k1", 4, 1),
        ("d4_k2", 4, 2),
    ]


def run_cases(max_multiplier: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for case_id, d, k in default_cases():
        metadata = period_index_metadata(d, k)
        for rank in range(1, max_multiplier * metadata.index + 1):
            result = check_period_index_obstruction(d, k, rank)
            losses = toy_prediction_losses(d, k, rank)
            rows.append(
                {
                    "case_id": case_id,
                    "d": result.d,
                    "k": result.k,
                    "period": result.period,
                    "index": result.index,
                    "candidate_rank": result.candidate_rank,
                    "period_divides_rank": result.period_divides_rank,
                    "index_divides_rank": result.index_divides_rank,
                    "obstruction_prediction": result.obstruction_prediction,
                    "constructed_lift_success": result.constructed_lift_success,
                    "max_relation_residual": result.max_relation_residual,
                    "is_minimal_success": result.is_minimal_success,
                    "ordinary_rank_r_loss": losses["ordinary_rank_r_loss"],
                    "lifted_rank_r_loss": losses["lifted_rank_r_loss"],
                    "branch_projective_loss": losses["branch_projective_loss"],
                    "extra_capacity_label": losses["extra_capacity_label"],
                    "prediction_note": losses["prediction_note"],
                    "lift_kind": result.lift_kind,
                    "ordinary_untwisted_descent_on_original_rank": result.ordinary_untwisted_descent_on_original_rank,
                    "original_class_vanishes_on_same_cover": result.original_class_vanishes_on_same_cover,
                    "interpretation": result.interpretation,
                }
            )

    df = pd.DataFrame(rows)
    summary_rows = []
    for case_id, group in df.groupby("case_id", sort=False):
        successes = group[group["constructed_lift_success"]]
        failures = group[~group["constructed_lift_success"]]
        period = int(group["period"].iloc[0])
        index = int(group["index"].iloc[0])
        success_exactly_index_divisible = bool(
            np.all(group["constructed_lift_success"].to_numpy() == (group["candidate_rank"].to_numpy() % index == 0))
        )
        period_only_failures = [
            int(row.candidate_rank)
            for row in group.itertuples()
            if bool(row.period_divides_rank) and not bool(row.index_divides_rank)
        ]
        rank_outcomes = ",".join(
            f"{int(row.candidate_rank)}:{'S' if bool(row.constructed_lift_success) else 'F'}"
            for row in group.itertuples()
        )
        minimal_success_rank = int(successes["candidate_rank"].min()) if not successes.empty else -1
        max_success_residual = float(successes["max_relation_residual"].max()) if not successes.empty else float("nan")
        min_failure_residual = float(failures["max_relation_residual"].min()) if not failures.empty else float("nan")
        summary_rows.append(
            {
                "case_id": case_id,
                "d": int(group["d"].iloc[0]),
                "k": int(group["k"].iloc[0]),
                "period": period,
                "index": index,
                "minimal_success_rank": minimal_success_rank,
                "success_exactly_when_index_divides_rank": success_exactly_index_divisible,
                "period_divisible_but_index_obstructed_ranks": ",".join(map(str, period_only_failures)),
                "rank_outcomes": rank_outcomes,
                "max_success_residual": max_success_residual,
                "min_failure_residual": min_failure_residual,
                "theorem_supported": bool(
                    success_exactly_index_divisible and minimal_success_rank == index and max_success_residual < 1e-10
                ),
                "interpretation": (
                    f"period={period}, index={index}; success ranks are exactly multiples of the index, "
                    "not merely multiples of the period"
                ),
            }
        )
    return df, pd.DataFrame(summary_rows)


def write_theorem_tex(path: Path) -> None:
    text = r"""\begin{theorem}[Central finite Heisenberg period-index benchmark]
Let $\zeta$ be a primitive $d$-th root of unity.  For $k \ge 1$, consider
generators $U_1,V_1,\ldots,U_k,V_k$ satisfying
\[
  U_i V_i = \zeta V_i U_i,\qquad
  U_i V_j = V_j U_i\quad (i \ne j),
\]
with all $U_i$ commuting and all $V_i$ commuting.  The associated central
projective class has period $d$, and the benchmark index is $d^k$.
\end{theorem}

\begin{proof}[Proof sketch]
For $k=1$, the determinant obstruction for $U V = \zeta V U$ forces
$d \mid r$, and the $d$-dimensional clock-shift pair realizes equality.  For
general $k$, the nondegenerate finite Heisenberg cocycle is realized by the
tensor product of $k$ independent clock-shift pairs on
$(\mathbb{C}^d)^{\otimes k}$.  Equivalently, the finite Heisenberg
Stone-von-Neumann theorem gives irreducible projective representation
dimension $d^k$.  Direct sums realize exactly ranks divisible by $d^k$.
\end{proof}

\paragraph{Benchmark interpretation.}
The construction is a controlled central/projective period-index model.  It
absorbs the class by a finite-rank projective/Morita lift and does not claim
ordinary trivialization of the original class on the same cover.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _selected_rows(df: pd.DataFrame) -> list[dict]:
    selected_rows = []
    for _case_id, group in df.groupby("case_id", sort=False):
        period = int(group["period"].iloc[0])
        index = int(group["index"].iloc[0])
        selected_ranks = {1, period, 2 * period, index, 2 * index}
        selected_ranks.update(rank for rank in [3 * period, index + period] if rank <= int(group["candidate_rank"].max()))
        selected = group[group["candidate_rank"].isin(sorted(selected_ranks))]
        selected_rows.extend(selected.drop_duplicates(subset=["case_id", "candidate_rank"]).to_dict("records"))
    return selected_rows


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    summary_columns = [
        "case_id",
        "d",
        "k",
        "period",
        "index",
        "minimal_success_rank",
        "success_exactly_when_index_divides_rank",
        "period_divisible_but_index_obstructed_ranks",
        "rank_outcomes",
        "theorem_supported",
    ]
    selected_columns = [
        "case_id",
        "candidate_rank",
        "period_divides_rank",
        "index_divides_rank",
        "obstruction_prediction",
        "constructed_lift_success",
        "max_relation_residual",
        "is_minimal_success",
    ]
    d3k2 = df[(df["d"] == 3) & (df["k"] == 2) & (df["candidate_rank"].isin([3, 6, 9]))]
    d2k3 = df[(df["d"] == 2) & (df["k"] == 3) & (df["candidate_rank"].isin([2, 4, 8]))]

    report = f"""# Central Period-Index Benchmark Report

This report is generated by `experiments/period_index_central_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Theorem Statement

For the `k`-pair finite Heisenberg projective system with primitive `d`-th root
`zeta`, the central projective cocycle has period `d` and minimal projective
representation dimension, or benchmark index, `d^k`.

## Proof Sketch

- For `k=1`, the determinant obstruction for `U V = zeta V U` forces `d | r`,
  and the clock-shift construction realizes rank `d`.
- For general `k`, the tensor product of `k` independent clock-shift pairs on
  `(C^d)^{{tensor k}}` gives dimension `d^k`.
- Equivalently, the finite Heisenberg/Stone-von-Neumann representation theory
  gives irreducible projective representation dimension `d^k`.
- The benchmark uses this as a controlled period-index model: ranks divisible
  by `d^k` are accepted by direct sums, and other ranks are rejected.

## Outputs

- Rank sweep CSV: `reports/csv/period_index_central_benchmark.csv`
- Summary CSV: `reports/csv/period_index_central_summary.csv`
- LaTeX theorem snippet: `reports/period_index_central_theorem.tex`
- This report: `reports/period_index_central_report.md`

## Main Table

{format_markdown_table(summary.to_dict("records"), summary_columns)}

The `rank_outcomes` column records every candidate rank as `rank:S` for an
exact direct-sum Heisenberg lift and `rank:F` for an obstructed rank.

## Selected Candidate Ranks

{format_markdown_table(_selected_rows(df), selected_columns)}

## Explicit Examples

For `d=3, k=2`, the period is `3` but the index is `9`:

{format_markdown_table(d3k2.to_dict("records"), selected_columns)}

For `d=2, k=3`, the period is `2` but the index is `8`:

{format_markdown_table(d2k3.to_dict("records"), selected_columns)}

## Relation To TwistedMerge++

- C2M3 is untwisted/permutation synchronization.
- The current real MNIST permutation residual artifacts are noncentral
  holonomy diagnostics, not scalar Brauer/projective classes.
- This benchmark tests the central period-index subtheory directly rather than
  claiming that such central classes occur naturally in MNIST/CIFAR residuals.

## Negative Boundaries

- This does not show real MNIST/CIFAR defects are Brauer classes.
- This does not show TwistedMerge++ beats C2M3.
- This does not trivialize the original class on the same cover.
- This is controlled central/projective evidence, absorbed by a finite-rank
  projective/Morita lift.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-multiplier", type=int, default=3)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    df, summary = run_cases(args.max_multiplier)
    csv_dir = args.reports_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "period_index_central_benchmark.csv"
    summary_path = csv_dir / "period_index_central_summary.csv"
    report_path = args.reports_dir / "period_index_central_report.md"
    tex_path = args.reports_dir / "period_index_central_theorem.tex"
    config_path = args.reports_dir / "configs" / "period_index_central_benchmark_config.json"

    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_theorem_tex(tex_path)
    write_report(args, df, summary, report_path)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "args": {"max_multiplier": args.max_multiplier, "reports_dir": str(args.reports_dir)},
            "environment": capture_environment(),
            "commit": git_commit(),
        },
    )
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
