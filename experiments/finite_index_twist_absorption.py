#!/usr/bin/env python
"""Finite-index torsion/projective twist absorption experiment."""

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

from src.finite_index_twists import (  # noqa: E402
    evaluate_rank_absorption,
    finite_torsion_class,
    toy_prediction_losses,
)
from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def default_cases() -> list[tuple[str, int, int]]:
    cases = [(f"primitive_d{d}", d, 1) for d in [2, 3, 4, 5, 6]]
    cases.extend(
        [
            ("nonprimitive_q6_a2_order3", 6, 2),
            ("nonprimitive_q6_a3_order2", 6, 3),
            ("nonprimitive_q8_a2_order4", 8, 2),
            ("nonprimitive_q8_a4_order2", 8, 4),
        ]
    )
    return cases


def run_cases(max_multiplier: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for case_id, q, exponent in default_cases():
        cls = finite_torsion_class(q, exponent)
        for rank in range(1, max_multiplier * cls.order + 1):
            result = evaluate_rank_absorption(q, exponent, rank)
            losses = toy_prediction_losses(q, exponent, rank)
            rows.append(
                {
                    "case_id": case_id,
                    "q": result.q,
                    "exponent_a": result.exponent,
                    "order_d": result.order,
                    "period": cls.period,
                    "expected_index": cls.expected_index,
                    "candidate_rank_r": result.rank,
                    "determinant_obstruction_prediction": result.determinant_allows,
                    "constructed_lift_success": result.constructed_lift_success,
                    "commutator_residual": result.commutator_residual,
                    "is_minimal_success": result.is_minimal_success,
                    "ordinary_untwisted_descent_on_original_rank": result.ordinary_untwisted_descent_on_original_rank,
                    "lift_kind": result.lift_kind,
                    "ordinary_rank_r_loss": losses["ordinary_rank_r_loss"],
                    "lifted_rank_r_loss": losses["lifted_rank_r_loss"],
                    "branch_projective_loss": losses["branch_projective_loss"],
                    "branch_extra_capacity": losses["branch_extra_capacity"],
                    "prediction_note": losses["prediction_note"],
                    "interpretation": result.interpretation,
                }
            )
    df = pd.DataFrame(rows)
    summary_rows = []
    for case_id, group in df.groupby("case_id", sort=False):
        successes = group[group["constructed_lift_success"]]
        failures = group[~group["constructed_lift_success"]]
        rank_outcomes = ",".join(
            f"{int(row.candidate_rank_r)}:{'S' if bool(row.constructed_lift_success) else 'F'}"
            for row in group.itertuples()
        )
        order = int(group["order_d"].iloc[0])
        success_exactly_divisible = bool(
            np.all(group["constructed_lift_success"].to_numpy() == (group["candidate_rank_r"].to_numpy() % order == 0))
        )
        minimal_success_rank = int(successes["candidate_rank_r"].min()) if not successes.empty else -1
        min_failure_residual = float(failures["commutator_residual"].min()) if not failures.empty else float("nan")
        max_success_residual = float(successes["commutator_residual"].max()) if not successes.empty else float("nan")
        summary_rows.append(
            {
                "case_id": case_id,
                "q": int(group["q"].iloc[0]),
                "exponent_a": int(group["exponent_a"].iloc[0]),
                "order_d": order,
                "period": int(group["period"].iloc[0]),
                "expected_index": int(group["expected_index"].iloc[0]),
                "minimal_success_rank": minimal_success_rank,
                "success_exactly_when_d_divides_r": success_exactly_divisible,
                "max_success_residual": max_success_residual,
                "min_failure_residual": min_failure_residual,
                "rank_outcomes": rank_outcomes,
                "theorem_supported": bool(success_exactly_divisible and minimal_success_rank == order and max_success_residual < 1e-10),
                "interpretation": (
                    "success ranks are exactly multiples of the torsion order; "
                    "the first success realizes the period/index threshold"
                ),
            }
        )
    return df, pd.DataFrame(summary_rows)


def write_theorem_tex(path: Path) -> None:
    text = r"""\begin{proposition}[Determinant obstruction for a torsion projective pair]
Let $\zeta$ be a primitive $d$-th root of unity.  Suppose $A,B \in
\mathrm{GL}_r(\mathbb{C})$ satisfy
\[
  A B = \zeta B A .
\]
Then $d$ divides $r$.
\end{proposition}

\begin{proof}
Taking determinants gives
\[
  \det(A)\det(B) = \det(\zeta B A) = \zeta^r \det(B)\det(A).
\]
Since $A$ and $B$ are invertible, $\det(A)\det(B) \ne 0$, hence
$\zeta^r = 1$.  Because $\zeta$ has order $d$, this is equivalent to
$d \mid r$.
\end{proof}

\begin{proposition}[Clock-shift construction]
Let $\zeta$ be a primitive $d$-th root of unity.  Define
\[
  U_d = \mathrm{diag}(1,\zeta,\zeta^2,\ldots,\zeta^{d-1})
\]
and let $V_d$ be the cyclic shift matrix $V_d e_j=e_{j+1}$ with indices
modulo $d$.  Then
\[
  U_d V_d = \zeta V_d U_d .
\]
\end{proposition}

\begin{corollary}
In this toy projective torsion model, the minimal linearization rank equals
the order (period/index) $d$.  Ranks that are multiples of $d$ are obtained by
direct sums of the clock-shift representation.
\end{corollary}

\paragraph{ML interpretation.}
The projective transition relation cannot be represented by an ordinary
rank-$r$ linear feature space unless $d \mid r$.  Passing to a finite-rank
projective or Morita lift of rank $d$ absorbs the torsion defect in the lifted
representation.  This does not mean the original cocycle vanishes on the same
cover, and branch/projective predictors should be labeled as extra-capacity
unless a capacity-matched merged model is constructed.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    summary_columns = [
        "case_id",
        "q",
        "exponent_a",
        "order_d",
        "expected_index",
        "minimal_success_rank",
        "success_exactly_when_d_divides_r",
        "rank_outcomes",
        "theorem_supported",
    ]
    selected_rows = []
    for _case_id, group in df.groupby("case_id", sort=False):
        order = int(group["order_d"].iloc[0])
        selected = group[group["candidate_rank_r"].isin([1, max(order - 1, 1), order, order + 1, 2 * order])]
        selected_rows.extend(selected.drop_duplicates(subset=["case_id", "candidate_rank_r"]).to_dict("records"))
    selected_columns = [
        "case_id",
        "candidate_rank_r",
        "determinant_obstruction_prediction",
        "constructed_lift_success",
        "commutator_residual",
        "is_minimal_success",
        "ordinary_rank_r_loss",
        "lifted_rank_r_loss",
        "branch_projective_loss",
        "branch_extra_capacity",
    ]
    report = f"""# Finite-Index Torsion Twist Absorption Report

This report is generated by `experiments/finite_index_twist_absorption.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Theorem Statement

Let `zeta` be a primitive `d`-th root of unity.  If invertible complex
`r x r` matrices `A` and `B` satisfy `A B = zeta B A`, then `d` divides `r`.
Conversely, the `d x d` clock and shift matrices satisfy the relation, and
direct sums realize every rank that is a multiple of `d`.

## Determinant Proof

Taking determinants gives

```text
det(A) det(B) = det(zeta B A) = zeta^r det(B) det(A).
```

Since `A` and `B` are invertible, `det(A) det(B)` is nonzero, so
`zeta^r = 1`.  For primitive order `d`, this is equivalent to `d | r`.

## Outputs

- Rank sweep CSV: `reports/csv/finite_index_twist_absorption.csv`
- Summary CSV: `reports/csv/finite_index_twist_summary.csv`
- LaTeX theorem snippet: `reports/finite_index_twist_theorem.tex`
- This report: `reports/finite_index_twist_report.md`

## Experimental Cases

{format_markdown_table(summary.to_dict("records"), summary_columns)}

The `rank_outcomes` column records every candidate rank as `rank:S` for an
exact constructed clock-shift/direct-sum lift and `rank:F` for a determinant
obstruction failure.  The full per-rank table is in the CSV.

## Selected Candidate Ranks

{format_markdown_table(selected_rows, selected_columns)}

## Period Versus Index

For the root `exp(2 pi i a / q)`, the effective torsion order is
`d = q / gcd(q, a)`.  The period in this toy model is `d`, and the determinant
argument shows that the minimal linearization rank, or index, is also `d`.
Nonprimitive examples therefore reduce to the order of the root: for example
`q=6, a=2` has order `3`, while `q=6, a=3` has order `2`.

## Interpretation

The lift absorbs the torsion/projective defect in a finite-rank projective or
Morita representation.  It does not show that a nonzero cohomology class
vanishes on the original cover, and it should not be described as ordinary
untwisted descent at the forbidden ranks.

The toy prediction columns are algebraic proxies only: `lifted_rank_r_loss`
is zero exactly when the clock-shift relation is realized; the branch/projective
loss is zero but is explicitly labeled extra capacity.

## Relation To TwistedMerge++

- C2M3 is the trivial-cocycle or synchronization-resolved case.
- Finite-index TwistedMerge is the projective/torsion-lift case: a finite
  rank threshold absorbs an order-`d` projective defect.
- The finite central coboundary examples remain a separate edge-cochain lift.
- Nonzero `H^2` without such a transition-level lift remains branch-only or an
  obstruction witness in the current repository.

## Negative Boundaries

- This experiment does not show that real neural model-merging defects have
  this exact finite-index form.
- It does not show that TwistedMerge beats C2M3 on natural MNIST/CIFAR.
- It does not show that every torsion class in the paper's broad setting is
  trivialized on the original cover.
- It does not make branch/projective prediction a capacity-matched single
  merged model.

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
    results_path = csv_dir / "finite_index_twist_absorption.csv"
    summary_path = csv_dir / "finite_index_twist_summary.csv"
    report_path = args.reports_dir / "finite_index_twist_report.md"
    tex_path = args.reports_dir / "finite_index_twist_theorem.tex"
    config_path = args.reports_dir / "configs" / "finite_index_twist_absorption_config.json"

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
