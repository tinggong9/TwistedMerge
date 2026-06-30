#!/usr/bin/env python
"""TwistedMerge++ finite-index projective residual demo."""

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

from src.finite_index_twists import clock_matrix, root_of_unity, shift_matrix, torsion_order  # noqa: E402
from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402
from src.twisted_merge_plus import TwistedMergePlus  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def projective_pairwise(q: int, exponent: int) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    order = torsion_order(q, exponent)
    zeta = root_of_unity(q, exponent)
    U = clock_matrix(order, zeta)
    V = shift_matrix(order)
    return {
        (0, 0): np.eye(order, dtype=complex),
        (1, 1): np.eye(order, dtype=complex),
        (2, 2): np.eye(order, dtype=complex),
        (0, 1): U,
        (1, 2): V,
        (2, 0): np.linalg.inv(U) @ np.linalg.inv(V),
    }, order


def random_noncentral_pairwise() -> dict[tuple[int, int], np.ndarray]:
    matrix = np.array(
        [
            [1.0, 0.4, 0.0],
            [0.0, 1.0, 0.2],
            [0.1, 0.0, 1.0],
        ]
    )
    return {
        (0, 0): np.eye(3),
        (1, 1): np.eye(3),
        (2, 2): np.eye(3),
        (0, 1): matrix,
        (1, 2): np.eye(3),
        (2, 0): np.eye(3),
    }


def identity_permutations(n_models: int, width: int) -> dict[tuple[int, int], np.ndarray]:
    return {
        (i, j): np.arange(width)
        for i in range(n_models)
        for j in range(n_models)
    }


def c2m3_edge_noise() -> dict[tuple[int, int], np.ndarray]:
    pairwise = identity_permutations(4, 6)
    swap = np.arange(6)
    swap[0], swap[1] = swap[1], swap[0]
    pairwise[(0, 1)] = swap
    pairwise[(1, 0)] = swap
    return pairwise


def pack_row(scenario: str, result, candidate_lift_rank: int, notes: str) -> dict:
    finite = result.diagnostics.finite_index
    return {
        "scenario": scenario,
        "detected_order_d": result.diagnostics.root_order_d,
        "candidate_lift_rank": candidate_lift_rank,
        "rank_divisible_by_order": result.diagnostics.rank_divisible_by_order,
        "determinant_obstruction_allows": result.diagnostics.determinant_obstruction_allows,
        "classification": result.diagnostics.classification,
        "status": result.status,
        "selected_method": result.selected_method,
        "recommended_min_lift_rank": result.diagnostics.recommended_min_lift_rank,
        "lift_residual": result.diagnostics.finite_index_lift_residual,
        "centrality_score": finite.centrality_score if finite is not None else result.diagnostics.centrality_score,
        "phase_residual": result.diagnostics.phase_residual,
        "c2m3_residual": result.diagnostics.c2m3_residual,
        "cycle_score": result.diagnostics.cycle_score,
        "notes": notes + " " + " ".join(result.notes),
    }


def scenario_rows() -> list[dict]:
    tmpp = TwistedMergePlus()
    rows: list[dict] = []

    pairwise, width = projective_pairwise(3, 1)
    result = tmpp.run(pairwise, n_models=3, width=width, candidate_lift_rank=2)
    rows.append(pack_row("E1_order3_rank2_obstructed", result, 2, "expected order 3 rejected at rank 2"))

    result = tmpp.run(pairwise, n_models=3, width=width, candidate_lift_rank=3)
    rows.append(pack_row("E2_order3_rank3_lift", result, 3, "expected order 3 accepted at rank 3"))

    pairwise, width = projective_pairwise(6, 2)
    result = tmpp.run(pairwise, n_models=3, width=width, candidate_lift_rank=3)
    rows.append(pack_row("E3_nonprimitive_q6_a2_rank3_lift", result, 3, "q=6,a=2 has detected order 3"))

    pairwise, width = projective_pairwise(6, 3)
    result = tmpp.run(pairwise, n_models=3, width=width, candidate_lift_rank=2)
    rows.append(pack_row("E4_q6_a3_order2_rank2_lift", result, 2, "q=6,a=3 has detected order 2"))

    result = tmpp.run(random_noncentral_pairwise(), n_models=3, width=3, candidate_lift_rank=3)
    rows.append(pack_row("E5_random_noncentral_rejected", result, 3, "random noncentral residual should not be finite-index"))

    result = tmpp.run(c2m3_edge_noise(), n_models=4, width=6, candidate_lift_rank=3)
    rows.append(pack_row("E6_c2m3_edge_noise_priority", result, 3, "C2M3-fixable edge noise keeps C2M3 priority"))
    return rows


def write_report(args, df: pd.DataFrame, path: Path) -> None:
    columns = [
        "scenario",
        "detected_order_d",
        "candidate_lift_rank",
        "rank_divisible_by_order",
        "determinant_obstruction_allows",
        "classification",
        "selected_method",
        "recommended_min_lift_rank",
        "lift_residual",
        "centrality_score",
        "phase_residual",
    ]
    report = f"""# TwistedMerge++ Finite-Index Report

This report is generated by `experiments/twisted_merge_plus_finite_index_demo.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Scenario Table

{format_markdown_table(df.to_dict("records"), columns)}

## Interpretation

- `finite_index_projective_obstructed` means the residual is close to a scalar finite root-of-unity phase, but the candidate lift rank fails the determinant threshold.
- `finite_index_projective_lift` means the scalar phase order `d` was detected and the candidate rank is divisible by `d`; the selected method is a finite-rank projective/Morita lift.
- The lift absorbs the projective/torsion residual in the lifted representation.  It is not a claim that the original cohomology class vanishes on the same cover.
- C2M3/edge-outlier handling still has priority over finite-index language when the residual is a C2M3-fixable permutation issue, as in scenario E6.
- Random noncentral residuals are rejected rather than described with finite-index language.

## Relation To Existing TwistedMerge++

C2M3 remains the trivial/resolved-cocycle case.  Central coboundary lifting remains the `rho(beta_ij) tensor G_ij` edge-cochain case.  The new finite-index branch is separate: it detects a scalar projective phase and applies the determinant rank threshold before selecting a clock-shift/direct-sum projective lift.

## Negative Boundaries

- This does not show that real neural model-merging defects have finite-index clock-shift form.
- This does not show TwistedMerge++ beats C2M3 on MNIST/CIFAR.
- This does not make finite-index/projective or branch lifts capacity-matched single merged models.

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
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    df = pd.DataFrame(scenario_rows())
    csv_path = args.reports_dir / "csv" / "twisted_merge_plus_finite_index_demo.csv"
    report_path = args.reports_dir / "twisted_merge_plus_finite_index_report.md"
    config_path = args.reports_dir / "configs" / "twisted_merge_plus_finite_index_demo_config.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "environment": capture_environment(),
            "commit": git_commit(),
        },
    )
    write_report(args, df, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
