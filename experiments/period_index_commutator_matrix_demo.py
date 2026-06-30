#!/usr/bin/env python
"""Central commutator-matrix period-index detector demo."""

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
from src.model_merging_benchmark import format_markdown_table, permutation_matrix  # noqa: E402
from src.period_index_central import clock_matrix, heisenberg_generators, shift_matrix  # noqa: E402
from src.period_index_detector import detect_commutator_matrix_period_index  # noqa: E402
from src.twisted_merge_plus import TwistedMergePlus  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def generator_dict(d: int, k: int) -> dict[str, np.ndarray]:
    system = heisenberg_generators(d, k)
    generators: dict[str, np.ndarray] = {}
    for idx in range(k):
        generators[f"U{idx + 1}"] = system.U[idx]
        generators[f"V{idx + 1}"] = system.V[idx]
    return generators


def shuffled_generator_dict(d: int, k: int) -> dict[str, np.ndarray]:
    system = heisenberg_generators(d, k)
    if d == 2 and k == 3:
        return {
            "A": system.U[1],
            "B": system.U[0],
            "C": system.V[2],
            "D": system.V[0],
            "E": system.U[2],
            "F": system.V[1],
        }
    if d == 3 and k == 2:
        return {"A": system.V[1], "B": system.U[0], "C": system.V[0], "D": system.U[1]}
    raise ValueError("unsupported shuffle")


def rank_deficient_generators(d: int) -> dict[str, np.ndarray]:
    return {
        "A": clock_matrix(d),
        "B": shift_matrix(d),
        "C": np.eye(d, dtype=complex),
        "D": np.eye(d, dtype=complex),
    }


def mixed_period_generators() -> dict[str, np.ndarray]:
    identity3 = np.eye(3, dtype=complex)
    identity4 = np.eye(4, dtype=complex)
    return {
        "U3": np.kron(clock_matrix(3), identity4),
        "V3": np.kron(shift_matrix(3), identity4),
        "U4": np.kron(identity3, clock_matrix(4)),
        "V4": np.kron(identity3, shift_matrix(4)),
    }


def s3_noncentral_generators() -> dict[str, np.ndarray]:
    return {
        "s12": permutation_matrix(np.array([1, 0, 2])),
        "s23": permutation_matrix(np.array([0, 2, 1])),
    }


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


def run_tmpp(generators: dict[str, np.ndarray], rank: int, max_root_order: int = 12):
    width = next(iter(generators.values())).shape[0]
    return TwistedMergePlus().run(
        unresolved_pairwise(width),
        n_models=3,
        width=width,
        period_index_generators=generators,
        candidate_lift_rank=rank,
        max_root_order=max_root_order,
    )


def pack_row(
    scenario: str,
    generators: dict[str, np.ndarray],
    candidate_rank: int,
    *,
    max_root_order: int = 12,
) -> dict:
    detection = detect_commutator_matrix_period_index(
        generators,
        candidate_rank=candidate_rank,
        max_root_order=max_root_order,
    )
    result = run_tmpp(generators, candidate_rank, max_root_order=max_root_order)
    return {
        "scenario": scenario,
        "generator_count": len(generators),
        "detector_mode": detection.detector_mode,
        "period": detection.period,
        "index": detection.index,
        "alternating_rank": detection.alternating_rank,
        "radical_size": detection.radical_size,
        "quotient_size": detection.quotient_size,
        "candidate_rank": detection.candidate_rank,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": detection.decision,
        "selected_method": result.selected_method,
        "classification": result.diagnostics.classification,
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "notes": " ".join(detection.notes),
    }


def scenario_rows() -> list[dict]:
    return [
        pack_row("heisenberg_d2_k2_rank2", generator_dict(2, 2), 2),
        pack_row("heisenberg_d2_k2_rank4", generator_dict(2, 2), 4),
        pack_row("heisenberg_d3_k2_rank3", generator_dict(3, 2), 3),
        pack_row("heisenberg_d3_k2_rank9", generator_dict(3, 2), 9),
        pack_row("rank_deficient_d3_four_generators", rank_deficient_generators(3), 3),
        pack_row("shuffled_generators_d2_k3", shuffled_generator_dict(2, 3), 8),
        pack_row("composite_d4_k1", generator_dict(4, 1), 4),
        pack_row("mixed_period_common_d12_unknown", mixed_period_generators(), 12, max_root_order=4),
        pack_row("noncentral_control", s3_noncentral_generators(), 3),
    ]


def write_report(args, df: pd.DataFrame, path: Path) -> None:
    columns = [
        "scenario",
        "generator_count",
        "detector_mode",
        "period",
        "index",
        "alternating_rank",
        "radical_size",
        "quotient_size",
        "candidate_rank",
        "period_divides_rank",
        "index_divides_rank",
        "decision",
        "selected_method",
        "max_centrality_score",
        "max_phase_residual",
        "notes",
    ]
    rank_deficient = df[df["scenario"] == "rank_deficient_d3_four_generators"]
    shuffled = df[df["scenario"] == "shuffled_generators_d2_k3"]
    obstructed = df[df["scenario"].isin(["heisenberg_d2_k2_rank2", "heisenberg_d3_k2_rank3"])]
    report = f"""# Period-Index Commutator-Matrix Report

This report is generated by `experiments/period_index_commutator_matrix_demo.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Purpose

The previous TwistedMerge++ period-index detector recognized explicit
independent Heisenberg pairs.  This detector computes the central commutator
form directly from supplied generators, so the index is inferred from the
alternating exponent matrix rather than from hard-coded generator labels.

## Mathematical Explanation

For generators `A_i`, central commutators
`A_i A_j A_i^-1 A_j^-1 ~= zeta^e_ij I` define an alternating exponent matrix
`E=(e_ij)` modulo the detected common period `d`.  When `d` is prime, the
certified index is `d^(rank(E)/2)`.  For composite `d`, the implementation uses
a conservative brute-force radical computation on small `(Z/dZ)^m` state
spaces and sets

```text
index = sqrt(|(Z/dZ)^m / radical(E)|)
```

only when the quotient size is certified as a square.

## Scenario Table

{format_markdown_table(df.to_dict("records"), columns)}

## Explicit Examples

Rank-deficient four-generator case: only one independent pair is active, so
the alternating rank is `2` and the index is `3`, not `3^2`:

{format_markdown_table(rank_deficient.to_dict("records"), columns)}

Shuffled generator names do not matter; the commutator matrix still certifies
the `d=2,k=3` index `8`:

{format_markdown_table(shuffled.to_dict("records"), columns)}

Period-divisible but index-obstructed ranks are rejected:

{format_markdown_table(obstructed.to_dict("records"), columns)}

## Algorithmic Conclusion

TwistedMerge++ can now estimate index thresholds from central commutator data,
not only from explicitly labeled independent Heisenberg pairs.  It still
requires index divisibility before selecting a finite-rank projective/Morita
lift.

## Negative Boundaries

- This is controlled central/projective detection.
- Real MNIST/CIFAR residuals are still not shown to be Brauer classes.
- Unknown index cases are not accepted as lifts.
- Composite periods are handled conservatively; unsupported state spaces remain
  `central_projective_index_unknown` rather than overclaimed.

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
    csv_path = args.reports_dir / "csv" / "period_index_commutator_matrix_demo.csv"
    report_path = args.reports_dir / "period_index_commutator_matrix_report.md"
    config_path = args.reports_dir / "configs" / "period_index_commutator_matrix_demo_config.json"
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
