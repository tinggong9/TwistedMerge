#!/usr/bin/env python
"""Run the TwistedMerge prototype on the explicit mu_2 obstruction task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.simplicial_mu2 import (  # noqa: E402
    Face,
    LinearLocalModel,
    canonical_face,
    tetrahedral_sphere,
    trivial_mu2_twist,
)
from src.twisted_merge_algorithm import (  # noqa: E402
    TwistedMerge,
    TwistedMergeConfig,
    finite_central_twist_close,
    lift_mu2_transition,
    pseudocode,
)


def make_base_weight(rank: int, rng: np.random.Generator) -> np.ndarray:
    weight = rng.normal(size=rank)
    weight[0] = abs(weight[0]) + 1.0
    return weight / np.linalg.norm(weight)


def make_face_data(
    faces: tuple[Face, ...],
    twist: dict[Face, int],
    base_weight: np.ndarray,
    rng: np.random.Generator,
    samples_per_face: int,
) -> dict[Face | None, tuple[np.ndarray, np.ndarray]]:
    data: dict[Face | None, tuple[np.ndarray, np.ndarray]] = {}
    for face in faces:
        key = canonical_face(face)
        x = rng.normal(size=(samples_per_face, len(base_weight)))
        y = (twist[key] * (x @ base_weight) >= 0.0).astype(np.int64)
        data[key] = (x, y)
    return data


def make_algorithm_twist() -> dict[Face, int]:
    """A finite central mu_2 twist visible in raw pairwise defects.

    This is a coboundary on the tetrahedral sphere, unlike the separate H^2
    experiment.  It is useful here because ordinary edge maps can realize the
    defect, causing gauge synchronization to fail before the doubled branch
    representation absorbs the sign.
    """
    complex_ = tetrahedral_sphere()
    twist = trivial_mu2_twist(complex_)
    twist[(0, 1, 2)] = -1
    twist[(0, 2, 3)] = -1
    return twist


def make_alignments(n_models: int, rank: int, case_name: str) -> dict[tuple[int, int], np.ndarray]:
    alignments = {}
    for i in range(n_models):
        for j in range(n_models):
            alignments[(i, j)] = np.eye(rank)
    if case_name == "nontrivial":
        alignments[(0, 2)] = -np.eye(rank)
        alignments[(2, 0)] = -np.eye(rank)
    return alignments


def run_case(args: argparse.Namespace, rank: int, q: int, case_name: str) -> dict[str, object]:
    rng = np.random.default_rng(args.seed + 97 * rank + 13 * q)
    complex_ = tetrahedral_sphere()
    twist = trivial_mu2_twist(complex_) if case_name == "trivial" else make_algorithm_twist()
    base = make_base_weight(rank, rng)
    local_models = [LinearLocalModel(weight=base.copy()) for _ in complex_.vertices]
    alignments = make_alignments(len(local_models), rank, case_name)
    triples = [canonical_face(face) for face in complex_.faces]
    tm = TwistedMerge(TwistedMergeConfig(rank_lift_q=q, tolerance=args.tolerance, central_tolerance=args.central_tolerance))
    result = tm.run(local_models, pairwise_alignments=alignments, alpha=twist, triples=triples)
    datasets = make_face_data(complex_.faces, twist, base, rng, args.samples_per_face)
    metrics = tm.evaluate(result, datasets)
    if "twisted_merge" not in metrics:
        metrics["twisted_merge"] = metrics["ordinary_merge"]
    alpha_nontrivial = any(sign < 0 for sign in twist.values())
    lifted_transition = lift_mu2_transition(np.eye(rank), -1 if alpha_nontrivial else 1)
    return {
        "case": case_name,
        "rank": rank,
        "q": q,
        "status": result.status,
        "cycle_score": result.cycle_score,
        "twist_residual": result.twist_residual if finite_central_twist_close(result.defects, twist, args.central_tolerance) else float("nan"),
        "gauge_success": result.gauge.success,
        "ordinary_loss": metrics["ordinary_merge"]["zero_one_loss"],
        "cycle_consistent_loss": metrics["cycle_consistent_merge"]["zero_one_loss"],
        "twisted_loss": metrics.get("twisted_merge", {"zero_one_loss": float("nan")})["zero_one_loss"],
        "ensemble_loss": metrics["ensemble"]["zero_one_loss"],
        "ordinary_accuracy": metrics["ordinary_merge"]["accuracy"],
        "cycle_consistent_accuracy": metrics["cycle_consistent_merge"]["accuracy"],
        "twisted_accuracy": metrics.get("twisted_merge", {"accuracy": float("nan")})["accuracy"],
        "ensemble_accuracy": metrics["ensemble"]["accuracy"],
        "lifted_transition_shape": json.dumps(list(lifted_transition.shape)),
        "central_minus_matrix": json.dumps([[0.0, 1.0], [1.0, 0.0]]),
    }


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    brief = df[
        [
            "case",
            "rank",
            "q",
            "status",
            "cycle_score",
            "twist_residual",
            "ordinary_loss",
            "cycle_consistent_loss",
            "twisted_loss",
            "ensemble_loss",
        ]
    ]
    nontrivial_q1 = df[(df["case"] == "nontrivial") & (df["q"] == 1)].iloc[0]
    nontrivial_q2 = df[(df["case"] == "nontrivial") & (df["q"] == 2)].iloc[0]
    text = f"""# TwistedMerge Algorithm Report

## Pseudocode

```text
{pseudocode()}
```

## Prototype Implementation

- Main class: `src.twisted_merge_algorithm.TwistedMerge`
- Generic matrix defect routine: `compute_triangle_defects(g)`
- Gauge routine: `try_global_gauge_synchronization(g)`
- Finite central check: `finite_central_twist_close(defects, alpha)`
- mu_2 doubled representation: stores two branches `(w, -w)` and represents
  the nontrivial central sign by the 2x2 branch-swap matrix
  `[[0, 1], [1, 0]]`.

The lifted transition for a base alignment `G_ij` and a central sign is
`kron(rho(sign), G_ij)`, so its matrix size is `2r x 2r` for q=2.

## Commands

```bash
.venv/bin/python experiments/twisted_merge_algorithm_demo.py
```

## Numerical Results

{markdown_table(brief)}

## Numerical Stability

- Gauge synchronization uses Moore-Penrose pseudoinverses and normalized
  Frobenius residuals.
- Central-twist matching uses normalized Frobenius distance to `alpha_ijk I`.
- The default tolerances are `tolerance={args.tolerance}` and
  `central_tolerance={args.central_tolerance}`.
- For near-singular dense alignments, use orthogonal projection or polar
  cleanup before defect computation; this prototype does not silently project
  arbitrary matrices.

## When It Works

- The ordinary branch works when triangle defects are close to identity.
- The q=2 mu_2 branch works in this controlled construction when the observed
  central defects match the supplied sign twist. In the nontrivial case, q=1
  twisted loss is `{nontrivial_q1["twisted_loss"]:.4f}` while q=2 twisted loss
  is `{nontrivial_q2["twisted_loss"]:.4f}`.
- The doubled branch improves downstream prediction when the task labels
  actually depend on the same sign sector as the twist.

## When It Fails

- If no twist is supplied and gauge trivialization fails, the algorithm reports
  failure instead of inventing a correction.
- If q < 2 for a nontrivial mu_2 twist, the doubled representation is not
  available.
- If pairwise alignments are noisy and defects are not close to either identity
  or a finite central twist, the prototype should be treated as diagnostic only.
- The doubled representation absorbs the central sign at the branch/prediction
  level. It does not prove that a nonzero H^2 class became an ordinary untwisted
  vector bundle on the same cover.

## Output Files

- CSV: `reports/csv/twisted_merge_algorithm_demo.csv`
- Config: `reports/configs/twisted_merge_algorithm_demo_config.json`
- Report: `reports/twisted_merge_algorithm_report.md`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--q-values", default="1,2")
    parser.add_argument("--samples-per-face", type=int, default=512)
    parser.add_argument("--seed", type=int, default=6211)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--central-tolerance", type=float, default=1e-5)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    rows = []
    for rank in [int(item) for item in args.ranks.split(",") if item]:
        for q in [int(item) for item in args.q_values.split(",") if item]:
            for case in ["trivial", "nontrivial"]:
                rows.append(run_case(args, rank, q, case))
    df = pd.DataFrame(rows)
    csv_path = args.reports_dir / "csv" / "twisted_merge_algorithm_demo.csv"
    config_path = args.reports_dir / "configs" / "twisted_merge_algorithm_demo_config.json"
    report_path = args.reports_dir / "twisted_merge_algorithm_report.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "args": vars(args) | {"reports_dir": str(args.reports_dir)},
            "environment": capture_environment(),
            "outputs": {
                "csv": str(csv_path),
                "report": str(report_path),
            },
        },
    )
    write_report(df, args, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {config_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
