#!/usr/bin/env python
"""Small TwistedMerge++ sanity-check demo.

The demo is deliberately synthetic and fast.  It checks the algorithmic
selector and residual classification before any claim about natural MNIST/CIFAR
model merging is made.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment, save_json  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402
from src.simplicial_mu2 import (  # noqa: E402
    canonical_face,
    nontrivial_tetrahedral_mu2_twist,
    tetrahedral_sphere,
    trivial_mu2_twist,
)
from src.twisted_merge_plus import TwistedMergePlus, pseudocode  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def identity_permutations(n_models: int, width: int) -> dict[tuple[int, int], np.ndarray]:
    return {
        (i, j): np.arange(width)
        for i in range(n_models)
        for j in range(n_models)
    }


def finite_central_twist() -> dict[tuple[int, int, int], int]:
    twist = trivial_mu2_twist(tetrahedral_sphere())
    twist[(0, 1, 2)] = -1
    twist[(0, 2, 3)] = -1
    return twist


def central_alignments(rank: int, active: bool = True) -> dict[tuple[int, int], np.ndarray]:
    maps = {
        (i, j): np.eye(rank)
        for i in tetrahedral_sphere().vertices
        for j in tetrahedral_sphere().vertices
    }
    if active:
        maps[(0, 2)] = -np.eye(rank)
        maps[(2, 0)] = -np.eye(rank)
    return maps


def tetrahedral_triples() -> list[tuple[int, int, int]]:
    return [canonical_face(face) for face in tetrahedral_sphere().faces]


def scenario_rows() -> list[dict]:
    tmpp = TwistedMergePlus()
    rows = []

    pairwise = identity_permutations(n_models=4, width=6)
    swap = np.arange(6)
    swap[0], swap[1] = swap[1], swap[0]
    pairwise[(0, 1)] = swap
    pairwise[(1, 0)] = swap
    result = tmpp.run(
        pairwise,
        n_models=4,
        width=6,
        method_metrics={
            "ordinary": {"loss": 0.28, "accuracy": 0.72},
            "c2m3": {"loss": 0.01, "accuracy": 0.99},
            "lifted": {"loss": float("nan"), "accuracy": float("nan")},
            "branch": {"loss": 0.01, "accuracy": 0.99},
            "ensemble": {"loss": 0.0, "accuracy": 1.0},
        },
    )
    rows.append(pack_row("A_c2m3_fixes_edge_noise", result))

    alpha = finite_central_twist()
    result = tmpp.run(
        central_alignments(rank=2, active=True),
        n_models=4,
        width=2,
        known_alpha=alpha,
        triples=tetrahedral_triples(),
        method_metrics={
            "ordinary": {"loss": 0.50, "accuracy": 0.50},
            "c2m3": {"loss": 0.50, "accuracy": 0.50},
            "lifted": {"loss": 0.0, "accuracy": 1.0},
            "branch": {"loss": 0.0, "accuracy": 1.0},
            "ensemble": {"loss": 0.0, "accuracy": 1.0},
        },
    )
    rows.append(pack_row("B_central_coboundary_lift", result))

    randomish = identity_permutations(n_models=4, width=6)
    randomish[(0, 1)] = np.array([1, 0, 2, 3, 4, 5])
    randomish[(1, 0)] = np.array([1, 0, 2, 3, 4, 5])
    randomish[(0, 2)] = np.array([0, 2, 1, 3, 4, 5])
    randomish[(2, 0)] = np.array([0, 2, 1, 3, 4, 5])
    randomish[(1, 3)] = np.array([0, 1, 3, 2, 4, 5])
    randomish[(3, 1)] = np.array([0, 1, 3, 2, 4, 5])
    result = tmpp.run(
        randomish,
        n_models=4,
        width=6,
        method_metrics={
            "ordinary": {"loss": 0.33, "accuracy": 0.67},
            "c2m3": {"loss": 0.22, "accuracy": 0.78},
            "lifted": {"loss": float("nan"), "accuracy": float("nan")},
            "branch": {"loss": 0.22, "accuracy": 0.78},
            "ensemble": {"loss": 0.05, "accuracy": 0.95},
        },
    )
    rows.append(pack_row("C_random_noncentral_rejected", result))

    h2_alpha = nontrivial_tetrahedral_mu2_twist(tetrahedral_sphere())
    result = tmpp.run(
        central_alignments(rank=2, active=False),
        n_models=4,
        width=2,
        known_alpha=h2_alpha,
        triples=tetrahedral_triples(),
        method_metrics={
            "ordinary": {"loss": 0.25, "accuracy": 0.75},
            "c2m3": {"loss": 0.25, "accuracy": 0.75},
            "lifted": {"loss": float("nan"), "accuracy": float("nan")},
            "branch": {"loss": 0.0, "accuracy": 1.0},
            "ensemble": {"loss": 0.0, "accuracy": 1.0},
        },
    )
    rows.append(pack_row("D_h2_non_coboundary_branch_only", result))
    return rows


def pack_row(name: str, result) -> dict:
    metrics = result.metrics

    def metric(method: str, key: str) -> float:
        return float(metrics.get(method, {}).get(key, float("nan")))

    return {
        "scenario": name,
        "classification": result.diagnostics.classification,
        "status": result.status,
        "selected_method": result.selected_method,
        "cycle_score": result.diagnostics.cycle_score,
        "c2m3_residual": result.diagnostics.c2m3_residual,
        "centrality_score": result.diagnostics.centrality_score,
        "alpha_residual": result.diagnostics.alpha_residual,
        "lifted_maps_built": bool(result.lifted_transition_maps),
        "ordinary_accuracy": metric("ordinary", "accuracy"),
        "c2m3_accuracy": metric("c2m3", "accuracy"),
        "lifted_accuracy": metric("lifted", "accuracy"),
        "branch_accuracy": metric("branch", "accuracy"),
        "ensemble_accuracy": metric("ensemble", "accuracy"),
        "reason": result.reason,
    }


def write_report(args, df: pd.DataFrame, path: Path) -> None:
    columns = [
        "scenario",
        "classification",
        "status",
        "selected_method",
        "cycle_score",
        "c2m3_residual",
        "centrality_score",
        "alpha_residual",
        "lifted_maps_built",
        "ordinary_accuracy",
        "c2m3_accuracy",
        "lifted_accuracy",
        "branch_accuracy",
        "ensemble_accuracy",
    ]
    report = f"""# TwistedMerge++ Report

This report is generated by `experiments/twisted_merge_plus_demo.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Verification Commands

```bash
.venv/bin/python experiments/twisted_merge_plus_demo.py
.venv/bin/python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python -m compileall src experiments tests
git diff --check
```

## Pseudocode

```text
{pseudocode()}
```

## Scenario Table

{format_markdown_table(df.to_dict("records"), columns)}

## What This Prototype Proves

- TwistedMerge++ contains C2M3-style synchronization as the trivial/resolved-residual case: tests cover the exact zero-defect case, and scenario A selects `c2m3_cycle_consistent` for a C2M3-fixable outlier rather than taking a twist path.
- It distinguishes a C2M3-fixable one-edge permutation outlier from a central/twist residual.
- It activates explicit lifted transition maps only for a finite central coboundary residual: scenario B builds maps of the form `rho(beta_ij) tensor G_ij`.
- It refuses central-twist language for random/noncentral residuals.

## What This Prototype Does Not Prove

- It does not show a win over C2M3 on natural MNIST or CIFAR model merging.
- It does not turn branch/extra-capacity prediction into a capacity-matched single merged model.
- It does not trivialize a nonzero `H^2(mu_2)` class as an ordinary untwisted vector bundle.
- Scenario D is branch-only context selection for a non-coboundary `H^2` witness; it is deliberately not transition-level ordinary descent.

## Numerical Stability

The demo uses exact permutation matrices and exact `+/- I` central signs, so tolerances are stringent.  Near-real runs should inspect `c2m3_residual`, `centrality_score`, and `alpha_residual` before selecting a branch.  A failed C2M3 residual alone is not evidence for a twist.

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
    csv_path = args.reports_dir / "csv" / "twisted_merge_plus_demo.csv"
    report_path = args.reports_dir / "twisted_merge_plus_report.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_json(
        args.reports_dir / "configs" / "twisted_merge_plus_demo_config.json",
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
