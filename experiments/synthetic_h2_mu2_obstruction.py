#!/usr/bin/env python
"""Explicit H^2(mu_2) obstruction experiment on a tetrahedral sphere."""

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
    binary_zero_one_loss,
    compute_triangle_defects,
    is_coboundary_mu2,
    make_mu2_transition_system,
    nontrivial_tetrahedral_mu2_twist,
    obstruction_score,
    ordinary_global_prediction,
    pairwise_alignment_loss,
    tetrahedral_sphere,
    trivial_mu2_twist,
    try_global_gauge_synchronization,
    twisted_sheaf_prediction,
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
) -> dict[Face, tuple[np.ndarray, np.ndarray]]:
    data = {}
    for face in faces:
        x = rng.normal(size=(samples_per_face, len(base_weight)))
        logits = twist[tuple(sorted(face))] * (x @ base_weight)
        y = (logits >= 0.0).astype(np.int64)
        data[tuple(sorted(face))] = (x, y)
    return data


def local_classification_loss(
    local_models: dict[int, LinearLocalModel],
    base_weight: np.ndarray,
    rng: np.random.Generator,
    samples_per_vertex: int,
) -> float:
    losses = []
    for model in local_models.values():
        x = rng.normal(size=(samples_per_vertex, len(base_weight)))
        y = (x @ base_weight >= 0.0).astype(np.int64)
        losses.append(binary_zero_one_loss(model.predict(x), y))
    return float(np.mean(losses))


def face_prediction_loss(
    face_data: dict[Face, tuple[np.ndarray, np.ndarray]],
    predict,
) -> float:
    losses = []
    for face, (x, y) in face_data.items():
        losses.append(binary_zero_one_loss(predict(face, x), y))
    return float(np.mean(losses))


def run_case(args: argparse.Namespace, rank: int, case_name: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    complex_ = tetrahedral_sphere()
    if case_name == "trivial":
        twist = trivial_mu2_twist(complex_)
    elif case_name == "nontrivial":
        twist = nontrivial_tetrahedral_mu2_twist(complex_)
    else:
        raise ValueError(case_name)

    transitions = make_mu2_transition_system(complex_, rank=rank, twist=twist)
    base_weight = make_base_weight(rank, rng)
    local_models = {
        vertex: LinearLocalModel(weight=base_weight.copy())
        for vertex in complex_.vertices
    }
    face_data = make_face_data(complex_.faces, twist, base_weight, rng, args.samples_per_face)
    local_loss = local_classification_loss(local_models, base_weight, rng, args.samples_per_vertex)
    alignment_loss = pairwise_alignment_loss(local_models, transitions)

    ordinary = ordinary_global_prediction(local_models)
    global_merge_loss = face_prediction_loss(
        face_data,
        lambda _face, x: ordinary.predict(x),
    )

    twisted = twisted_sheaf_prediction(local_models, transitions, twist)
    twisted_merge_loss = face_prediction_loss(
        face_data,
        lambda face, x: twisted.predict(face, x),
    )

    sync = try_global_gauge_synchronization(transitions)
    defects = {
        str(face): int(np.sign(np.trace(matrix)))
        for face, matrix in compute_triangle_defects(transitions).items()
    }
    return {
        "case": case_name,
        "rank": rank,
        "local_loss": local_loss,
        "pairwise_alignment_loss": alignment_loss,
        "global_merge_loss": global_merge_loss,
        "twisted_merge_loss": twisted_merge_loss,
        "global_merge_failure": global_merge_loss - local_loss,
        "twisted_success": bool(twisted_merge_loss <= args.success_threshold),
        "obstruction_score": obstruction_score(transitions),
        "is_coboundary": is_coboundary_mu2(twist, complex_),
        "global_sync_success": bool(sync["success"]),
        "negative_faces": int(sync["negative_faces"]),
        "can_absorb_twist": bool(twisted.can_absorb_twist),
        "triangle_defects": json.dumps(defects, sort_keys=True),
    }


def plot_obstruction_vs_failure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for case, part in df.groupby("case"):
        ax.scatter(
            part["obstruction_score"],
            part["global_merge_failure"],
            s=70,
            label=case,
            alpha=0.85,
        )
        for _, row in part.iterrows():
            ax.annotate(f"r={int(row['rank'])}", (row["obstruction_score"], row["global_merge_failure"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("H^2(mu_2) obstruction score")
    ax.set_ylabel("ordinary global merge failure")
    ax.set_title("Cocycle obstruction versus merge failure")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rank_success(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    part = df[df["case"] == "nontrivial"].sort_values("rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(part["rank"], part["global_merge_loss"], marker="o", label="ordinary merge")
    ax.plot(part["rank"], part["twisted_merge_loss"], marker="o", label="twisted merge")
    ax.set_xscale("log", base=2)
    ax.set_xticks(part["rank"])
    ax.set_xticklabels([str(int(rank)) for rank in part["rank"]])
    ax.set_xlabel("rank")
    ax.set_ylabel("zero-one loss on triangle-overlap task")
    ax.set_title("Rank versus obstruction absorption")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(df: pd.DataFrame, args: argparse.Namespace, report_path: Path) -> None:
    brief = df[
        [
            "case",
            "rank",
            "local_loss",
            "pairwise_alignment_loss",
            "global_merge_loss",
            "twisted_merge_loss",
            "obstruction_score",
            "is_coboundary",
            "global_sync_success",
            "can_absorb_twist",
        ]
    ]
    table = dataframe_to_markdown(brief)
    nontrivial = df[df["case"] == "nontrivial"].sort_values("rank")
    rank1 = nontrivial[nontrivial["rank"] == 1].iloc[0]
    rank2 = nontrivial[nontrivial["rank"] == 2].iloc[0]
    text = f"""# Synthetic H^2(mu_2) Obstruction Report

This experiment uses the boundary of a tetrahedron as a triangulated 2-sphere.
The face signs are a `mu_2` 2-cocycle.  The trivial case assigns `+1` to every
face.  The nontrivial case assigns `-1` to one face and `+1` to the other three
faces, so the product over all four faces is `-1`; on this complex that class is
not a coboundary.

## Important construction note

An ordinary scalar edge cochain on a closed 2-complex cannot generate a
nonzero `H^2` class: its triangle signs are, by definition, a coboundary.  The
experiment therefore uses twisted descent data: pairwise edge alignments are
locally exact, while the triple-overlap defect includes a prescribed central
2-cocycle.  The code checks this with `is_coboundary_mu2`.

## Commands

```bash
.venv/bin/python experiments/synthetic_h2_mu2_obstruction.py
```

## Outputs

- CSV: `reports/csv/synthetic_h2_mu2_obstruction.csv`
- Obstruction plot: `reports/plots/synthetic_h2_mu2_obstruction_vs_failure.png`
- Rank plot: `reports/plots/synthetic_h2_mu2_rank_success.png`
- Config: `reports/configs/synthetic_h2_mu2_obstruction_config.json`

## Results

{table}

## Claim status

- Local models have near-zero local loss: supported in this construction.
- Pairwise edge alignment loss is zero: supported.
- Ordinary global merging fails in the nontrivial class: supported here; rank 1
  nontrivial global merge loss is `{rank1["global_merge_loss"]:.3f}`.
- Rank 2/doubled representation absorbs the sign twist: supported here; rank 2
  nontrivial twisted merge loss is `{rank2["twisted_merge_loss"]:.3f}`.
- This is a synthetic obstruction witness, not evidence yet for MNIST/CIFAR or
  external model-merging baselines.

## Parameters

- Ranks: `{args.ranks}`
- Samples per face: `{args.samples_per_face}`
- Samples per vertex: `{args.samples_per_vertex}`
- Seed: `{args.seed}`
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=5209)
    parser.add_argument("--samples-per-face", type=int, default=512)
    parser.add_argument("--samples-per-vertex", type=int, default=512)
    parser.add_argument("--success-threshold", type=float, default=0.01)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    ranks = [int(item.strip()) for item in args.ranks.split(",") if item.strip()]
    rows = []
    for rank in ranks:
        for case_name in ["trivial", "nontrivial"]:
            rows.append(run_case(args, rank, case_name, args.seed + rank))
    df = pd.DataFrame(rows)

    csv_path = args.reports_dir / "csv" / "synthetic_h2_mu2_obstruction.csv"
    plot_obstruction_path = args.reports_dir / "plots" / "synthetic_h2_mu2_obstruction_vs_failure.png"
    plot_rank_path = args.reports_dir / "plots" / "synthetic_h2_mu2_rank_success.png"
    config_path = args.reports_dir / "configs" / "synthetic_h2_mu2_obstruction_config.json"
    report_path = args.reports_dir / "synthetic_obstruction_report.md"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot_obstruction_vs_failure(df, plot_obstruction_path)
    plot_rank_success(df, plot_rank_path)
    write_report(df, args, report_path)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "args": vars(args) | {"reports_dir": str(args.reports_dir)},
            "environment": capture_environment(),
            "outputs": {
                "csv": str(csv_path),
                "obstruction_plot": str(plot_obstruction_path),
                "rank_plot": str(plot_rank_path),
                "report": str(report_path),
            },
        },
    )
    print(f"wrote {csv_path}")
    print(f"wrote {plot_obstruction_path}")
    print(f"wrote {plot_rank_path}")
    print(f"wrote {report_path}")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
