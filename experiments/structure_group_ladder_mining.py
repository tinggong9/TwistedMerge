#!/usr/bin/env python
"""Run structure-group ladder diagnostics on controls and MNIST MLPs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.finite_index_twists import clock_matrix, root_of_unity, shift_matrix  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    collect_features,
    device_from_arg,
    format_markdown_table,
    load_dataset,
    make_loader,
    make_model,
    set_seed,
    train_model,
)
from src.noncentral_holonomy import compose_permutations, invert_permutation, noncentral_matrix_example  # noqa: E402
from src.structure_group_ladder import (  # noqa: E402
    LadderDiagnostics,
    LadderResult,
    StructureGroupLadderMerge,
    estimate_pairwise_permutations_from_activations,
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=complex,
    )


def block_diag(mats: list[np.ndarray]) -> np.ndarray:
    size = sum(mat.shape[0] for mat in mats)
    out = np.zeros((size, size), dtype=complex)
    cursor = 0
    for mat in mats:
        n = mat.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = mat
        cursor += n
    return out


def s3_pairwise() -> dict[tuple[int, int], np.ndarray]:
    p = np.array([1, 0, 2])
    q = np.array([0, 2, 1])
    tail = np.array(compose_permutations(invert_permutation(p), invert_permutation(q)))
    return {
        (0, 0): np.arange(3),
        (1, 1): np.arange(3),
        (2, 2): np.arange(3),
        (0, 1): p,
        (1, 2): q,
        (2, 0): tail,
    }


def signed_mu2_pairwise() -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    permutation = {(i, j): np.arange(2) for i in range(3) for j in range(3)}
    signed = {
        (0, 0): np.eye(2),
        (1, 1): np.eye(2),
        (2, 2): np.eye(2),
        (0, 1): np.eye(2),
        (1, 2): np.eye(2),
        (2, 0): -np.eye(2),
    }
    return permutation, signed


def clock_shift_pairwise(order: int = 3) -> dict[tuple[int, int], np.ndarray]:
    zeta = root_of_unity(order, 1)
    U = clock_matrix(order, zeta)
    V = shift_matrix(order)
    return {
        (0, 0): np.eye(order, dtype=complex),
        (1, 1): np.eye(order, dtype=complex),
        (2, 2): np.eye(order, dtype=complex),
        (0, 1): U,
        (1, 2): V,
        (2, 0): np.linalg.inv(U) @ np.linalg.inv(V),
    }


def block_rotation_pairwise() -> dict[tuple[int, int], np.ndarray]:
    A = block_diag([rotation(0.4), rotation(-0.25)])
    B = A.conj().T
    return {
        (0, 0): np.eye(4, dtype=complex),
        (1, 1): np.eye(4, dtype=complex),
        (2, 2): np.eye(4, dtype=complex),
        (0, 1): A,
        (1, 2): B,
        (2, 0): np.eye(4, dtype=complex),
    }


def random_gl_pairwise() -> dict[tuple[int, int], np.ndarray]:
    example = noncentral_matrix_example()
    A = example["A"]
    B = example["B"]
    return {
        (0, 0): np.eye(2, dtype=complex),
        (1, 1): np.eye(2, dtype=complex),
        (2, 2): np.eye(2, dtype=complex),
        (0, 1): A,
        (1, 2): B,
        (2, 0): np.linalg.inv(A) @ np.linalg.inv(B),
    }


def row_from_diag(
    *,
    source: str,
    setting_id: str,
    n_models: int,
    width: int,
    seed: int,
    triangle: str,
    result: LadderResult,
    diag: LadderDiagnostics,
) -> dict:
    return {
        "setting_id": setting_id,
        "source": source,
        "level": diag.level,
        "n_models": n_models,
        "width": width,
        "seed": seed,
        "triangle": triangle,
        "cycle_score": diag.cycle_score,
        "centrality_score": diag.centrality_score,
        "phase_residual": diag.phase_residual,
        "detected_order_d": diag.detected_order_d,
        "rank_allowed": diag.rank_allowed,
        "residual_type": diag.residual_type,
        "selected_resolution": diag.selected_resolution,
        "centrality_improvement_from_previous_level": diag.centrality_improvement_from_previous_level,
        "supports_brauer_projective_interpretation": diag.supports_brauer_projective_interpretation,
        "is_finite_index_candidate": diag.is_finite_index_candidate,
        "final_decision": result.final_decision,
        "selected_level": result.selected_level,
        "notes": " ".join(diag.notes),
    }


def rows_from_result(
    *,
    source: str,
    setting_id: str,
    n_models: int,
    width: int,
    seed: int,
    triangle: tuple[int, int, int],
    result: LadderResult,
) -> list[dict]:
    label = "-".join(str(x) for x in triangle)
    return [
        row_from_diag(
            source=source,
            setting_id=setting_id,
            n_models=n_models,
            width=width,
            seed=seed,
            triangle=label,
            result=result,
            diag=diag,
        )
        for diag in result.diagnostics
    ]


def controlled_rows(max_order: int) -> list[dict]:
    ladder = StructureGroupLadderMerge(max_order=max_order)
    rows: list[dict] = []

    examples = [
        (
            "permutation_S3_noncentral",
            {"permutation": s3_pairwise()},
            3,
            3,
            3,
        ),
        (
            "signed_mu2_central",
            dict(zip(["permutation", "signed_permutation"], signed_mu2_pairwise(), strict=True)),
            3,
            2,
            2,
        ),
        (
            "clock_shift_order_3_rank2",
            {"monomial_phase_or_scale": clock_shift_pairwise(3)},
            3,
            3,
            2,
        ),
        (
            "clock_shift_order_3_rank3",
            {"monomial_phase_or_scale": clock_shift_pairwise(3)},
            3,
            3,
            3,
        ),
        (
            "block_rotation_synthetic_only",
            {"block_orthogonal": block_rotation_pairwise()},
            3,
            4,
            4,
        ),
        (
            "random_GL_noncentral",
            {"low_rank_GL": random_gl_pairwise()},
            3,
            2,
            2,
        ),
    ]
    for setting_id, levels, n_models, width, candidate_rank in examples:
        result = ladder.run(
            levels,
            n_models=n_models,
            width=width,
            candidate_lift_rank=candidate_rank,
            triples=[(0, 1, 2)],
        )
        rows.extend(
            rows_from_result(
                source="synthetic",
                setting_id=setting_id,
                n_models=n_models,
                width=width,
                seed=-1,
                triangle=(0, 1, 2),
                result=result,
            )
        )
    return rows


def run_real_setting(args, spec, train_data, seed: int, n_models: int, width: int) -> list[dict]:
    device = device_from_arg(args.device)
    models = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        model = make_model("mlp", spec, width)
        train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=model_seed + 11)
        train_model(model, train_loader, args.epochs, args.lr, device)
        models.append(model)

    match_loader = make_loader(train_data, args.batch_size, shuffle=False, seed=args.dataset_seed + 501)
    activations = {
        idx: collect_features(model, match_loader, device)
        for idx, model in enumerate(models)
    }
    for model in models:
        model.to("cpu")

    pairwise = estimate_pairwise_permutations_from_activations(activations, n_models, width)
    ladder = StructureGroupLadderMerge(max_order=args.max_order)
    rows = []
    setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
    for triangle in combinations(range(n_models), 3):
        result = ladder.run(
            {"permutation": pairwise},
            n_models=n_models,
            width=width,
            activations=activations,
            candidate_lift_rank=width,
            triples=[tuple(triangle)],
        )
        rows.extend(
            rows_from_result(
                source="real_mnist",
                setting_id=setting_id,
                n_models=n_models,
                width=width,
                seed=seed,
                triangle=tuple(triangle),
                result=result,
            )
        )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, level), group in df.groupby(["source", "level"], dropna=False):
        central = group["supports_brauer_projective_interpretation"].fillna(False).astype(bool)
        noncentral = group["residual_type"].astype(str).str.contains("noncentral")
        not_eval = group["residual_type"].eq("not_evaluated")
        improvements = pd.to_numeric(group["centrality_improvement_from_previous_level"], errors="coerce")
        residual_counts = group["residual_type"].value_counts(dropna=False)
        most_common = str(residual_counts.index[0]) if not residual_counts.empty else "none"
        if not_eval.all():
            interpretation = "not evaluated for this source/level"
        elif central.any() and source == "real_mnist":
            interpretation = "descriptive central candidate found; no merge improvement claimed"
        elif noncentral.mean() >= 0.5:
            interpretation = "mostly noncentral holonomy under this tested structure group"
        else:
            interpretation = "controlled or diagnostic residual reduction"
        rows.append(
            {
                "source": source,
                "level": level,
                "n_rows": int(len(group)),
                "mean_centrality_score": float(pd.to_numeric(group["centrality_score"], errors="coerce").mean()),
                "min_centrality_score": float(pd.to_numeric(group["centrality_score"], errors="coerce").min()),
                "fraction_central_projective_candidates": float(central.mean()),
                "fraction_noncentral_holonomy": float(noncentral.mean()),
                "fraction_not_evaluated": float(not_eval.mean()),
                "mean_centrality_improvement_from_previous_level": float(improvements.mean()),
                "most_common_residual_type": most_common,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    return format_markdown_table(rows, columns)


def finite_float(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    controlled = df[df["source"] == "synthetic"].copy()
    real = df[df["source"] == "real_mnist"].copy()
    controlled_selected = controlled[controlled["residual_type"] != "not_evaluated"].to_dict("records")
    real_summary = summary[summary["source"] == "real_mnist"].to_dict("records")
    real_examples = []
    if not real.empty:
        for level_name, group in real[real["residual_type"] != "not_evaluated"].groupby("level"):
            clean = group[pd.to_numeric(group["centrality_score"], errors="coerce").notna()]
            if not clean.empty:
                real_examples.extend(clean.sort_values("centrality_score").head(3).to_dict("records"))
    real_projective_fraction = (
        float(real["supports_brauer_projective_interpretation"].fillna(False).astype(bool).mean())
        if not real.empty
        else 0.0
    )
    if real.empty:
        interpretation = "Real MNIST mining was skipped."
    elif real_projective_fraction > 0:
        interpretation = (
            "Some real MNIST rows passed central/projective diagnostics.  This is descriptive only; "
            "no practical merge improvement or Brauer-class claim is made without replication."
        )
    else:
        interpretation = (
            "In this run, real MNIST residuals did not become valid scalar finite-index/projective "
            "candidates under the tested signed, monomial, or low-rank GL diagnostics."
        )

    control_cols = [
        "setting_id",
        "level",
        "residual_type",
        "centrality_score",
        "phase_residual",
        "detected_order_d",
        "rank_allowed",
        "selected_resolution",
    ]
    summary_cols = [
        "source",
        "level",
        "n_rows",
        "mean_centrality_score",
        "min_centrality_score",
        "fraction_central_projective_candidates",
        "fraction_noncentral_holonomy",
        "fraction_not_evaluated",
        "mean_centrality_improvement_from_previous_level",
        "most_common_residual_type",
    ]
    example_cols = [
        "setting_id",
        "level",
        "triangle",
        "centrality_score",
        "phase_residual",
        "detected_order_d",
        "residual_type",
        "selected_resolution",
    ]
    report = f"""# Structure-Group Ladder Report

This report is generated by `experiments/structure_group_ladder_mining.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Structure-Group Ladder

```text
S_h
subset signed permutations
subset monomial phase/sign/scale gauges
subset block-orthogonal or block-unitary gauges
subset low-rank GL gauges
subset projective/PGL-type quotient
```

Pure permutation residuals may be noncentral because the structure group is
nonabelian.  Brauer/projective interpretations require central/scalar
residuals.  Enlarging from permutations to signed, monomial, block, or GL
gauges can reduce residuals or reveal central structure, but only rows that
pass centrality and root-of-unity checks are labeled projective candidates.

## Controlled Examples

{table(controlled_selected, control_cols)}

## Real MNIST Summary By Level

{table(real_summary, summary_cols)}

## Most Central Real MNIST Rows By Evaluated Level

{table(real_examples, example_cols)}

## Interpretation

{interpretation}

C2M3/permutation synchronization keeps priority when the permutation residual
is already gauge-trivial.  Signed or monomial corrections are heuristic for
real-valued MLP activations.  The low-rank GL level is diagnostic only and does
not claim a practical merged-model improvement.

## Algorithmic Conclusion

- Use C2M3 when the permutation level is resolved.
- Try signed or monomial diagnostics only when they improve centrality and have
  a valid central/torsion interpretation.
- Try finite-index projective lifting only when an order `d > 1` is detected
  and the candidate rank is divisible by `d`.
- Otherwise report noncentral holonomy and avoid Brauer language.

## Negative Boundaries

- This does not prove real neural defects are Brauer classes.
- This does not prove TwistedMerge++ beats C2M3.
- Structure-group enlargement does not automatically reveal finite-index torsion.
- Branch/regular/projective lifts are extra capacity unless compressed.
- No practical merge improvement is claimed here.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1500,1501,1502,1503,1504")
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    rows = controlled_rows(args.max_order)
    if not args.skip_real:
        spec, train_data, _test_data = load_dataset(
            "mnist",
            args.data_dir,
            args.max_train_samples,
            args.max_test_samples,
            args.dataset_seed,
        )
        for seed in parse_csv(args.seeds, int):
            for n_models in parse_csv(args.model_counts, int):
                for width in parse_csv(args.widths, int):
                    rows.extend(run_real_setting(args, spec, train_data, seed, n_models, width))

    df = pd.DataFrame(rows)
    summary = summarize(df)
    csv_dir = args.reports_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "structure_group_ladder_mining.csv"
    summary_path = csv_dir / "structure_group_ladder_summary.csv"
    report_path = args.reports_dir / "structure_group_ladder_report.md"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(args, df, summary, report_path)

    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
