#!/usr/bin/env python
"""Distinguish scalar Brauer/projective twists from noncentral holonomy."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import format_markdown_table  # noqa: E402
from src.noncentral_holonomy import (  # noqa: E402
    classify_matrix_defect,
    classify_mnist_residual_row,
    clock_shift_projective_example,
    cycle_type,
    fixed_point_fraction,
    matrix_centrality_score,
    noncentral_matrix_example,
    permutation_commutator,
    permutation_to_matrix,
    regular_branch_lift,
    s3_noncentral_permutation_example,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def maybe_int(value) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def row_from_classification(
    *,
    family: str,
    example_id: str,
    structure_group: str,
    classification,
    cycle: str,
    notes: str,
    source_setting_id: str = "",
    tmpp_classification: str = "",
    finite_index_candidate_score: float | None = None,
) -> dict:
    return {
        "family": family,
        "example_id": example_id,
        "structure_group": structure_group,
        "commutator_or_defect_type": classification.commutator_or_defect_type,
        "classification": classification.classification,
        "centrality_score": classification.centrality_score,
        "phase_residual": classification.phase_residual,
        "detected_order_d": classification.detected_order_d,
        "is_scalar_finite_index_candidate": classification.is_scalar_finite_index_candidate,
        "is_noncentral_holonomy": classification.is_noncentral_holonomy,
        "possible_resolution": classification.possible_resolution,
        "brauer_interpretation": classification.brauer_interpretation,
        "cycle_type": cycle,
        "source_setting_id": source_setting_id,
        "tmpp_classification": tmpp_classification,
        "finite_index_candidate_score": finite_index_candidate_score,
        "notes": notes,
    }


def central_clock_shift_rows(max_order: int) -> list[dict]:
    rows = []
    for order in [2, 3, 4]:
        example = clock_shift_projective_example(order)
        classification = classify_matrix_defect(
            example["commutator"],
            structure_group="monomial_phase_or_GL_projective",
            max_order=max_order,
        )
        rows.append(
            row_from_classification(
                family="central_clock_shift",
                example_id=f"clock_shift_order_{order}",
                structure_group="monomial phase / GL_h projective",
                classification=classification,
                cycle="scalar phase",
                notes=(
                    "Clock-shift relation has scalar root-of-unity commutator; "
                    "this is the finite-index projective positive control."
                ),
                finite_index_candidate_score=classification.centrality_score
                + (classification.phase_residual or 0.0),
            )
        )
    return rows


def s3_rows(max_order: int) -> list[dict]:
    example = s3_noncentral_permutation_example()
    comm = example["commutator"]
    classification = classify_matrix_defect(
        permutation_to_matrix(comm),
        structure_group="S_3 permutation",
        max_order=max_order,
    )
    lift = regular_branch_lift([example["p"], example["q"]])
    notes = (
        "p=(12), q=(23); commutator is a 3-cycle and is not central in S_3. "
        f"A regular branch lift exists with label {lift.label}, group size {lift.group_size}; "
        "it is extra capacity, not a scalar Brauer/projective lift."
    )
    return [
        row_from_classification(
            family="noncentral_S3_permutation",
            example_id="s3_transposition_commutator",
            structure_group="S_3 permutation",
            classification=classification,
            cycle=str(cycle_type(comm)),
            notes=notes,
            finite_index_candidate_score=classification.centrality_score
            + (classification.phase_residual or 0.0),
        )
    ]


def gl_rows(max_order: int) -> list[dict]:
    example = noncentral_matrix_example()
    classification = classify_matrix_defect(
        example["commutator"],
        structure_group="GL_2",
        max_order=max_order,
    )
    scalar = matrix_centrality_score(example["commutator"])[1]
    return [
        row_from_classification(
            family="noncentral_GL_matrix",
            example_id="gl2_shear_commutator",
            structure_group="GL_2",
            classification=classification,
            cycle="not_applicable",
            notes=f"Two simple shear matrices have non-scalar commutator; trace scalar is {scalar.real:.4f}.",
            finite_index_candidate_score=classification.centrality_score
            + (classification.phase_residual or 0.0),
        )
    ]


def mnist_cycle_summary(row: pd.Series) -> str:
    if pd.notna(row.get("permutation_num_cycles")):
        return (
            f"stored summary: cycles={int(row['permutation_num_cycles'])}, "
            f"max_len={int(row['permutation_max_cycle_length'])}, "
            f"avg_len={float(row['permutation_avg_cycle_length']):.4f}, "
            f"fixed={float(row['fixed_point_fraction']):.4f}"
        )
    return "exact cycle decomposition not stored"


def real_mnist_rows(input_csv: Path, max_rows: int) -> tuple[list[dict], str]:
    if not input_csv.exists():
        return [], f"Skipped: input CSV missing at {input_csv}"
    df = pd.read_csv(input_csv)
    real = df[df["source"] == "real_mnist"].copy()
    if real.empty:
        return [], f"Skipped: no real_mnist rows in {input_csv}"
    real["finite_index_candidate_score_numeric"] = pd.to_numeric(
        real["finite_index_candidate_score"],
        errors="coerce",
    )
    sample = real.sort_values("finite_index_candidate_score_numeric").head(max_rows)
    rows = []
    for _, item in sample.iterrows():
        classification = classify_mnist_residual_row(item.to_dict())
        rows.append(
            row_from_classification(
                family="real_mnist_residual_samples",
                example_id=f"{item['setting_id']}:{item['triangle']}",
                structure_group=f"S_{int(item['width'])} permutation",
                classification=classification,
                cycle=mnist_cycle_summary(item),
                source_setting_id=str(item["setting_id"]),
                tmpp_classification=str(item.get("tmpp_classification", "")),
                finite_index_candidate_score=float(item["finite_index_candidate_score_numeric"]),
                notes=(
                    "Read from finite_index_residual_mining.csv; exact defect permutation was not stored, "
                    "so cycle_type reports stored cycle statistics."
                ),
            )
        )
    return rows, f"Loaded {len(sample)} most finite-index-like real MNIST residual rows from {input_csv}"


def table(df: pd.DataFrame, family: str, columns: list[str]) -> str:
    rows = df[df["family"] == family].to_dict("records")
    if not rows:
        return "_No rows._"
    return format_markdown_table(rows, columns)


def write_tex(path: Path) -> None:
    content = r"""\section*{Noncentral Holonomy Versus Brauer-Type Projective Twists}

Central/projective obstructions and noncentral nonabelian holonomy are different
phenomena.  A Brauer-type or projective torsion diagnostic is looking for a
central scalar residual, for example a commutator equal to a root of unity times
the identity.  A permutation-valued residual can instead be noncentral inside
the generated permutation group.  Such a residual may obstruct naive descent or
synchronization, but it is not by itself a scalar Brauer/projective class.

\paragraph{Example.}
In \(S_3\), let \(p=(12)\) and \(q=(23)\).  The commutator
\([p,q]=p q p^{-1}q^{-1}\) is a nontrivial 3-cycle.  Since the center of
\(S_3\) is trivial, this 3-cycle is not central in \(S_3\).  Therefore this
obstruction is a noncentral permutation holonomy example, not a scalar
Brauer/projective class.

Pure permutation synchronization, as in C2M3-style alignment, is an untwisted
or nonabelian gauge-synchronization problem.  TwistedMerge++ extends this by
testing whether the residual descends to a central scalar torsion class after
the structure group has been enlarged to signed, phase, monomial, block, or
projective gauges.

Failure to find finite-index scalar residuals in permutation alignments does
not mean all obstruction theory is trivial.  It means that the detected
residuals are noncentral in the chosen structure group, or that a larger
structure group must be tested before making Brauer/projective claims.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_report(args, df: pd.DataFrame, mnist_status: str, path: Path) -> None:
    command = args.command_string
    control_cols = [
        "example_id",
        "structure_group",
        "classification",
        "centrality_score",
        "phase_residual",
        "detected_order_d",
        "brauer_interpretation",
    ]
    noncentral_cols = [
        "example_id",
        "structure_group",
        "classification",
        "centrality_score",
        "cycle_type",
        "possible_resolution",
        "brauer_interpretation",
    ]
    mnist_cols = [
        "example_id",
        "centrality_score",
        "phase_residual",
        "detected_order_d",
        "classification",
        "tmpp_classification",
        "brauer_interpretation",
    ]
    real = df[df["family"] == "real_mnist_residual_samples"]
    if real.empty:
        mnist_interpretation = "No real MNIST residual rows were available for this ladder run."
    elif bool(real["is_scalar_finite_index_candidate"].any()):
        mnist_interpretation = (
            "At least one sampled MNIST row met the scalar finite-index threshold. "
            "This is descriptive only and should be re-mined with larger structure groups."
        )
    else:
        mnist_interpretation = (
            "The sampled MNIST permutation residuals are better described as noncentral "
            "permutation holonomy than as finite-index scalar/Brauer twists."
        )
    report = f"""# Noncentral Holonomy Ladder Report

This report is generated by `experiments/noncentral_holonomy_ladder.py`.

## Exact Command

```bash
{command}
```

## Commit Hash

`{git_commit()}`

## Conceptual Boundary

Pure permutation residuals live in a nonabelian structure group.  Brauer or
projective torsion diagnostics are central/scalar diagnostics.  Therefore a
noncentral permutation commutator is not a Brauer class.  If a scalar detector
finds nothing, that does not imply there is no obstruction; it may mean the
residual is noncentral in the chosen structure group.

## Structure-Group Ladder

```text
S_h
subset signed permutations
subset monomial phase group U(1)^h semidirect S_h
subset block-unitary or GL_h gauges
subset projective/PGL-type quotient
```

C2M3 works at the permutation/gauge synchronization level.  Finite-index
TwistedMerge works when residuals become scalar projective phases.  Noncentral
holonomy requires either nonabelian synchronization or an explicitly labeled
branch/regular representation lift; the latter is extra capacity unless it is
compressed back to a capacity-matched single model.

## Central Clock-Shift Positive Controls

{table(df, "central_clock_shift", control_cols)}

## Noncentral S3 Permutation Control

{table(df, "noncentral_S3_permutation", noncentral_cols)}

## Noncentral GL Matrix Control

{table(df, "noncentral_GL_matrix", noncentral_cols)}

## Real MNIST Residual Samples

Input status: {mnist_status}

{table(df, "real_mnist_residual_samples", mnist_cols)}

## Interpretation

{mnist_interpretation}

The central clock-shift rows are scalar/projective positive controls.  The
`S_3` row is a noncentral 3-cycle commutator, so it is not a Brauer or scalar
projective class.  The GL row demonstrates the same rejection in matrix form.

The next possible algorithmic improvement is to enlarge the structure group
from permutations to signed, phase, monomial, or block gauges and mine again.
Only residuals that become central/scalar in that enlarged group should be
reported as finite-index projective candidates.

## Negative Boundaries

- This does not prove real neural defects are Brauer classes.
- This does not prove TwistedMerge++ beats C2M3.
- Noncentral branch/regular lifts are extra capacity unless compressed to a single model.
- Pure permutation residuals are not the same as scalar finite-index twists.
- Enlarging the structure group does not automatically reveal finite-index torsion.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--max-real-rows", type=int, default=10)
    parser.add_argument(
        "--finite-index-csv",
        type=Path,
        default=ROOT / "reports" / "csv" / "finite_index_residual_mining.csv",
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])

    rows = []
    rows.extend(central_clock_shift_rows(args.max_order))
    rows.extend(s3_rows(args.max_order))
    rows.extend(gl_rows(args.max_order))
    mnist_rows, mnist_status = real_mnist_rows(args.finite_index_csv, args.max_real_rows)
    rows.extend(mnist_rows)

    df = pd.DataFrame(rows)
    csv_path = args.reports_dir / "csv" / "noncentral_holonomy_ladder.csv"
    report_path = args.reports_dir / "noncentral_holonomy_ladder_report.md"
    tex_path = args.reports_dir / "noncentral_vs_brauer_note.tex"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_report(args, df, mnist_status, report_path)
    write_tex(tex_path)

    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
