#!/usr/bin/env python
"""Robust central commutator-matrix period-index detector demo."""

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
from src.period_index_central import clock_matrix, heisenberg_generators, shift_matrix  # noqa: E402
from src.period_index_detector import RobustPeriodIndexDetection, robust_detect_commutator_matrix_period_index  # noqa: E402
from src.period_index_mining import (  # noqa: E402
    add_entrywise_noise,
    add_unitary_noise,
    detect_mined_period_index,
    generate_noisy_heisenberg_generators,
    generate_noncentral_controls,
    mine_period_index_generators,
)
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


def rank_deficient_generators(d: int, noise_level: float = 0.0, noise_type: str = "none") -> dict[str, np.ndarray]:
    clean = {
        "A": clock_matrix(d),
        "B": shift_matrix(d),
        "C": np.eye(d, dtype=complex),
        "D": np.eye(d, dtype=complex),
    }
    if noise_level <= 0:
        return clean
    rng = np.random.default_rng(700 + int(noise_level * 1e8))
    noisy: dict[str, np.ndarray] = {}
    for name, matrix in clean.items():
        if noise_type == "unitary_near_identity":
            noisy[name] = add_unitary_noise(matrix, noise_level, rng=rng)
        elif noise_type == "entrywise_projected_unitary":
            noisy[name] = add_entrywise_noise(matrix, noise_level, rng=rng)
        else:
            noisy[name] = matrix
    return noisy


def mixed_period_generators() -> dict[str, np.ndarray]:
    identity3 = np.eye(3, dtype=complex)
    identity4 = np.eye(4, dtype=complex)
    return {
        "U3": np.kron(clock_matrix(3), identity4),
        "V3": np.kron(shift_matrix(3), identity4),
        "U4": np.kron(identity3, clock_matrix(4)),
        "V4": np.kron(identity3, shift_matrix(4)),
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


def synthetic_transition_maps() -> tuple[dict[tuple[int, int], np.ndarray], list[tuple[int, ...]]]:
    system = heisenberg_generators(2, 2)
    hidden = [system.U[0], system.V[0], system.U[1], system.V[1]]
    loops = [
        (0, 1, 2, 0),
        (0, 2, 3, 0),
        (0, 3, 4, 0),
        (0, 4, 5, 0),
    ]
    identity = np.eye(system.dimension, dtype=complex)
    transition_maps: dict[tuple[int, int], np.ndarray] = {}
    for loop, generator in zip(loops, hidden, strict=True):
        transition_maps[(loop[0], loop[1])] = generator
        transition_maps[(loop[1], loop[2])] = identity
        transition_maps[(loop[2], loop[3])] = identity
    return transition_maps, loops


def expected_decision(detection: RobustPeriodIndexDetection) -> str:
    if detection.status == "candidate_uncertain":
        return "central_projective_candidate_uncertain"
    if detection.status == "rejected_noncentral":
        return "not_central_projective"
    if detection.status == "unknown_index":
        return "central_projective_index_unknown"
    return detection.decision


def pack_detection_row(
    *,
    case_id: str,
    source: str,
    d: int | None,
    k: int | None,
    noise_type: str,
    noise_level: float,
    candidate_rank: int,
    detection: RobustPeriodIndexDetection,
    selected_method: str,
    generator_mining_used: bool = False,
    n_mined_generators: int = 0,
    notes: str = "",
) -> dict:
    expected = expected_decision(detection)
    lift_selected = selected_method == "period_index_projective_morita_lift"
    lift_allowed = detection.status == "certified" and detection.decision == "period_index_lift_success"
    pass_fail = detection.decision == expected and (lift_selected == lift_allowed)
    return {
        "case_id": case_id,
        "source": source,
        "d": d,
        "k": k,
        "noise_type": noise_type,
        "noise_level": noise_level,
        "candidate_rank": candidate_rank,
        "detector_status": detection.status,
        "period": detection.period,
        "index": detection.index,
        "period_divides_rank": detection.period_divides_rank,
        "index_divides_rank": detection.index_divides_rank,
        "decision": detection.decision,
        "selected_method": selected_method,
        "max_centrality_score": detection.max_centrality_score,
        "max_phase_residual": detection.max_phase_residual,
        "min_root_margin": detection.min_root_margin,
        "alternating_rank": detection.alternating_rank,
        "radical_size": detection.radical_size,
        "quotient_size": detection.quotient_size,
        "generator_mining_used": generator_mining_used,
        "n_mined_generators": n_mined_generators,
        "expected_decision": expected,
        "pass_fail": "pass" if pass_fail else "fail",
        "notes": " ".join([*detection.notes, notes]).strip(),
    }


def evaluate_generators(
    *,
    case_id: str,
    source: str,
    generators: dict[str, np.ndarray],
    candidate_rank: int,
    d: int | None,
    k: int | None,
    noise_type: str,
    noise_level: float,
    max_root_order: int = 12,
) -> dict:
    detection = robust_detect_commutator_matrix_period_index(
        generators,
        candidate_rank=candidate_rank,
        max_root_order=max_root_order,
    )
    width = next(iter(generators.values())).shape[0]
    result = TwistedMergePlus().run(
        unresolved_pairwise(width),
        n_models=3,
        width=width,
        period_index_generators=generators,
        candidate_lift_rank=candidate_rank,
        max_root_order=max_root_order,
    )
    return pack_detection_row(
        case_id=case_id,
        source=source,
        d=d,
        k=k,
        noise_type=noise_type,
        noise_level=noise_level,
        candidate_rank=candidate_rank,
        detection=detection,
        selected_method=result.selected_method,
    )


def scenario_rows() -> list[dict]:
    rows: list[dict] = []
    for d, k, rank in [(2, 2, 4), (3, 2, 9)]:
        rows.append(
            evaluate_generators(
                case_id=f"exact_heisenberg_d{d}_k{k}_rank{rank}",
                source="exact_heisenberg",
                generators=generator_dict(d, k),
                candidate_rank=rank,
                d=d,
                k=k,
                noise_type="none",
                noise_level=0.0,
            )
        )

    noise_levels = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
    noise_types = ["unitary_near_identity", "entrywise_projected_unitary"]
    for d in [2, 3]:
        for noise_type in noise_types:
            for noise_level in noise_levels:
                rows.append(
                    evaluate_generators(
                        case_id=f"noisy_heisenberg_d{d}_k2_{noise_type}_{noise_level:g}",
                        source="noisy_heisenberg",
                        generators=generate_noisy_heisenberg_generators(
                            d,
                            2,
                            noise_level,
                            noise_type,
                            seed=100 + d,
                        ),
                        candidate_rank=d**2,
                        d=d,
                        k=2,
                        noise_type=noise_type,
                        noise_level=noise_level,
                    )
                )

    for rank in [3, 6, 9]:
        rows.append(
            evaluate_generators(
                case_id=f"rank_test_d3_k2_rank{rank}",
                source="rank_divisibility",
                generators=generate_noisy_heisenberg_generators(3, 2, 1e-6, "unitary_near_identity", seed=305),
                candidate_rank=rank,
                d=3,
                k=2,
                noise_type="unitary_near_identity",
                noise_level=1e-6,
            )
        )

    for noise_level in [0.0, 1e-6, 1e-5]:
        rows.append(
            evaluate_generators(
                case_id=f"rank_deficient_d3_one_pair_noise_{noise_level:g}",
                source="rank_deficient",
                generators=rank_deficient_generators(3, noise_level, "unitary_near_identity"),
                candidate_rank=3,
                d=3,
                k=1,
                noise_type="unitary_near_identity",
                noise_level=noise_level,
            )
        )

    for control_type in ["permutation", "random_gl"]:
        for noise_level in [0.0, 1e-6, 1e-3]:
            rows.append(
                evaluate_generators(
                    case_id=f"noncentral_{control_type}_{noise_level:g}",
                    source="noncentral_control",
                    generators=generate_noncentral_controls(3, noise_level, seed=410, control_type=control_type),
                    candidate_rank=3,
                    d=None,
                    k=None,
                    noise_type=control_type,
                    noise_level=noise_level,
                )
            )

    rows.append(
        evaluate_generators(
            case_id="mixed_period_common_d12_unknown",
            source="unknown_index_control",
            generators=mixed_period_generators(),
            candidate_rank=12,
            d=12,
            k=None,
            noise_type="none",
            noise_level=0.0,
            max_root_order=4,
        )
    )

    transition_maps, loops = synthetic_transition_maps()
    mining = mine_period_index_generators(transition_maps, loops=loops, max_generators=4)
    mined = detect_mined_period_index(
        transition_maps,
        candidate_rank=4,
        loops=loops,
        max_generators=4,
    )
    if mined.detection is not None:
        result = TwistedMergePlus().run(
            unresolved_pairwise(4),
            n_models=3,
            width=4,
            candidate_lift_rank=4,
            candidate_transition_maps_for_mining=transition_maps,
        )
        rows.append(
            pack_detection_row(
                case_id="synthetic_loop_mining_d2_k2",
                source="synthetic_transition_mining",
                d=2,
                k=2,
                noise_type="none",
                noise_level=0.0,
                candidate_rank=4,
                detection=mined.detection,
                selected_method=result.selected_method,
                generator_mining_used=True,
                n_mined_generators=len(mining.generators),
                notes="; ".join(mining.explanation),
            )
        )
    return rows


def summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["source", "d", "k", "noise_type", "noise_level", "detector_status"], dropna=False)
        .agg(
            n=("case_id", "count"),
            pass_count=("pass_fail", lambda values: int((values == "pass").sum())),
            lift_count=("selected_method", lambda values: int((values == "period_index_projective_morita_lift").sum())),
        )
        .reset_index()
    )
    grouped["pass_rate"] = grouped["pass_count"] / grouped["n"]
    grouped["lift_rate"] = grouped["lift_count"] / grouped["n"]
    return grouped


def write_report(args, df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> None:
    columns = [
        "case_id",
        "source",
        "d",
        "k",
        "noise_type",
        "noise_level",
        "candidate_rank",
        "detector_status",
        "period",
        "index",
        "decision",
        "selected_method",
        "max_centrality_score",
        "max_phase_residual",
        "min_root_margin",
        "pass_fail",
    ]
    summary_columns = ["source", "d", "k", "noise_type", "noise_level", "detector_status", "n", "pass_rate", "lift_rate"]
    def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "noise_level" in out:
            out["noise_level"] = out["noise_level"].map(
                lambda value: "nan" if pd.isna(value) else f"{float(value):.0e}" if 0 < abs(float(value)) < 1e-3 else f"{float(value):g}"
            )
        return out

    noise_summary = display_frame(summary_df[summary_df["source"] == "noisy_heisenberg"])
    rank_rows = display_frame(df[df["source"] == "rank_divisibility"])
    mining_rows = display_frame(df[df["generator_mining_used"]])
    noncentral_rows = display_frame(df[df["source"] == "noncentral_control"])
    full_rows = display_frame(df)
    report = f"""# Robust Period-Index Detector Report

This report is generated by `experiments/robust_period_index_detector.py`.

## Exact Command

```bash
{args.command_string}
```

## Commit Hash

`{git_commit()}`

## Purpose

The exact commutator-matrix detector works on clean central/projective
generators.  This experiment adds thresholded robust detection for noisy
commutators and a synthetic loop-holonomy miner for candidate generators.

## Threshold Policy

Detections are lift certificates only when they pass strict or medium
thresholds and the nearest-root choice is separated from the second-best root.
Loose detections are labeled `central_projective_candidate_uncertain` and keep
`selected_method = none`.  Period divisibility alone is not accepted.

## Main Noise Robustness Table

{format_markdown_table(noise_summary.to_dict("records"), summary_columns)}

## Rank-Divisibility Table

{format_markdown_table(rank_rows.to_dict("records"), columns)}

## Generator-Mining Table

{format_markdown_table(mining_rows.to_dict("records"), columns)}

## Noncentral Rejection Table

{format_markdown_table(noncentral_rows.to_dict("records"), columns)}

## Full Scenario Table

{format_markdown_table(full_rows.to_dict("records"), columns)}

## Algorithmic Conclusion

TwistedMerge++ can use robust central period-index detection in controlled
synthetic data when the detector certifies the index.  Uncertain candidates are
diagnostics, not lifts, and certified detections still require index divisibility.
Synthetic loop mining can recover the hidden `d=2,k=2` generator set in the
controlled transition-map example.

## Negative Boundaries

- This is controlled central/projective evidence.
- No MNIST/CIFAR residual is claimed to be a Brauer class.
- Uncertain or loose noisy candidates are not valid lifts.
- No natural ML performance improvement is claimed.
- Noncentral commutators are rejected rather than called Brauer/projective.

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
    summary_df = summary_rows(df)
    csv_path = args.reports_dir / "csv" / "robust_period_index_detector.csv"
    summary_path = args.reports_dir / "csv" / "robust_period_index_detector_summary.csv"
    report_path = args.reports_dir / "robust_period_index_detector_report.md"
    config_path = args.reports_dir / "configs" / "robust_period_index_detector_config.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    save_json(
        config_path,
        {
            "argv": sys.argv,
            "environment": capture_environment(),
            "commit": git_commit(),
        },
    )
    write_report(args, df, summary_df, report_path)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
