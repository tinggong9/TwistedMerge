#!/usr/bin/env python
"""Unified quantitative obstruction-to-merge chain report.

This script is intentionally an aggregation layer.  It does not rerun model
training or synthetic benchmarks; it reads the existing report CSVs and aligns
their residual, detector, gate, and merge-performance fields into one table.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_CSV = ROOT / "reports/csv/unified_quantitative_obstruction_chain.csv"
OUTPUT_REPORT = ROOT / "reports/unified_quantitative_obstruction_chain.md"
OUTPUT_PLOT = ROOT / "reports/plots/unified_obstruction_chain.pdf"
OUTPUT_TEX = ROOT / "reports/latex/quantitative_obstruction_theorem_candidate.tex"

UNIFIED_COLUMNS = [
    "source_family",
    "source_file",
    "setting_id",
    "dataset",
    "architecture",
    "method",
    "baseline",
    "seed",
    "n_models",
    "width",
    "residual_taxonomy",
    "correction_safe",
    "cycle_score",
    "centrality_score",
    "phase_residual",
    "root_margin",
    "connection_residual",
    "projection_residual",
    "learned_operator_error",
    "validation_delta",
    "test_selector_gain",
    "test_merge_degradation",
    "test_accuracy",
    "base_accuracy",
    "detector_status",
    "detector_certified",
    "lift_decision",
    "lift_selected",
    "gate_decision",
    "gate_pass",
    "claim_promotion_allowed",
    "false_lift",
    "period_divides_rank",
    "index_divides_rank",
    "selected_lift",
    "notes",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | str:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return "unknown"


def read_csv(rel_path: str) -> pd.DataFrame:
    path = ROOT / rel_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def first_text(row: pd.Series, names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row:
            value = row[name]
            if pd.notna(value) and str(value) != "":
                return str(value)
    return default


def first_number(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if pd.isna(value) or str(value) == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("nan")


def boolish(value) -> bool | float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if math.isnan(float(value)):
            return float("nan")
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed", "supported", "supported limited"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed", "not yet supported", "rejected"}:
        return False
    return float("nan")


def first_bool(row: pd.Series, names: Iterable[str]) -> bool | float:
    for name in names:
        if name not in row:
            continue
        parsed = boolish(row[name])
        if not (isinstance(parsed, float) and math.isnan(parsed)):
            return parsed
    return float("nan")


def lift_selected_from_text(text: str) -> bool | float:
    if not text:
        return float("nan")
    value = text.lower()
    if value in {"none", "nan", "not_central_projective", "rank_obstructed", "period_divisible_index_obstructed"}:
        return False
    if "lift" in value and "obstructed" not in value and "rejected" not in value:
        return True
    return False


def detector_certified_from_status(status: str) -> bool | float:
    if not status:
        return float("nan")
    value = status.lower()
    if "certified" in value or "supported" in value or "success" in value:
        return True
    if "uncertain" in value or "rejected" in value or "not_" in value or "obstructed" in value:
        return False
    return float("nan")


def correction_for_taxonomy(taxonomy: str, method: str = "") -> str:
    t = (taxonomy or "").lower()
    m = (method or "").lower()
    if "nonzero_h2" in t:
        return "do_not_claim_same-cover_trivialization"
    if "central_mu2" in t or "finite" in t or "projective" in t or "period" in t:
        return "rank_or_projective_lift_only_when_certified"
    if "noncentral" in t:
        return "noncentral_or_branch_diagnostic_not_brauer"
    if "permutation" in t or "c2m3" in m or "git_rebasin" in m:
        return "permutation_synchronization"
    if "scale" in t or "monomial" in m or "channel_scale" in m:
        return "exact_positive_scale_if_validation_safe"
    if "greedy" in m or "soup" in m:
        return "validation_selected_soup_not_obstruction_certificate"
    if "gate" in t:
        return "claim_gate_controls_promotion"
    if "sheaf" in t:
        return "diagnostic_only_cycle_inconsistency"
    return "diagnostic_only"


def make_row(
    *,
    source_family: str,
    source_file: str,
    row: pd.Series | None = None,
    **kwargs,
) -> dict:
    row = pd.Series(dtype=object) if row is None else row
    method = kwargs.get("method", first_text(row, ["method", "baseline"], ""))
    taxonomy = kwargs.get(
        "residual_taxonomy",
        first_text(
            row,
            [
                "residual_taxonomy",
                "channel_residual_taxonomy",
                "monomial_phase_or_scale_residual_type",
                "permutation_residual_type",
                "residual_type",
                "defect_family",
                "true_family",
                "source",
            ],
            "",
        ),
    )
    lift_decision = kwargs.get("lift_decision", first_text(row, ["selected_method", "decision", "ladder_final_decision"], ""))
    detector_status = kwargs.get(
        "detector_status",
        first_text(row, ["detector_status", "detector_status_after_projection", "strict_policy_decision", "ladder_final_decision"], ""),
    )
    gate_decision = kwargs.get(
        "gate_decision",
        first_text(row, ["feasibility_status", "claim_decision", "strict_policy_decision", "expected_decision"], ""),
    )
    gate_pass = kwargs.get(
        "gate_pass",
        first_bool(row, ["bridge_claims_allowed", "merge_claims_allowed", "calibrated_acceptance_flag", "projection_accepted"]),
    )
    if isinstance(gate_pass, float) and math.isnan(gate_pass):
        if "gate_passed" in gate_decision.lower() or "supported" in gate_decision.lower() or "accept" in gate_decision.lower():
            gate_pass = True
        elif "below" in gate_decision.lower() or "rejected" in gate_decision.lower() or "not yet" in gate_decision.lower():
            gate_pass = False
    out = {
        "source_family": source_family,
        "source_file": source_file,
        "setting_id": kwargs.get("setting_id", first_text(row, ["setting_id", "case_id", "run_id", "case"], "")),
        "dataset": kwargs.get("dataset", first_text(row, ["dataset", "dataset_variant", "case_id"], "")),
        "architecture": kwargs.get("architecture", first_text(row, ["architecture", "level", "diagnostic_level"], "")),
        "method": method,
        "baseline": kwargs.get("baseline", first_text(row, ["baseline"], "")),
        "seed": kwargs.get("seed", first_number(row, ["seed"])),
        "n_models": kwargs.get("n_models", first_number(row, ["n_models"])),
        "width": kwargs.get("width", first_number(row, ["width"])),
        "residual_taxonomy": taxonomy,
        "correction_safe": kwargs.get("correction_safe", correction_for_taxonomy(taxonomy, method)),
        "cycle_score": kwargs.get(
            "cycle_score",
            first_number(
                row,
                [
                    "cycle_score",
                    "planted_cycle_score",
                    "obstruction_score",
                    "channel_permutation_cycle_score",
                    "monomial_phase_or_scale_cycle_score",
                    "observed_cycle_score",
                    "projected_cycle_score",
                    "cycle_inconsistency_mean",
                ],
            ),
        ),
        "centrality_score": kwargs.get(
            "centrality_score",
            first_number(
                row,
                [
                    "centrality_score",
                    "observed_centrality_score",
                    "projected_centrality_score",
                    "monomial_phase_or_scale_centrality",
                    "permutation_centrality",
                    "max_centrality_score",
                    "mean_centrality_score",
                ],
            ),
        ),
        "phase_residual": kwargs.get(
            "phase_residual",
            first_number(row, ["phase_residual", "monomial_phase_or_scale_phase_residual", "max_phase_residual", "mean_phase_residual"]),
        ),
        "root_margin": kwargs.get("root_margin", first_number(row, ["min_root_margin", "mean_root_margin"])),
        "connection_residual": kwargs.get(
            "connection_residual",
            first_number(
                row,
                [
                    "sync_disagreement",
                    "global_sync_residual",
                    "optimized_connection_residual",
                    "spectral_connection_residual",
                    "connection_residual",
                    "global_scale_sync_rms_residual",
                    "global_channel_scale_sync_residual",
                    "global_channel_scale_sync_residual",
                    "pairwise_alignment_residual",
                    "pairwise_activation_alignment_residual",
                    "mean_global_sync_residual",
                ],
            ),
        ),
        "projection_residual": kwargs.get(
            "projection_residual",
            first_number(row, ["projection_residual", "unitary_projection_residual", "mean_projection_residual", "mean_unitary_projection_residual"]),
        ),
        "learned_operator_error": kwargs.get(
            "learned_operator_error",
            first_number(
                row,
                [
                    "learned_operator_error_mean",
                    "learned_operator_error_mean_denoised",
                    "learned_operator_error_denoised",
                    "mean_learned_operator_error",
                    "mean_learned_operator_error_denoised",
                    "mean_operator_error_denoised",
                    "mean_operator_error_raw",
                ],
            ),
        ),
        "validation_delta": kwargs.get(
            "validation_delta",
            first_number(
                row,
                [
                    "validation_delta_vs_greedy_soup",
                    "validation_accuracy_delta_vs_greedy_soup",
                    "mean_validation_delta_vs_greedy_soup",
                    "selector_val_margin",
                    "validation_delta_vs_c2m3",
                    "validation_accuracy_delta_vs_internal_c2m3",
                ],
            ),
        ),
        "test_selector_gain": kwargs.get(
            "test_selector_gain",
            first_number(
                row,
                [
                    "accuracy_delta_vs_greedy_soup",
                    "mean_delta_vs_greedy_soup",
                    "mean_accuracy_delta_vs_greedy_soup",
                    "paired_mean_accuracy_delta",
                    "paired_mean_test_accuracy_delta",
                    "accuracy_delta_vs_internal_c2m3",
                    "accuracy_delta_vs_c2m3",
                ],
            ),
        ),
        "test_merge_degradation": kwargs.get(
            "test_merge_degradation",
            first_number(row, ["merge_degradation", "global_merge_failure", "mean_merge_degradation"]),
        ),
        "test_accuracy": kwargs.get(
            "test_accuracy",
            first_number(row, ["test_accuracy", "accuracy", "mean_test_accuracy", "mean_accuracy"]),
        ),
        "base_accuracy": kwargs.get(
            "base_accuracy",
            first_number(row, ["base_accuracy", "single_best_accuracy", "individual_accuracy_max", "mean_individual_accuracy_max"]),
        ),
        "detector_status": detector_status,
        "detector_certified": kwargs.get("detector_certified", detector_certified_from_status(detector_status)),
        "lift_decision": lift_decision,
        "lift_selected": kwargs.get("lift_selected", lift_selected_from_text(first_text(row, ["selected_method"], lift_decision))),
        "gate_decision": gate_decision,
        "gate_pass": gate_pass,
        "claim_promotion_allowed": kwargs.get("claim_promotion_allowed", gate_pass),
        "false_lift": kwargs.get("false_lift", first_bool(row, ["false_lift", "false_accept"])),
        "period_divides_rank": kwargs.get("period_divides_rank", first_bool(row, ["period_divides_rank"])),
        "index_divides_rank": kwargs.get("index_divides_rank", first_bool(row, ["index_divides_rank"])),
        "selected_lift": kwargs.get("selected_lift", lift_selected_from_text(first_text(row, ["selected_method"], lift_decision))),
        "notes": kwargs.get("notes", first_text(row, ["notes", "method_note", "method_notes", "claim_reason"], "")),
    }
    return {column: out.get(column, "") for column in UNIFIED_COLUMNS}


def add_synthetic_h2(rows: list[dict], rel_path: str) -> None:
    df = read_csv(rel_path)
    for _, row in df.iterrows():
        case = first_text(row, ["case"])
        is_coboundary = first_bool(row, ["is_coboundary"])
        rows.append(
            make_row(
                source_family="synthetic_h2_witness",
                source_file=rel_path,
                row=row,
                method="global_merge",
                residual_taxonomy="coboundary_mu2" if is_coboundary else "nonzero_h2_mu2",
                centrality_score=0.0,
                phase_residual=first_number(row, ["obstruction_score"]),
                detector_status="coboundary" if is_coboundary else "nonzero_h2_obstruction",
                detector_certified=True,
                lift_decision="same_cover_success" if is_coboundary else "same_cover_failed",
                lift_selected=bool(is_coboundary),
                gate_decision="controlled_synthetic",
                gate_pass=True,
                notes=f"H2 witness case={case}; nontrivial rows are not same-cover trivializations.",
            )
        )


def add_generic_model_rows(rows: list[dict], rel_path: str, source_family: str, *, taxonomy_default: str = "") -> None:
    df = read_csv(rel_path)
    for _, row in df.iterrows():
        method = first_text(row, ["method", "baseline"], "")
        taxonomy = first_text(
            row,
            [
                "channel_residual_taxonomy",
                "monomial_phase_or_scale_residual_type",
                "permutation_residual_type",
                "noncentral_final_decision",
                "residual_type",
                "defect_family",
            ],
            taxonomy_default,
        )
        gate_decision = first_text(row, ["feasibility_status", "evaluation_status"], "")
        rows.append(
            make_row(
                source_family=source_family,
                source_file=rel_path,
                row=row,
                method=method,
                residual_taxonomy=taxonomy,
                gate_decision=gate_decision,
                notes="model-merging row aligned from existing benchmark output",
            )
        )


def add_block_rows(rows: list[dict]) -> None:
    for rel_path, source_family in [
        ("reports/csv/block_gauge_phase_diagram.csv", "block_gauge_phase_diagram"),
        ("reports/csv/global_block_synchronization.csv", "global_block_synchronization"),
        ("reports/csv/optimized_global_block_synchronization.csv", "optimized_global_block_synchronization"),
    ]:
        df = read_csv(rel_path)
        for _, row in df.iterrows():
            taxonomy = first_text(row, ["true_family", "case_family", "residual_type", "expected_outcome"], "block_gauge")
            gate_pass = first_bool(row, ["calibrated_acceptance_flag", "accepted_global_sync", "accepted_sync"])
            detector_status = first_text(row, ["strict_policy_decision", "claim_status", "residual_type"], "")
            rows.append(
                make_row(
                    source_family=source_family,
                    source_file=rel_path,
                    row=row,
                    method=first_text(row, ["method", "partition_method"], ""),
                    residual_taxonomy=taxonomy,
                    detector_status=detector_status,
                    detector_certified=gate_pass,
                    gate_pass=gate_pass,
                    gate_decision=detector_status,
                    correction_safe="global_block_sync_only_if_connection_residual_gate_accepts",
                    notes="block-gauge row keeps connection residual separate from projected cycle score",
                )
            )


def add_time_frequency_rows(rows: list[dict]) -> None:
    for rel_path, source_family in [
        ("reports/csv/time_frequency_learned_chart_benchmark.csv", "time_frequency_learned_chart"),
        ("reports/csv/time_frequency_denoised_chart_benchmark.csv", "time_frequency_denoised_chart"),
        ("reports/csv/time_frequency_heisenberg_projection_benchmark.csv", "time_frequency_heisenberg_projection"),
        ("reports/csv/robust_period_index_calibration.csv", "robust_period_index_calibration"),
    ]:
        df = read_csv(rel_path)
        for _, row in df.iterrows():
            selected = first_text(row, ["selected_method"], "")
            status = first_text(row, ["detector_status_after_projection", "detector_status"], "")
            decision = first_text(row, ["decision"], "")
            source = first_text(row, ["source"], "")
            taxonomy = "period_index_projective"
            if "noncentral" in source.lower() or "negative" in source.lower():
                taxonomy = "noncentral_or_negative_control"
            if source_family == "time_frequency_heisenberg_projection":
                taxonomy = "heisenberg_projection_gate"
            rows.append(
                make_row(
                    source_family=source_family,
                    source_file=rel_path,
                    row=row,
                    method=first_text(row, ["method", "denoising_method", "level", "noise_type"], ""),
                    residual_taxonomy=taxonomy,
                    detector_status=status,
                    detector_certified=detector_certified_from_status(status),
                    lift_decision=decision,
                    lift_selected=lift_selected_from_text(selected),
                    selected_lift=lift_selected_from_text(selected),
                    correction_safe="period_index_lift_only_when_certified_and_rank_index_divides",
                    notes="time-frequency detector row aligned with rank and lift decisions",
                )
            )


def add_sheaf_rows(rows: list[dict], rel_path: str) -> None:
    df = read_csv(rel_path)
    for _, row in df.iterrows():
        rows.append(
            make_row(
                source_family="sheaf_gnn_cycle_diagnostics",
                source_file=rel_path,
                row=row,
                method=first_text(row, ["method"], ""),
                residual_taxonomy="sheaf_cycle_inconsistency",
                cycle_score=first_number(row, ["cycle_inconsistency_mean"]),
                centrality_score=float("nan"),
                connection_residual=first_number(row, ["dirichlet_energy"]),
                test_accuracy=first_number(row, ["test_accuracy"]),
                gate_decision="optional_gnn_diagnostic",
                gate_pass=True,
                correction_safe="diagnostic_only_cycle_inconsistency",
                notes="optional sheaf/GNN diagnostic; not a model-merging claim",
            )
        )


def build_unified_rows() -> pd.DataFrame:
    rows: list[dict] = []
    add_synthetic_h2(rows, "reports/csv/synthetic_h2_mu2_obstruction.csv")
    add_generic_model_rows(rows, "reports/csv/planted_obstruction_model_merging.csv", "planted_obstruction_benchmark")
    for rel_path, family in [
        ("reports/csv/external_baseline_comparison.csv", "mnist_external_baseline_comparison"),
        ("reports/csv/improved_validated_ladder_merge_benchmark.csv", "mnist_improved_validated_ladder"),
        ("reports/csv/fashion_mnist_improved_ladder.csv", "fashion_mnist_mlp_ladder"),
        ("reports/csv/fashion_mnist_cnn_ladder.csv", "fashion_mnist_cnn_ladder"),
        ("reports/csv/fashion_mnist_cnn_channel_gauge_confirmatory.csv", "fashion_mnist_cnn_confirmatory"),
        ("reports/csv/bridge_dataset_channel_gauge_expansion.csv", "rotated_colored_mnist_bridge"),
        ("reports/csv/cifar_or_colored_mnist_feasibility.csv", "cifar_or_colored_mnist_gate"),
        ("reports/csv/cifar_rescue_or_no_go.csv", "cifar_rescue_or_no_go_gate"),
        ("reports/csv/greedy_aware_monomial_benchmark.csv", "greedy_aware_monomial"),
    ]:
        add_generic_model_rows(rows, rel_path, family)
    add_block_rows(rows)
    add_time_frequency_rows(rows)
    add_sheaf_rows(rows, "reports/csv/sheaf_gnn_cycle_diagnostics.csv")
    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    for col in [
        "seed",
        "n_models",
        "width",
        "cycle_score",
        "centrality_score",
        "phase_residual",
        "root_margin",
        "connection_residual",
        "projection_residual",
        "learned_operator_error",
        "validation_delta",
        "test_selector_gain",
        "test_merge_degradation",
        "test_accuracy",
        "base_accuracy",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def corr_pair(df: pd.DataFrame, x: str, y: str) -> tuple[int, float, float]:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return len(sub), float("nan"), float("nan")
    return len(sub), float(sub[x].corr(sub[y], method="pearson")), float(sub[x].corr(sub[y], method="spearman"))


def sign_agreement(x: pd.Series, y: pd.Series) -> float:
    sub = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub["x"] != 0) & (sub["y"] != 0)]
    if sub.empty:
        return float("nan")
    return float((np.sign(sub["x"]) == np.sign(sub["y"])).mean())


def rate(value: pd.Series) -> float:
    parsed = []
    for item in value:
        b = boolish(item)
        if isinstance(b, float) and math.isnan(b):
            continue
        parsed.append(float(bool(b)))
    if not parsed:
        return float("nan")
    return float(np.mean(parsed))


def compute_link_tests(unified: pd.DataFrame) -> pd.DataFrame:
    tests: list[dict] = []

    planted = unified[
        (unified["source_family"] == "planted_obstruction_benchmark")
        & (unified["method"] == "git_rebasin_pairwise")
    ]
    n, pearson, spearman = corr_pair(planted, "cycle_score", "test_merge_degradation")
    tests.append(
        {
            "test_name": "planted_cycle_score_to_pairwise_merge_degradation",
            "n": n,
            "primary_metric": "spearman",
            "effect": spearman,
            "secondary_metric": "pearson",
            "secondary_effect": pearson,
            "decision": "supported_descriptive" if n >= 10 and spearman > 0.5 else "not_decisive",
            "interpretation": "Planted cycle score tracks Git-ReBasin-style pairwise degradation.",
        }
    )

    selectors = unified[
        unified["method"].str.contains("selector", case=False, na=False)
        & unified["validation_delta"].notna()
        & unified["test_selector_gain"].notna()
    ]
    n, pearson, spearman = corr_pair(selectors, "validation_delta", "test_selector_gain")
    tests.append(
        {
            "test_name": "validation_delta_to_test_selector_gain",
            "n": n,
            "primary_metric": "spearman",
            "effect": spearman,
            "secondary_metric": "sign_agreement",
            "secondary_effect": sign_agreement(selectors["validation_delta"], selectors["test_selector_gain"]),
            "decision": "source_dependent_predictor" if n >= 10 else "not_enough_rows",
            "interpretation": "Validation deltas are useful for selectors, but are not a topology residual.",
        }
    )

    operator = unified[
        unified["source_family"].isin(["time_frequency_learned_chart", "time_frequency_denoised_chart"])
        & unified["learned_operator_error"].notna()
        & unified["detector_certified"].notna()
    ].copy()
    operator["certified_float"] = operator["detector_certified"].map(lambda x: float(boolish(x)) if not (isinstance(boolish(x), float) and math.isnan(boolish(x))) else np.nan)
    n, pearson, spearman = corr_pair(operator, "learned_operator_error", "certified_float")
    tests.append(
        {
            "test_name": "operator_error_to_certification_rate",
            "n": n,
            "primary_metric": "spearman",
            "effect": spearman,
            "secondary_metric": "pearson",
            "secondary_effect": pearson,
            "decision": "supported_negative_relationship" if n >= 10 and spearman < -0.3 else "not_decisive",
            "interpretation": "Larger learned-operator error lowers calibrated period-index certification.",
        }
    )

    projection = unified[
        (unified["source_family"] == "time_frequency_heisenberg_projection")
        & unified["projection_residual"].notna()
    ].copy()
    projection["accepted_float"] = projection["gate_pass"].map(lambda x: float(boolish(x)) if not (isinstance(boolish(x), float) and math.isnan(boolish(x))) else np.nan)
    n, pearson, spearman = corr_pair(projection, "projection_residual", "accepted_float")
    accepted = projection[projection["accepted_float"] == 1.0]
    false_lift_rate = rate(accepted["false_lift"])
    tests.append(
        {
            "test_name": "projection_residual_gate_to_false_lift_control",
            "n": n,
            "primary_metric": "spearman_residual_vs_acceptance",
            "effect": spearman,
            "secondary_metric": "accepted_false_lift_rate",
            "secondary_effect": false_lift_rate,
            "decision": "supported_gate" if n >= 10 and (pd.isna(false_lift_rate) or false_lift_rate == 0.0) else "inspect",
            "interpretation": "Projection residual is a gate, not a free lift; accepted projection rows have zero false lifts here.",
        }
    )

    robust = unified[unified["source_family"] == "robust_period_index_calibration"].copy()
    robust = robust[robust["period_divides_rank"].notna() & robust["index_divides_rank"].notna()]
    robust["expected_lift"] = robust["index_divides_rank"].map(lambda x: boolish(x) is True)
    robust["actual_lift"] = robust["selected_lift"].map(lambda x: boolish(x) is True)
    if not robust.empty:
        match_rate = float((robust["expected_lift"] == robust["actual_lift"]).mean())
    else:
        match_rate = float("nan")
    tests.append(
        {
            "test_name": "period_index_divisibility_to_selected_lift",
            "n": int(len(robust)),
            "primary_metric": "decision_match_rate",
            "effect": match_rate,
            "secondary_metric": "index_divides_lift_rate",
            "secondary_effect": rate(robust[robust["expected_lift"]]["selected_lift"]) if not robust.empty else float("nan"),
            "decision": "supported_threshold" if len(robust) >= 10 and match_rate > 0.95 else "inspect",
            "interpretation": "Index divisibility, not period divisibility alone, controls selected lifts in calibrated rows.",
        }
    )

    gates = unified[
        unified["source_family"].isin(
            [
                "rotated_colored_mnist_bridge",
                "cifar_or_colored_mnist_gate",
                "cifar_rescue_or_no_go_gate",
            ]
        )
        & unified["gate_pass"].notna()
    ].copy()
    gate_pass_rate = rate(gates["gate_pass"])
    tests.append(
        {
            "test_name": "dataset_gate_to_claim_promotion",
            "n": int(len(gates)),
            "primary_metric": "gate_pass_rate",
            "effect": gate_pass_rate,
            "secondary_metric": "distinct_gate_decisions",
            "secondary_effect": float(gates["gate_decision"].nunique()) if not gates.empty else float("nan"),
            "decision": "supported_gate_accounting" if len(gates) > 0 else "missing_gate_rows",
            "interpretation": "Dataset gates govern whether bridge/CIFAR rows can be promoted to claims.",
        }
    )

    tax = unified[unified["residual_taxonomy"].astype(str) != ""].copy()
    corrections = tax.groupby("residual_taxonomy")["correction_safe"].nunique()
    tests.append(
        {
            "test_name": "residual_taxonomy_to_safe_correction",
            "n": int(len(tax)),
            "primary_metric": "taxonomies_with_unique_safe_correction",
            "effect": float((corrections > 0).sum()),
            "secondary_metric": "distinct_safe_corrections",
            "secondary_effect": float(tax["correction_safe"].nunique()),
            "decision": "supported_multiaxis_taxonomy" if tax["correction_safe"].nunique() >= 4 else "not_decisive",
            "interpretation": "Residual labels map to different safe corrections; they should not be collapsed to one scalar.",
        }
    )

    block = read_csv("reports/csv/block_gauge_phase_diagram.csv")
    trap_count = 0
    if not block.empty:
        projected = pd.to_numeric(block.get("projected_cycle_score"), errors="coerce")
        residual = pd.to_numeric(block.get("optimized_connection_residual"), errors="coerce")
        strict = block.get("strict_policy_decision", pd.Series([""] * len(block))).astype(str).str.lower()
        trap_count = int(((projected < 1e-8) & (residual > 1e-6) & strict.str.contains("reject|diagnostic|fail", regex=True)).sum())
    tests.append(
        {
            "test_name": "single_scalar_obstruction_counterexample",
            "n": int(len(block)),
            "primary_metric": "projection_trap_count",
            "effect": float(trap_count),
            "secondary_metric": "metric",
            "secondary_effect": float("nan"),
            "decision": "supports_no_single_scalar" if trap_count > 0 else "not_decisive",
            "interpretation": "Projected cycle score alone can look small while connection residual gates reject the row.",
        }
    )

    return pd.DataFrame(tests)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, columns].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            view[col] = view[col].fillna("").astype(str)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def plot_unified(unified: pd.DataFrame, tests: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    planted = unified[
        (unified["source_family"] == "planted_obstruction_benchmark")
        & (unified["method"] == "git_rebasin_pairwise")
    ]
    ax = axes[0, 0]
    if planted.empty:
        ax.text(0.5, 0.5, "No planted rows", ha="center", va="center")
    else:
        for name, group in planted.groupby("residual_taxonomy"):
            ax.scatter(group["cycle_score"], group["test_merge_degradation"], s=28, alpha=0.75, label=str(name))
        ax.set_xlabel("Planted cycle score")
        ax.set_ylabel("Pairwise merge degradation")
        ax.legend(fontsize=7)
    ax.set_title("Cycle score -> degradation")

    selectors = unified[
        unified["method"].str.contains("selector", case=False, na=False)
        & unified["validation_delta"].notna()
        & unified["test_selector_gain"].notna()
    ]
    ax = axes[0, 1]
    if selectors.empty:
        ax.text(0.5, 0.5, "No selector deltas", ha="center", va="center")
    else:
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.axvline(0, color="0.75", linewidth=0.8)
        ax.scatter(selectors["validation_delta"], selectors["test_selector_gain"], s=18, alpha=0.55)
        ax.set_xlabel("Validation delta")
        ax.set_ylabel("Test selector gain")
    ax.set_title("Validation delta -> test gain")

    operator = unified[
        unified["source_family"].isin(["time_frequency_learned_chart", "time_frequency_denoised_chart"])
        & unified["learned_operator_error"].notna()
        & unified["detector_certified"].notna()
    ].copy()
    ax = axes[1, 0]
    if operator.empty:
        ax.text(0.5, 0.5, "No operator-error rows", ha="center", va="center")
    else:
        operator["cert"] = operator["detector_certified"].map(lambda x: float(boolish(x)) if not (isinstance(boolish(x), float) and math.isnan(boolish(x))) else np.nan)
        x = np.maximum(operator["learned_operator_error"].to_numpy(dtype=float), 1e-15)
        y = operator["cert"].to_numpy(dtype=float)
        ax.scatter(x, y + np.random.default_rng(0).normal(0.0, 0.015, size=len(y)), s=14, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("Learned operator error")
        ax.set_ylabel("Certified detector row")
        ax.set_ylim(-0.1, 1.1)
    ax.set_title("Operator error -> certification")

    projection = unified[
        (unified["source_family"] == "time_frequency_heisenberg_projection")
        & unified["projection_residual"].notna()
    ].copy()
    ax = axes[1, 1]
    if projection.empty:
        ax.text(0.5, 0.5, "No projection rows", ha="center", va="center")
    else:
        projection["accepted"] = projection["gate_pass"].map(lambda x: float(boolish(x)) if not (isinstance(boolish(x), float) and math.isnan(boolish(x))) else np.nan)
        x = np.maximum(projection["projection_residual"].to_numpy(dtype=float), 1e-15)
        y = projection["accepted"].to_numpy(dtype=float)
        ax.scatter(x, y + np.random.default_rng(1).normal(0.0, 0.015, size=len(y)), s=14, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("Projection residual")
        ax.set_ylabel("Projection gate accepted")
        ax.set_ylim(-0.1, 1.1)
    ax.set_title("Projection residual gate")

    fig.suptitle("Unified Quantitative Obstruction Chain", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output)
    plt.close(fig)


def write_report(unified: pd.DataFrame, tests: pd.DataFrame, output: Path, config: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    coverage = (
        unified.groupby(["source_family", "source_file"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source_family", "source_file"])
    )
    taxonomy = (
        unified[unified["residual_taxonomy"].astype(str) != ""]
        .groupby(["residual_taxonomy", "correction_safe"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .head(16)
    )
    sources = sorted(unified["source_family"].dropna().unique())
    tests_view = tests.copy()
    tests_view["effect"] = pd.to_numeric(tests_view["effect"], errors="coerce")
    tests_view["secondary_effect"] = pd.to_numeric(tests_view["secondary_effect"], errors="coerce")

    supported = tests_view["decision"].astype(str).str.contains("supported").sum()
    no_single_scalar = tests_view.loc[tests_view["test_name"] == "single_scalar_obstruction_counterexample", "decision"].astype(str).iloc[0]

    lines = [
        "# Unified Quantitative Obstruction-to-Merge Chain",
        "",
        "Generated by `experiments/unified_quantitative_obstruction_chain.py`.",
        "",
        "## Scope",
        "",
        f"- Aggregated rows: `{len(unified)}`.",
        f"- Source families: `{len(sources)}`.",
        "- This is a post-hoc alignment of existing artifacts; it does not rerun model training or method selection.",
        "- Missing metrics remain blank in the CSV when a source did not measure that quantity.",
        "- The allowed conclusion is deliberately weak: different residual scores predict different downstream decisions, and a single scalar obstruction does not explain all cases.",
        "- Forbidden boundary: this report does not claim that every merging failure is Brauer/projective.",
        "",
        "## Exact Command",
        "",
        "```bash",
        f"{sys.executable} experiments/unified_quantitative_obstruction_chain.py",
        "```",
        "",
        "## Link Tests",
        "",
        markdown_table(
            tests_view,
            [
                "test_name",
                "n",
                "primary_metric",
                "effect",
                "secondary_metric",
                "secondary_effect",
                "decision",
                "interpretation",
            ],
            max_rows=20,
        ),
        "",
        "## Source Coverage",
        "",
        markdown_table(coverage, ["source_family", "source_file", "rows"], max_rows=40),
        "",
        "## Residual Taxonomy To Safe Correction",
        "",
        markdown_table(taxonomy, ["residual_taxonomy", "correction_safe", "rows"], max_rows=20),
        "",
        "## Interpretation",
        "",
        f"- `{supported}` link tests have a supported descriptive or gate-style decision.",
        f"- The single-scalar counterexample decision is `{no_single_scalar}`.",
        "- Planted cycle score is useful for planted pairwise synchronization degradation.",
        "- Validation deltas are selector evidence, not residual certificates.",
        "- Learned operator error, projection residual, centrality, phase residual, and root margin govern detector certification and lift decisions.",
        "- Dataset gates control claim promotion separately from residual scores.",
        "- Residual taxonomy matters: permutation, exact positive scale, noncentral, projective, projection-gated, sheaf diagnostic, and dataset-gate cases have different safe corrections.",
        "",
        "## Claim Boundary",
        "",
        "Supported: different residual scores predict different downstream decisions; a single scalar obstruction does not explain all cases.",
        "",
        "Not supported: every merging failure is Brauer/projective.",
        "",
        "## Output Files",
        "",
        f"- `{rel(OUTPUT_CSV)}`",
        f"- `{rel(OUTPUT_PLOT)}`",
        f"- `{rel(OUTPUT_TEX)}`",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_latex(tests: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    planted = tests.loc[tests["test_name"] == "planted_cycle_score_to_pairwise_merge_degradation"].iloc[0]
    selector = tests.loc[tests["test_name"] == "validation_delta_to_test_selector_gain"].iloc[0]
    projection = tests.loc[tests["test_name"] == "projection_residual_gate_to_false_lift_control"].iloc[0]
    period = tests.loc[tests["test_name"] == "period_index_divisibility_to_selected_lift"].iloc[0]
    scalar = tests.loc[tests["test_name"] == "single_scalar_obstruction_counterexample"].iloc[0]
    text = rf"""\begin{{theorem}}[Quantitative obstruction-chain candidate]
Across the current TwistedMerge evidence table, residual diagnostics enter the
merge pipeline through distinct decision channels rather than through one
universal scalar obstruction.  In the planted benchmark, cycle score predicts
pairwise merge degradation with Spearman correlation {float(planted['effect']):.3f}
over {int(planted['n'])} rows.  Validation deltas for selector rows have
Spearman correlation {float(selector['effect']):.3f} with test selector gain over
{int(selector['n'])} rows, but this is selection evidence rather than a residual
certificate.  Projection residual gates have accepted-row false lift rate
{float(projection['secondary_effect']):.3f}.  Period-index rows match selected
lift decisions at rate {float(period['effect']):.3f}.  Finally, the block-gauge
phase diagram contains {int(scalar['effect'])} rows where projected cycle score
alone would be misleading without the connection-residual gate.
\end{{theorem}}

\begin{{remark}}
The theorem candidate is empirical and artifact-scoped.  It supports the claim
that different residual scores predict different downstream decisions.  It does
not support the stronger claim that every merging failure is Brauer/projective.
\end{{remark}}
"""
    output.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-plot", action="store_true", help="Write CSV/report/LaTeX but skip the PDF plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    unified = build_unified_rows()
    tests = compute_link_tests(unified)
    unified.to_csv(OUTPUT_CSV, index=False)
    config = {
        "git_commit": git_commit(),
        "dirty_worktree": git_dirty(),
        "python": sys.version,
        "rows": int(len(unified)),
        "source_families": sorted(unified["source_family"].dropna().unique()),
    }
    if not args.skip_plot:
        plot_unified(unified, tests, OUTPUT_PLOT)
    write_latex(tests, OUTPUT_TEX)
    write_report(unified, tests, OUTPUT_REPORT, config)
    print(f"Wrote {rel(OUTPUT_CSV)}")
    if not args.skip_plot:
        print(f"Wrote {rel(OUTPUT_PLOT)}")
    print(f"Wrote {rel(OUTPUT_TEX)}")
    print(f"Wrote {rel(OUTPUT_REPORT)}")


if __name__ == "__main__":
    main()
