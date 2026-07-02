"""Diagnostic Q-branch lift metadata for small quotient holonomy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.nonabelian_invariant_pooling import (
    naive_representation_residual,
    pooling_residual,
    regular_action_permutation,
)
from src.small_quotient_holonomy import QuotientFit


Q_LIFT_METHODS = (
    "small_quotient_regular_branch_lift",
    "small_quotient_orbit_branch_lift",
    "small_quotient_invariant_pooling_lift",
    "random_same_Q_branch_control",
    "wrong_quotient_lift_control",
    "shuffled_quotient_lift_control",
)


def quotient_pooling_residual_summary(fit: QuotientFit, feature_dim: int = 1) -> dict:
    naive = []
    pooled = []
    for element in fit.quotient_holonomies:
        branch_perm = regular_action_permutation(fit.Q_group, element, side="left")
        naive.append(naive_representation_residual(branch_perm, feature_dim=1))
        pooled.append(pooling_residual(branch_perm, feature_dim=int(feature_dim)))
    return {
        "naive_quotient_residual": float(np.mean(naive)) if naive else np.nan,
        "naive_quotient_residual_max": float(np.max(naive)) if naive else np.nan,
        "invariant_pooling_residual": float(np.mean(pooled)) if pooled else np.nan,
        "invariant_pooling_residual_max": float(np.max(pooled)) if pooled else np.nan,
        "feature_dim": int(feature_dim),
    }


def quotient_capacity_metadata(method_name: str, Q_order: int, lift_implemented: bool = False) -> dict:
    branch_count = int(max(1, Q_order))
    is_branch = str(method_name) in Q_LIFT_METHODS
    return {
        "lift_implemented": bool(lift_implemented),
        "branch_count": branch_count if is_branch else 1,
        "parameter_multiplier": float(branch_count) if is_branch else 1.0,
        "inference_multiplier": float(branch_count) if is_branch else 1.0,
        "is_single_model": False if is_branch else True,
        "is_extra_capacity": bool(is_branch and branch_count > 1),
        "capacity_matched_to_same_branch_control": method_name
        in {
            "small_quotient_regular_branch_lift",
            "small_quotient_orbit_branch_lift",
            "small_quotient_invariant_pooling_lift",
            "random_same_Q_branch_control",
            "wrong_quotient_lift_control",
            "shuffled_quotient_lift_control",
        },
        "capacity_matched_to_weight_average": not is_branch,
        "capacity_matched_to_rank_lift": bool(is_branch),
        "lift_level": "level_1_diagnostic_only" if not lift_implemented else "level_2_branch_lift",
    }


def build_pooling_rows(candidate_rows: pd.DataFrame, fits_by_key: dict[tuple[str, str], QuotientFit], thresholds) -> pd.DataFrame:
    rows = []
    if candidate_rows.empty:
        return pd.DataFrame()
    for _, candidate in candidate_rows.iterrows():
        key = (str(candidate.get("run_id")), str(candidate.get("Q_name")))
        fit = fits_by_key.get(key)
        if fit is None:
            continue
        summary = quotient_pooling_residual_summary(fit)
        for threshold in thresholds:
            pass_gate = bool(
                pd.to_numeric(pd.Series([summary["invariant_pooling_residual"]]), errors="coerce").iloc[0]
                <= float(threshold)
            )
            rows.append(
                {
                    **candidate.to_dict(),
                    **summary,
                    "pooling_threshold": float(threshold),
                    "pooling_gate_passed": pass_gate,
                }
            )
    return pd.DataFrame(rows)


def build_small_quotient_lift_candidates(pooling_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pooling_rows.empty:
        return pd.DataFrame()
    base_columns = [
        "residual_source",
        "run_id",
        "dataset",
        "n_models",
        "width",
        "seed",
        "Q_name",
        "Q_order",
        "quotient_fit_status",
        "relation_threshold",
        "nontrivial_threshold",
        "quotient_certified",
        "relation_violation_rate",
        "pooling_threshold",
        "naive_quotient_residual",
        "invariant_pooling_residual",
        "pooling_gate_passed",
    ]
    for _, row in pooling_rows.iterrows():
        certified = bool(row.get("quotient_certified", False))
        pooling_gate = bool(row.get("pooling_gate_passed", False))
        core = {column: row.get(column, np.nan) for column in base_columns}
        for method in Q_LIFT_METHODS:
            implemented = False
            diagnostic_gate = certified and pooling_gate
            rows.append(
                {
                    **core,
                    "candidate_method": method,
                    "diagnostic_gate_passed": diagnostic_gate,
                    "lift_implemented": implemented,
                    "validation_accuracy": np.nan,
                    "validation_loss": np.nan,
                    "val_accuracy": np.nan,
                    "val_loss": np.nan,
                    "test_accuracy": np.nan,
                    "test_loss": np.nan,
                    "uses_validation_data": False,
                    "uses_test_data_for_selection": False,
                    "reason": (
                        "diagnostic_only"
                        if diagnostic_gate
                        else "gate_failed"
                    ),
                    **quotient_capacity_metadata(method, row.get("Q_order", 1), implemented),
                }
            )
    return pd.DataFrame(rows)
