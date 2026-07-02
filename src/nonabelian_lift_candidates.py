"""Lift-candidate metadata and validation-safe selection for nonabelian holonomy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation_gated_period_index_lift import SelectorPolicy, best_overall_fallback, torsion_safe_selector


REPRESENTATION_TO_LIFT = {
    "orbit_representation": "orbit_split_lift",
    "regular_representation": "regular_representation_lift",
    "quotient_representation": "quotient_representation_lift",
    "low_dimensional_permutation_subrepresentation": "minimal_detected_representation_lift",
    "existing_permutation_representation": "branch_holonomy_lift",
    "random_same_dimension_representation_control": "random_same_rank_lift_control",
}


def capacity_metadata(method_name: str, representation_dimension, base_width: int | None) -> dict:
    dim = int(representation_dimension) if pd.notna(representation_dimension) else 0
    width = int(base_width) if base_width else 0
    multiplier = float(dim / width) if width > 0 and dim > 0 else np.nan
    is_branch = str(method_name) in {"branch_holonomy_lift", "random_same_rank_lift_control"}
    diagnostic_only = str(method_name) in {
        "orbit_split_lift",
        "regular_representation_lift",
        "quotient_representation_lift",
        "minimal_detected_representation_lift",
        "invariant_projection_lift",
    }
    return {
        "parameter_multiplier": multiplier,
        "inference_multiplier": float(max(1, dim)) if is_branch and dim else (1.0 if diagnostic_only else multiplier),
        "branch_count": int(max(1, dim)) if is_branch and dim else 1,
        "capacity_matched_to_weight_average": False,
        "capacity_matched_to_same_rank_control": str(method_name) == "random_same_rank_lift_control",
        "is_single_model": False if is_branch else bool(diagnostic_only),
        "is_extra_capacity": bool(dim and width and dim > width),
        "lift_level": "level_1_diagnostic_only",
    }


def build_lift_candidate_rows(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scores.iterrows():
        method = REPRESENTATION_TO_LIFT.get(str(row.get("representation_name")), "invariant_projection_lift")
        split_pass = bool(row.get("split_success_flag", False))
        implemented = False
        rows.append(
            {
                **row.to_dict(),
                "candidate_method": method,
                "lift_implemented": implemented,
                "selected_method": "none",
                "diagnostic_gate_passed": split_pass,
                "capacity_matched": False,
                "reason": (
                    "diagnostic_split_but_no_model_lift_implementation"
                    if split_pass
                    else "representation_does_not_pass_split_gate"
                ),
                "uses_validation_data": False,
                "uses_test_data_for_selection": False,
                **capacity_metadata(method, row.get("representation_dimension"), row.get("width")),
            }
        )
    return pd.DataFrame(rows)


def nonabelian_holonomy_safe_selector(
    fallback_rows: pd.DataFrame,
    lift_rows: pd.DataFrame,
    policy: SelectorPolicy,
) -> pd.DataFrame:
    fallback = best_overall_fallback(fallback_rows)
    if fallback.empty:
        return fallback
    eligible = pd.DataFrame()
    if not lift_rows.empty:
        eligible = lift_rows[
            (lift_rows.get("diagnostic_gate_passed", False) == True)  # noqa: E712
            & (lift_rows.get("lift_implemented", False) == True)  # noqa: E712
        ].copy()
    if eligible.empty:
        selected = torsion_safe_selector(fallback_rows, pd.DataFrame(), policy)
        selected["selector_method"] = "nonabelian_holonomy_safe_selector"
        selected["selected_nonabelian_lift"] = False
        selected["selector_no_test_leakage"] = True
        return selected

    selections = []
    for _, base in fallback.iterrows():
        run_id = str(base["run_id"])
        pool = eligible[eligible["run_id"].astype(str).eq(run_id)].copy()
        selected = base.to_dict()
        selected.update(
            {
                "selector_method": "nonabelian_holonomy_safe_selector",
                "selected_candidate_method": base.get("candidate_method", "best_fallback"),
                "selected_nonabelian_lift": False,
                "selected_lift": False,
                "selector_no_test_leakage": True,
                "selector_epsilon": float(policy.epsilon),
                "selector_loss_slack": float(policy.loss_slack),
                "best_fallback_val_accuracy": float(base.get("val_accuracy", np.nan)),
                "best_fallback_val_loss": float(base.get("val_loss", np.nan)),
                "best_fallback_test_accuracy": float(base.get("test_accuracy", np.nan)),
                "best_fallback_test_loss": float(base.get("test_loss", np.nan)),
            }
        )
        if not pool.empty:
            pool["val_accuracy"] = pd.to_numeric(pool["val_accuracy"], errors="coerce")
            pool["val_loss"] = pd.to_numeric(pool["val_loss"], errors="coerce")
            passing = pool[
                (pool["val_accuracy"] >= float(base["val_accuracy"]) + float(policy.epsilon))
                & (pool["val_loss"] <= float(base["val_loss"]) + float(policy.loss_slack))
            ].sort_values(["val_accuracy", "val_loss", "candidate_method"], ascending=[False, True, True])
            if not passing.empty:
                lift = passing.iloc[0].to_dict()
                selected.update(lift)
                selected.update(
                    {
                        "selector_method": "nonabelian_holonomy_safe_selector",
                        "selected_candidate_method": lift.get("candidate_method", "nonabelian_lift"),
                        "selected_nonabelian_lift": True,
                        "selected_lift": True,
                        "selector_no_test_leakage": True,
                    }
                )
        selections.append(selected)
    return pd.DataFrame(selections)
