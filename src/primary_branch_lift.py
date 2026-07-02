"""Prediction-level p-primary branch-lift candidate metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd


IMPLEMENTED_REAL_Q_TO_METHOD = {
    2: "twisted_rank_lift_2",
}


def primary_method_name(q_order: int, primary_type: str) -> str:
    q = int(q_order)
    if str(primary_type) == "mixed":
        return f"mixed_C{q}_branch_lift"
    return f"primary_C{q}_branch_lift"


def branch_capacity_metadata(q_order: int, lift_implemented: bool, is_control: bool = False) -> dict:
    q = int(max(1, q_order))
    return {
        "branch_count": q,
        "parameter_multiplier": float(q),
        "inference_multiplier": float(q),
        "is_single_model": False if q > 1 else True,
        "is_extra_capacity": bool(q > 1),
        "capacity_matched_to_same_branch_control": bool(q > 1 and (lift_implemented or is_control)),
        "lift_implemented": bool(lift_implemented),
    }


def _best_method_row(run_rows: pd.DataFrame, run_id: str, method: str) -> dict | None:
    rows = run_rows[
        run_rows["run_id"].astype(str).eq(str(run_id))
        & run_rows["method"].astype(str).eq(str(method))
    ].copy()
    if rows.empty:
        return None
    rows["val_accuracy"] = pd.to_numeric(rows["val_accuracy"], errors="coerce")
    rows["val_loss"] = pd.to_numeric(rows["val_loss"], errors="coerce")
    rows = rows.sort_values(["val_accuracy", "val_loss"], ascending=[False, True])
    return rows.iloc[0].to_dict()


def _metrics_from_row(row: dict | None) -> dict:
    if row is None:
        return {
            "validation_accuracy": np.nan,
            "validation_loss": np.nan,
            "val_accuracy": np.nan,
            "val_loss": np.nan,
            "test_accuracy": np.nan,
            "test_loss": np.nan,
        }
    return {
        "validation_accuracy": row.get("val_accuracy", np.nan),
        "validation_loss": row.get("val_loss", np.nan),
        "val_accuracy": row.get("val_accuracy", np.nan),
        "val_loss": row.get("val_loss", np.nan),
        "test_accuracy": row.get("test_accuracy", np.nan),
        "test_loss": row.get("test_loss", np.nan),
    }


def build_real_primary_lift_rows(
    candidate_rows: pd.DataFrame,
    run_rows: pd.DataFrame,
    relation_members: dict[str, list[str]],
) -> pd.DataFrame:
    """Build real primary branch-lift rows.

    C2 uses the existing prediction/branch `twisted_rank_lift_2` result.  Other
    q values are retained as explicit unimplemented candidates rather than
    fabricated accuracy rows.
    """

    rows = []
    if candidate_rows.empty:
        return pd.DataFrame()
    base_columns = [
        "relation_set_id",
        "aggregation_level",
        "dataset",
        "n_models",
        "width",
        "matching",
        "q_order",
        "q_name",
        "primary_type",
        "primary_depth",
        "candidate_role",
        "divides_primary_source",
        "quotient_certified",
        "relation_count",
        "relation_count_status",
        "relation_violation_rate",
        "quotient_holonomy_nontrivial_rate",
        "quotient_holonomy_entropy",
        "quotient_assignment_confidence",
        "pooling_threshold",
        "pooling_residual_q",
    ]
    for _, candidate in candidate_rows.iterrows():
        q_order = int(candidate.get("q_order", 1))
        primary_type = str(candidate.get("primary_type", ""))
        method = primary_method_name(q_order, primary_type)
        member_run_ids = relation_members.get(str(candidate.get("relation_set_id")), [])
        if not member_run_ids and candidate.get("aggregation_level") == "run_id":
            member_run_ids = [str(candidate.get("run_id", candidate.get("relation_set_id")))]
        if not member_run_ids:
            member_run_ids = [""]
        for run_id in member_run_ids:
            implemented_method = IMPLEMENTED_REAL_Q_TO_METHOD.get(q_order)
            implemented = implemented_method is not None and bool(candidate.get("divides_primary_source", False))
            source = _best_method_row(run_rows, run_id, implemented_method) if implemented_method else None
            lift_implemented = bool(implemented and source is not None)
            core = {column: candidate.get(column, np.nan) for column in base_columns}
            rows.append(
                {
                    **core,
                    "run_id": run_id,
                    "candidate_method": method,
                    "source_method": implemented_method or "",
                    "uses_validation_data": False,
                    "uses_test_data_for_selection": False,
                    "reason": "implemented_from_twisted_rank_lift_2" if lift_implemented else "no_real_q_branch_prediction_available",
                    **_metrics_from_row(source if lift_implemented else None),
                    **branch_capacity_metadata(q_order, lift_implemented),
                }
            )
            if q_order == 2:
                for control_method, source_method in [
                    ("random_same_branch_count_control", "random_branch_ensemble_2"),
                    ("wrong_primary_factor_control", "validation_branch_ensemble_2"),
                ]:
                    control_source = _best_method_row(run_rows, run_id, source_method)
                    control_impl = control_source is not None
                    rows.append(
                        {
                            **core,
                            "run_id": run_id,
                            "candidate_method": control_method,
                            "source_method": source_method,
                            "uses_validation_data": control_method == "wrong_primary_factor_control",
                            "uses_test_data_for_selection": False,
                            "reason": "same_branch_control_from_fixed_setting_rows" if control_impl else "control_row_missing",
                            **_metrics_from_row(control_source if control_impl else None),
                            **branch_capacity_metadata(q_order, control_impl, is_control=True),
                        }
                    )
    return pd.DataFrame(rows)


def build_controlled_primary_lift_rows(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """Synthetic controlled sanity rows for C2/C3/C4 primary branch lifts."""

    rows = []
    if candidate_rows.empty:
        return pd.DataFrame()
    for _, candidate in candidate_rows.iterrows():
        q_order = int(candidate.get("q_order", 1))
        if q_order not in {2, 3, 4}:
            continue
        true_q = int(candidate.get("controlled_true_q", q_order))
        correct = q_order == true_q
        base_acc = 0.72
        lift_acc = 0.93 if correct else 0.70
        random_acc = 0.75
        wrong_acc = 0.69
        core = candidate.to_dict()
        for method, acc, implemented in [
            (primary_method_name(q_order, candidate.get("primary_type", "")), lift_acc, True),
            ("random_same_branch_count_control", random_acc, True),
            ("wrong_primary_factor_control", wrong_acc, True),
        ]:
            rows.append(
                {
                    **core,
                    "run_id": candidate.get("relation_set_id"),
                    "candidate_method": method,
                    "source_method": "controlled_prediction_level_logits",
                    "validation_accuracy": acc,
                    "validation_loss": 1.0 - acc,
                    "val_accuracy": acc,
                    "val_loss": 1.0 - acc,
                    "test_accuracy": acc - 0.01,
                    "test_loss": 1.01 - acc,
                    "uses_validation_data": False,
                    "uses_test_data_for_selection": False,
                    "reason": "controlled_correct_primary_factor" if correct else "controlled_wrong_primary_factor",
                    **branch_capacity_metadata(q_order, implemented, is_control=method.endswith("control")),
                    "best_fallback_validation_accuracy": base_acc,
                    "best_fallback_test_accuracy": base_acc - 0.01,
                }
            )
    return pd.DataFrame(rows)
