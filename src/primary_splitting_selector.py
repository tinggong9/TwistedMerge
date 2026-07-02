"""Validation-safe selector for p-primary holonomy branch lifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation_gated_period_index_lift import SelectorPolicy, best_overall_fallback, torsion_safe_selector


REAL_HEADLINE_METHOD_PREFIXES = ("prime_C",)
CONTROLLED_STRUCTURED_METHOD_PREFIXES = ("prime_C", "depth_C")


def _num(rows: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in rows:
        return pd.Series([default] * len(rows), index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _bool_col(rows: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in rows:
        return pd.Series([default] * len(rows), index=rows.index, dtype=bool)
    return rows[column].fillna(default).astype(bool)


def primary_holonomy_safe_selector(
    fallback_rows: pd.DataFrame,
    lift_rows: pd.DataFrame,
    policy: SelectorPolicy = SelectorPolicy(),
    epsilon_control: float = 0.0,
    pooling_threshold: float = 1e-8,
    lambda_branch: float = 0.0,
    lambda_residual: float = 0.0,
) -> pd.DataFrame:
    """Select primary lifts by validation only, with same-branch controls."""

    fallback = best_overall_fallback(fallback_rows)
    if fallback.empty:
        return fallback
    lifts = pd.DataFrame() if lift_rows is None else lift_rows.copy()
    if lifts.empty:
        selected = torsion_safe_selector(fallback_rows, pd.DataFrame(), policy)
        selected["selector_method"] = "primary_holonomy_safe_selector"
        selected["selected_primary_lift"] = False
        selected["selected_depth"] = 0
        selected["selector_no_test_leakage"] = True
        selected["selected_by_validation"] = False
        selected["control_gate_passed"] = False
        return selected

    lifts["val_accuracy"] = _num(lifts, "val_accuracy")
    lifts["val_loss"] = _num(lifts, "val_loss")
    lifts["pooling_residual_q"] = _num(lifts, "pooling_residual_q", np.inf)
    lifts["branch_count"] = _num(lifts, "branch_count", 1.0)
    lifts["selection_score"] = (
        lifts["val_accuracy"]
        - float(lambda_branch) * np.log(np.maximum(1.0, lifts["branch_count"]))
        - float(lambda_residual) * lifts["pooling_residual_q"].fillna(np.inf)
    )
    candidate_method = lifts["candidate_method"].astype(str)
    residual_source = lifts.get("residual_source", pd.Series(["real"] * len(lifts), index=lifts.index)).astype(str)
    relation_status = lifts.get("relation_count_status", pd.Series(["sufficient"] * len(lifts), index=lifts.index)).astype(str)
    real_headline_selectable = (
        ~residual_source.eq("controlled_sanity")
        & candidate_method.str.startswith(REAL_HEADLINE_METHOD_PREFIXES)
        & _bool_col(lifts, "is_prime_headline", False)
        & _bool_col(lifts, "eligible_by_observed_primary_depth", False)
        & ~_bool_col(lifts, "is_depth_control", False)
        & ~_bool_col(lifts, "is_mixed_control", False)
        & ~relation_status.eq("underconstrained")
    )
    controlled_selectable = (
        residual_source.eq("controlled_sanity")
        & candidate_method.str.startswith(CONTROLLED_STRUCTURED_METHOD_PREFIXES)
    )

    eligible = lifts[
        (lifts.get("quotient_certified", False) == True)  # noqa: E712
        & (lifts.get("lift_implemented", False) == True)  # noqa: E712
        & (lifts["pooling_residual_q"] <= float(pooling_threshold))
        & (real_headline_selectable | controlled_selectable)
    ].copy()
    if eligible.empty:
        selected = torsion_safe_selector(fallback_rows, pd.DataFrame(), policy)
        selected["selector_method"] = "primary_holonomy_safe_selector"
        selected["selected_primary_lift"] = False
        selected["selected_depth"] = 0
        selected["selector_no_test_leakage"] = True
        selected["selector_epsilon_control"] = float(epsilon_control)
        selected["pooling_threshold"] = float(pooling_threshold)
        selected["lambda_branch"] = float(lambda_branch)
        selected["lambda_residual"] = float(lambda_residual)
        selected["selected_by_validation"] = False
        selected["control_gate_passed"] = False
        return selected

    selections = []
    for _, base in fallback.iterrows():
        run_id = str(base["run_id"])
        pool = eligible[eligible["run_id"].astype(str).eq(run_id)].copy()
        selected = base.to_dict()
        selected.update(
            {
                "selector_method": "primary_holonomy_safe_selector",
                "selected_candidate_method": base.get("candidate_method", "best_fallback"),
                "selected_primary_lift": False,
                "selected_lift": False,
                "selected_q_order": 1,
                "selected_depth": 0,
                "selected_prime_axis": "none",
                "selected_by_validation": False,
                "control_gate_passed": False,
                "selector_no_test_leakage": True,
                "selector_epsilon": float(policy.epsilon),
                "selector_epsilon_control": float(epsilon_control),
                "selector_loss_slack": float(policy.loss_slack),
                "pooling_threshold": float(pooling_threshold),
                "lambda_branch": float(lambda_branch),
                "lambda_residual": float(lambda_residual),
                "best_fallback_val_accuracy": float(base.get("val_accuracy", np.nan)),
                "best_fallback_val_loss": float(base.get("val_loss", np.nan)),
                "best_fallback_test_accuracy": float(base.get("test_accuracy", np.nan)),
                "best_fallback_test_loss": float(base.get("test_loss", np.nan)),
            }
        )
        if pool.empty:
            selections.append(selected)
            continue
        controls = lifts[
            lifts["run_id"].astype(str).eq(run_id)
            & lifts["candidate_method"].astype(str).eq("random_same_branch_count_control")
        ].copy()
        wrong_controls = lifts[
            lifts["run_id"].astype(str).eq(run_id)
            & (
                lifts["candidate_method"].astype(str).eq("wrong_primary_factor_control")
                | lifts.get("candidate_role", pd.Series([""] * len(lifts), index=lifts.index)).astype(str).str.contains(
                    "wrong", case=False, na=False
                )
                | (
                    ~lifts.get("residual_source", pd.Series(["real"] * len(lifts), index=lifts.index)).astype(str).eq("controlled_sanity")
                    & (
                        _bool_col(lifts, "is_depth_control", False)
                        | _bool_col(lifts, "is_mixed_control", False)
                    )
                )
            )
        ].copy()
        control_best = {}
        for q_order, group in controls.groupby("q_order", sort=False):
            group = group.sort_values(["val_accuracy", "val_loss"], ascending=[False, True])
            control_best[int(q_order)] = float(group.iloc[0].get("val_accuracy", -np.inf))
        wrong_control_best = {}
        if not wrong_controls.empty:
            wrong_controls["val_accuracy"] = pd.to_numeric(wrong_controls["val_accuracy"], errors="coerce")
            wrong_controls["val_loss"] = pd.to_numeric(wrong_controls["val_loss"], errors="coerce")
            valid_wrong = wrong_controls[wrong_controls["val_accuracy"].notna()].copy()
            if not valid_wrong.empty:
                wrong_control_best["all"] = float(valid_wrong["val_accuracy"].max())
                for q_order, group in valid_wrong.groupby("q_order", sort=False):
                    group = group.sort_values(["val_accuracy", "val_loss"], ascending=[False, True])
                    wrong_control_best[int(q_order)] = float(group.iloc[0].get("val_accuracy", -np.inf))

        passing_rows = []
        for _, lift in pool.iterrows():
            q_order = int(lift.get("q_order", 1))
            control_val = control_best.get(q_order, -np.inf)
            wrong_val = max(
                wrong_control_best.get(q_order, -np.inf),
                wrong_control_best.get("all", -np.inf),
            )
            if (
                float(lift.get("val_accuracy", np.nan)) >= float(base["val_accuracy"]) + float(policy.epsilon)
                and float(lift.get("val_loss", np.nan)) <= float(base["val_loss"]) + float(policy.loss_slack)
                and float(lift.get("val_accuracy", np.nan)) >= control_val + float(epsilon_control)
                and float(lift.get("val_accuracy", np.nan)) >= wrong_val + float(epsilon_control)
            ):
                lifted = lift.copy()
                lifted["best_random_same_branch_val_accuracy"] = control_val
                lifted["best_wrong_prime_or_depth_val_accuracy"] = wrong_val
                lifted["control_gate_passed"] = True
                lifted["selected_by_validation"] = True
                passing_rows.append(lifted)
        if passing_rows:
            passing = pd.DataFrame(passing_rows).sort_values(
                ["selection_score", "val_accuracy", "val_loss", "candidate_method"],
                ascending=[False, False, True, True],
            )
            lift = passing.iloc[0].to_dict()
            selected.update(lift)
            selected.update(
                {
                    "selector_method": "primary_holonomy_safe_selector",
                    "selected_candidate_method": lift.get("candidate_method", "primary_lift"),
                    "selected_primary_lift": True,
                    "selected_lift": True,
                    "selected_q_order": int(lift.get("q_order", 1)),
                    "selected_depth": int(lift.get("primary_depth", 0)),
                    "selected_prime_axis": lift.get("prime_axis", "none"),
                    "selected_by_validation": True,
                    "control_gate_passed": True,
                    "selector_no_test_leakage": True,
                    "selector_epsilon": float(policy.epsilon),
                    "selector_epsilon_control": float(epsilon_control),
                    "selector_loss_slack": float(policy.loss_slack),
                    "pooling_threshold": float(pooling_threshold),
                    "lambda_branch": float(lambda_branch),
                    "lambda_residual": float(lambda_residual),
                }
            )
        selections.append(selected)
    return pd.DataFrame(selections)
