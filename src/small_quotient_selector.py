"""Validation-safe selection for small quotient branch lifts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation_gated_period_index_lift import SelectorPolicy, best_overall_fallback, torsion_safe_selector


def _as_numeric(rows: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in rows:
        return pd.Series([default] * len(rows), index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def small_quotient_holonomy_safe_selector(
    fallback_rows: pd.DataFrame,
    lift_rows: pd.DataFrame,
    policy: SelectorPolicy = SelectorPolicy(),
    epsilon_control: float = 0.0,
    pooling_threshold: float = 1e-8,
) -> pd.DataFrame:
    """Select Q-lifts only when validation and certification gates pass.

    The selector is intentionally conservative: test accuracy is never used for
    selection, and a Q-lift must beat the best fallback and a same-Q random
    branch control on validation before it can be selected.
    """

    fallback = best_overall_fallback(fallback_rows)
    if fallback.empty:
        return fallback
    lift_rows = pd.DataFrame() if lift_rows is None else lift_rows.copy()
    if lift_rows.empty:
        selected = torsion_safe_selector(fallback_rows, pd.DataFrame(), policy)
        selected["selector_method"] = "small_quotient_holonomy_safe_selector"
        selected["selected_small_quotient_lift"] = False
        selected["selected_Q_name"] = ""
        selected["selector_no_test_leakage"] = True
        selected["selector_epsilon_control"] = float(epsilon_control)
        selected["pooling_threshold"] = float(pooling_threshold)
        return selected

    lifts = lift_rows.copy()
    lifts["val_accuracy"] = _as_numeric(lifts, "val_accuracy")
    lifts["val_loss"] = _as_numeric(lifts, "val_loss")
    lifts["invariant_pooling_residual"] = _as_numeric(lifts, "invariant_pooling_residual", np.inf)
    eligible = lifts[
        (lifts.get("quotient_certified", False) == True)  # noqa: E712
        & (lifts.get("diagnostic_gate_passed", False) == True)  # noqa: E712
        & (lifts.get("lift_implemented", False) == True)  # noqa: E712
        & (lifts["invariant_pooling_residual"] <= float(pooling_threshold))
    ].copy()
    if eligible.empty:
        selected = torsion_safe_selector(fallback_rows, pd.DataFrame(), policy)
        selected["selector_method"] = "small_quotient_holonomy_safe_selector"
        selected["selected_small_quotient_lift"] = False
        selected["selected_Q_name"] = ""
        selected["selector_no_test_leakage"] = True
        selected["selector_epsilon_control"] = float(epsilon_control)
        selected["pooling_threshold"] = float(pooling_threshold)
        return selected

    selections = []
    for _, base in fallback.iterrows():
        run_id = str(base["run_id"])
        pool = eligible[eligible["run_id"].astype(str).eq(run_id)].copy()
        selected = base.to_dict()
        selected.update(
            {
                "selector_method": "small_quotient_holonomy_safe_selector",
                "selected_candidate_method": base.get("candidate_method", "best_fallback"),
                "selected_small_quotient_lift": False,
                "selected_lift": False,
                "selected_Q_name": "",
                "selector_no_test_leakage": True,
                "selector_epsilon": float(policy.epsilon),
                "selector_epsilon_control": float(epsilon_control),
                "selector_loss_slack": float(policy.loss_slack),
                "pooling_threshold": float(pooling_threshold),
                "best_fallback_val_accuracy": float(base.get("val_accuracy", np.nan)),
                "best_fallback_val_loss": float(base.get("val_loss", np.nan)),
                "best_fallback_test_accuracy": float(base.get("test_accuracy", np.nan)),
                "best_fallback_test_loss": float(base.get("test_loss", np.nan)),
            }
        )
        if pool.empty:
            selections.append(selected)
            continue
        controls = pool[pool["candidate_method"].astype(str).eq("random_same_Q_branch_control")]
        control_best = {}
        if not controls.empty:
            controls = controls.sort_values(["Q_name", "val_accuracy", "val_loss"], ascending=[True, False, True])
            for q_name, q_group in controls.groupby("Q_name", sort=False):
                control_best[str(q_name)] = float(q_group.iloc[0].get("val_accuracy", -np.inf))
        structured = pool[
            pool["candidate_method"].astype(str).isin(
                {
                    "small_quotient_regular_branch_lift",
                    "small_quotient_orbit_branch_lift",
                    "small_quotient_invariant_pooling_lift",
                }
            )
        ].copy()
        passing_rows = []
        for _, lift in structured.iterrows():
            q_name = str(lift.get("Q_name", ""))
            control_val = control_best.get(q_name, -np.inf)
            if (
                float(lift.get("val_accuracy", np.nan)) >= float(base["val_accuracy"]) + float(policy.epsilon)
                and float(lift.get("val_loss", np.nan)) <= float(base["val_loss"]) + float(policy.loss_slack)
                and float(lift.get("val_accuracy", np.nan)) >= control_val + float(epsilon_control)
            ):
                lift = lift.copy()
                lift["best_random_same_Q_val_accuracy"] = control_val
                passing_rows.append(lift)
        if passing_rows:
            passing = pd.DataFrame(passing_rows).sort_values(
                ["val_accuracy", "val_loss", "candidate_method"],
                ascending=[False, True, True],
            )
            lift = passing.iloc[0].to_dict()
            selected.update(lift)
            selected.update(
                {
                    "selector_method": "small_quotient_holonomy_safe_selector",
                    "selected_candidate_method": lift.get("candidate_method", "small_quotient_lift"),
                    "selected_small_quotient_lift": True,
                    "selected_lift": True,
                    "selected_Q_name": lift.get("Q_name", ""),
                    "selector_no_test_leakage": True,
                }
            )
        selections.append(selected)
    return pd.DataFrame(selections)
