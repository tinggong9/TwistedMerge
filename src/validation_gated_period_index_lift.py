"""Validation-gated period/index lift selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_RANKS = (1, 2, 3, 4, 6, 8, 9, 12, 16)


@dataclass(frozen=True)
class SelectorPolicy:
    epsilon: float = 0.0
    loss_slack: float = 0.0


def classify_rank(period: int | None, index: int | None, rank: int) -> str:
    """Classify a candidate lift rank against period/index divisibility."""

    if period is None or index is None or period <= 0 or index <= 0:
        return "no_certified_period_index"
    if int(rank) % int(period) != 0:
        return "rank_not_period_divisible"
    if int(rank) % int(index) != 0:
        return "period_divisible_index_obstructed"
    return "index_divisible_lift_allowed"


def period_index_rows_for_candidate(
    candidate: dict,
    candidate_ranks: Iterable[int] = DEFAULT_CANDIDATE_RANKS,
) -> list[dict]:
    period = candidate.get("estimated_period")
    index = candidate.get("estimated_index")
    rows = []
    for rank in candidate_ranks:
        decision = classify_rank(
            int(period) if pd.notna(period) else None,
            int(index) if pd.notna(index) else None,
            int(rank),
        )
        rows.append(
            {
                **candidate,
                "candidate_rank": int(rank),
                "rank_decision": decision,
                "lift_allowed_by_index": decision == "index_divisible_lift_allowed",
            }
        )
    return rows


def fallback_method_family(method: str) -> str | None:
    name = str(method)
    if name == "greedy_soup":
        return "fallback_greedy_soup"
    if name == "c2m3_synchronized":
        return "fallback_c2m3"
    if name.startswith("monomial_gauge"):
        return "fallback_monomial"
    if name in {"validated_ladder_selector", "improved_validated_selector"}:
        return "fallback_validated_selector"
    return None


def best_fallbacks(run_rows: pd.DataFrame) -> pd.DataFrame:
    """Return the best validation fallback row per run and fallback family."""

    if run_rows.empty:
        return pd.DataFrame()
    rows = run_rows.copy()
    rows["candidate_method"] = rows["method"].map(fallback_method_family)
    rows = rows[rows["candidate_method"].notna()].copy()
    if rows.empty:
        return rows
    rows["val_accuracy"] = pd.to_numeric(rows["val_accuracy"], errors="coerce")
    rows["val_loss"] = pd.to_numeric(rows["val_loss"], errors="coerce")
    rows["test_accuracy"] = pd.to_numeric(rows["test_accuracy"], errors="coerce")
    rows["test_loss"] = pd.to_numeric(rows["test_loss"], errors="coerce")
    sort_cols = [
        "run_id",
        "candidate_method",
        "val_accuracy",
        "val_loss",
        "method",
    ]
    rows = rows.sort_values(sort_cols, ascending=[True, True, False, True, True])
    return rows.groupby(["run_id", "candidate_method"], as_index=False, dropna=False).head(1)


def best_overall_fallback(run_rows: pd.DataFrame) -> pd.DataFrame:
    fallbacks = best_fallbacks(run_rows)
    if fallbacks.empty:
        return fallbacks
    sort_cols = ["run_id", "val_accuracy", "val_loss", "candidate_method"]
    fallbacks = fallbacks.sort_values(sort_cols, ascending=[True, False, True, True])
    out = fallbacks.groupby("run_id", as_index=False, dropna=False).head(1).copy()
    out["candidate_method"] = "best_fallback"
    return out


def torsion_safe_selector(
    fallback_rows: pd.DataFrame,
    lift_rows: pd.DataFrame | None = None,
    policy: SelectorPolicy = SelectorPolicy(),
) -> pd.DataFrame:
    """Select by validation only, activating lifts only when gates pass."""

    best_fb = best_overall_fallback(fallback_rows)
    if best_fb.empty:
        return best_fb
    lift_rows = pd.DataFrame() if lift_rows is None else lift_rows.copy()
    selections = []
    for _, fallback in best_fb.iterrows():
        run_id = fallback["run_id"]
        eligible = lift_rows[
            (lift_rows.get("run_id", pd.Series(dtype=object)).astype(str) == str(run_id))
            & (lift_rows.get("certified_torsion", False) == True)  # noqa: E712
            & (lift_rows.get("lift_allowed_by_index", False) == True)  # noqa: E712
        ].copy()
        selected = fallback.to_dict()
        selected.update(
            {
                "selector_method": "torsion_safe_selector",
                "selected_candidate_method": fallback.get("candidate_method", "best_fallback"),
                "selected_lift": False,
                "selector_no_test_leakage": True,
                "selector_epsilon": float(policy.epsilon),
                "selector_loss_slack": float(policy.loss_slack),
                "best_fallback_val_accuracy": float(fallback.get("val_accuracy", np.nan)),
                "best_fallback_val_loss": float(fallback.get("val_loss", np.nan)),
                "best_fallback_test_accuracy": float(fallback.get("test_accuracy", np.nan)),
                "best_fallback_test_loss": float(fallback.get("test_loss", np.nan)),
            }
        )
        if not eligible.empty:
            eligible["val_accuracy"] = pd.to_numeric(eligible["val_accuracy"], errors="coerce")
            eligible["val_loss"] = pd.to_numeric(eligible["val_loss"], errors="coerce")
            pass_mask = (
                eligible["val_accuracy"] >= float(fallback["val_accuracy"]) + float(policy.epsilon)
            ) & (
                eligible["val_loss"] <= float(fallback["val_loss"]) + float(policy.loss_slack)
            )
            eligible = eligible[pass_mask].sort_values(
                ["val_accuracy", "val_loss", "candidate_method"],
                ascending=[False, True, True],
            )
            if not eligible.empty:
                lift = eligible.iloc[0].to_dict()
                selected.update(lift)
                selected.update(
                    {
                        "selector_method": "torsion_safe_selector",
                        "selected_candidate_method": lift.get("candidate_method", "period_index_lift"),
                        "selected_lift": True,
                        "selector_no_test_leakage": True,
                        "selector_epsilon": float(policy.epsilon),
                        "selector_loss_slack": float(policy.loss_slack),
                        "best_fallback_val_accuracy": float(fallback.get("val_accuracy", np.nan)),
                        "best_fallback_val_loss": float(fallback.get("val_loss", np.nan)),
                        "best_fallback_test_accuracy": float(fallback.get("test_accuracy", np.nan)),
                        "best_fallback_test_loss": float(fallback.get("test_loss", np.nan)),
                    }
                )
        selections.append(selected)
    return pd.DataFrame(selections)


def selector_regret(selected: pd.DataFrame, all_rows: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    rows = []
    numeric = all_rows.copy()
    numeric["test_accuracy"] = pd.to_numeric(numeric["test_accuracy"], errors="coerce")
    for _, sel in selected.iterrows():
        run_id = sel["run_id"]
        pool = numeric[numeric["run_id"].astype(str) == str(run_id)]
        oracle = float(pool["test_accuracy"].max()) if not pool.empty else np.nan
        selected_test = float(sel.get("test_accuracy", np.nan))
        rows.append(
            {
                "run_id": run_id,
                "selector_method": sel.get("selector_method", "torsion_safe_selector"),
                "selected_candidate_method": sel.get("selected_candidate_method", ""),
                "selected_lift": bool(sel.get("selected_lift", False)),
                "selector_no_test_leakage": bool(sel.get("selector_no_test_leakage", True)),
                "selected_test_accuracy": selected_test,
                "oracle_pool_test_accuracy": oracle,
                "test_regret_vs_oracle_pool": float(oracle - selected_test) if np.isfinite(oracle) else np.nan,
                "best_fallback_test_accuracy": float(sel.get("best_fallback_test_accuracy", np.nan)),
                "delta_vs_best_fallback": selected_test - float(sel.get("best_fallback_test_accuracy", np.nan)),
            }
        )
    return pd.DataFrame(rows)
