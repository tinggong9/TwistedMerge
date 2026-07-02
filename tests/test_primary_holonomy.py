import pandas as pd

from src.primary_branch_lift import build_real_primary_lift_rows
from src.primary_holonomy import (
    bootstrap_primary_fit,
    candidate_q_orders_for_source,
    fit_primary_quotient,
    p_adic_valuation,
    primary_fit_certified,
    primary_pooling_residuals,
    triangle_relation_from_perms,
)
from src.primary_splitting_selector import primary_holonomy_safe_selector
from src.validation_gated_period_index_lift import SelectorPolicy


def test_p_adic_valuation_and_candidate_roles():
    assert p_adic_valuation(6006, 2) == 1
    assert p_adic_valuation(6006, 3) == 1
    candidates = {row["q_order"]: row for row in candidate_q_orders_for_source(12)}
    assert candidates[2]["divides_primary_source"] is True
    assert candidates[4]["divides_primary_source"] is True
    assert candidates[8]["candidate_role"] == "wrong_factor_control"
    assert candidates[3]["primary_type"] == "3-primary"


def test_primary_fit_and_pooling_for_c2_relation():
    identity = (0, 1)
    flip = (1, 0)
    relations = [triangle_relation_from_perms(identity, identity, flip, flip) for _ in range(4)]
    fit = fit_primary_quotient(relations, 2)
    boot = bootstrap_primary_fit(fit, n_bootstrap=8, seed=1, entropy_threshold=0.0)
    pooling = primary_pooling_residuals(fit)

    assert fit.relation_violation_rate == 0.0
    assert fit.quotient_holonomy_nontrivial_rate == 1.0
    assert primary_fit_certified(fit, boot, relation_count=4, entropy_threshold=0.0)
    assert pooling["naive_residual_q"] > 0.0
    assert pooling["pooling_residual_q"] < 1e-12


def test_real_primary_lift_uses_existing_c2_rank_lift_rows():
    candidate_rows = pd.DataFrame(
        [
            {
                "relation_set_id": "setting::x",
                "aggregation_level": "setting_id",
                "dataset": "mnist",
                "n_models": 3,
                "width": 64,
                "matching": "monomial_weight",
                "q_order": 2,
                "q_name": "C2",
                "primary_type": "2-primary",
                "primary_depth": 1,
                "candidate_role": "primary_candidate",
                "divides_primary_source": True,
                "quotient_certified": True,
                "relation_count": 4,
                "relation_count_status": "sufficient",
                "relation_violation_rate": 0.0,
                "quotient_holonomy_nontrivial_rate": 1.0,
                "quotient_holonomy_entropy": 0.5,
                "quotient_assignment_confidence": 1.0,
                "pooling_threshold": 1e-8,
                "pooling_residual_q": 0.0,
            }
        ]
    )
    run_rows = pd.DataFrame(
        [
            {"run_id": "r0", "method": "twisted_rank_lift_2", "val_accuracy": 0.82, "val_loss": 0.4, "test_accuracy": 0.81, "test_loss": 0.42},
            {"run_id": "r0", "method": "random_branch_ensemble_2", "val_accuracy": 0.78, "val_loss": 0.45, "test_accuracy": 0.77, "test_loss": 0.47},
            {"run_id": "r0", "method": "validation_branch_ensemble_2", "val_accuracy": 0.79, "val_loss": 0.44, "test_accuracy": 0.78, "test_loss": 0.46},
        ]
    )
    lifts = build_real_primary_lift_rows(candidate_rows, run_rows, {"setting::x": ["r0"]})

    primary = lifts[lifts["candidate_method"].eq("primary_C2_branch_lift")].iloc[0]
    assert bool(primary["lift_implemented"]) is True
    assert primary["branch_count"] == 2
    assert primary["test_accuracy"] == 0.81
    assert set(lifts["candidate_method"]) >= {
        "primary_C2_branch_lift",
        "random_same_branch_count_control",
        "wrong_primary_factor_control",
    }


def test_primary_selector_requires_random_control_margin():
    fallback = pd.DataFrame(
        [
            {"run_id": "r0", "method": "greedy_soup", "val_accuracy": 0.80, "val_loss": 0.4, "test_accuracy": 0.79, "test_loss": 0.42}
        ]
    )
    lifts = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "candidate_method": "primary_C2_branch_lift",
                "q_order": 2,
                "primary_depth": 1,
                "quotient_certified": True,
                "lift_implemented": True,
                "pooling_residual_q": 0.0,
                "branch_count": 2,
                "val_accuracy": 0.83,
                "val_loss": 0.35,
                "test_accuracy": 0.82,
                "test_loss": 0.37,
            },
            {
                "run_id": "r0",
                "candidate_method": "random_same_branch_count_control",
                "q_order": 2,
                "quotient_certified": True,
                "lift_implemented": True,
                "pooling_residual_q": 0.0,
                "branch_count": 2,
                "val_accuracy": 0.829,
                "val_loss": 0.35,
                "test_accuracy": 0.82,
                "test_loss": 0.37,
            },
        ]
    )
    selected = primary_holonomy_safe_selector(fallback, lifts, SelectorPolicy(), epsilon_control=0.002)
    assert bool(selected.iloc[0]["selected_primary_lift"]) is False

    selected = primary_holonomy_safe_selector(fallback, lifts, SelectorPolicy(), epsilon_control=0.0)
    assert bool(selected.iloc[0]["selected_primary_lift"]) is True
