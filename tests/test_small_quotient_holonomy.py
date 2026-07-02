import pandas as pd

from src.small_quotient_branch_lift import (
    build_small_quotient_lift_candidates,
    quotient_capacity_metadata,
    quotient_pooling_residual_summary,
)
from src.small_quotient_holonomy import (
    bootstrap_quotient_fit,
    fit_quotient_map,
    fit_summary_row,
    quotient_certified,
    quotient_group,
    triangle_relation_from_perms,
)
from src.small_quotient_selector import small_quotient_holonomy_safe_selector
from src.validation_gated_period_index_lift import SelectorPolicy


def test_c2_parity_quotient_fit_certifies_nontrivial_triangle():
    identity = (0, 1)
    flip = (1, 0)
    relation = triangle_relation_from_perms(flip, identity, identity, flip)
    fit = fit_quotient_map([relation], "C2", seed=0, random_restarts=2)
    boot = bootstrap_quotient_fit(fit, relation_threshold=0.0, nontrivial_threshold=0.5, n_bootstrap=10, seed=1)

    assert fit.Q_order == 2
    assert fit.relation_violation_rate == 0.0
    assert fit.quotient_holonomy_nontrivial_rate == 1.0
    assert quotient_certified(fit, boot, relation_threshold=0.0, nontrivial_threshold=0.5)


def test_small_quotient_pooling_kills_regular_branch_action():
    identity = (0, 1)
    flip = (1, 0)
    relation = triangle_relation_from_perms(flip, identity, identity, flip)
    fit = fit_quotient_map([relation], "C2", seed=0, random_restarts=0)
    residual = quotient_pooling_residual_summary(fit, feature_dim=3)

    assert residual["naive_quotient_residual"] > 0.0
    assert residual["invariant_pooling_residual"] < 1e-12


def test_lift_capacity_metadata_marks_branch_extra_capacity():
    meta = quotient_capacity_metadata("small_quotient_invariant_pooling_lift", Q_order=4)

    assert meta["branch_count"] == 4
    assert meta["inference_multiplier"] == 4.0
    assert meta["is_extra_capacity"] is True
    assert meta["capacity_matched_to_same_branch_control"] is True
    assert meta["lift_level"] == "level_1_diagnostic_only"


def test_selector_falls_back_when_q_lift_is_diagnostic_only():
    fallback = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "method": "greedy_soup",
                "val_accuracy": 0.8,
                "val_loss": 0.5,
                "test_accuracy": 0.79,
                "test_loss": 0.52,
            },
            {
                "run_id": "r0",
                "method": "c2m3_synchronized",
                "val_accuracy": 0.75,
                "val_loss": 0.55,
                "test_accuracy": 0.74,
                "test_loss": 0.57,
            },
        ]
    )
    identity = (0, 1)
    flip = (1, 0)
    relation = triangle_relation_from_perms(flip, identity, identity, flip)
    fit = fit_quotient_map([relation], "C2", seed=0, random_restarts=0)
    boot = bootstrap_quotient_fit(fit, relation_threshold=0.0, nontrivial_threshold=0.5, n_bootstrap=5, seed=2)
    row = fit_summary_row(
        fit,
        relation_threshold=0.0,
        nontrivial_threshold=0.5,
        bootstrap=boot,
        base={"run_id": "r0", "dataset": "unit", "residual_source": "unit"},
    )
    pooling = pd.DataFrame([{**row, **quotient_pooling_residual_summary(fit), "pooling_threshold": 1e-8, "pooling_gate_passed": True}])
    lifts = build_small_quotient_lift_candidates(pooling)

    selected = small_quotient_holonomy_safe_selector(fallback, lifts, SelectorPolicy())

    assert len(selected) == 1
    assert bool(selected.iloc[0]["selected_small_quotient_lift"]) is False
    assert selected.iloc[0]["selected_candidate_method"] == "best_fallback"


def test_quotient_group_catalog_sizes():
    assert quotient_group("C2").order == 2
    assert quotient_group("C3").order == 3
    assert quotient_group("C4").order == 4
    assert quotient_group("V4").order == 4
    assert quotient_group("S3").order == 6
    assert quotient_group("D4").order == 8
