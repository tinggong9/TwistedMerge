import pandas as pd

from src.controlled_nonabelian_holonomy import (
    controlled_group,
    controlled_nonabelian_safe_selector,
    planted_case,
    residuals_for_case,
)
from src.finite_group_cohomology import permutation_order


def test_s3_and_d4_group_operations_are_correct():
    s3 = controlled_group("S3")
    d4 = controlled_group("D4")
    assert s3.order == 6
    assert d4.order == 8
    assert all(s3.multiply(element, s3.inverse(element)) == s3.identity for element in s3.elements)
    assert all(d4.multiply(element, d4.inverse(element)) == d4.identity for element in d4.elements)
    assert max(permutation_order(element) for element in s3.elements) == 3
    assert max(permutation_order(element) for element in d4.elements) == 4


def test_family_a_has_trivial_holonomy():
    case = planted_case("S3", "trivial_coboundary")
    assert case.holonomy == case.group.identity
    assert case.holonomy_order == 1


def test_family_b_has_nonidentity_noncentral_holonomy():
    for group_name in ["S3", "D4"]:
        case = planted_case(group_name, "planted_nonabelian_holonomy")
        assert case.holonomy != case.group.identity
        assert not case.is_holonomy_central
        assert case.holonomy_order > 1


def test_residuals_show_naive_failure_and_pooled_success():
    case = planted_case("D4", "planted_nonabelian_holonomy")
    residuals = residuals_for_case(case, feature_dim=4)
    assert residuals["naive_representation_residual"] > 0.0
    assert residuals["invariant_pooling_residual"] < 1e-12


def test_null_family_is_not_marked_trivial_by_construction():
    case = planted_case("S3", "random_noncoherent_null", seed=7)
    assert case.holonomy != case.group.identity


def test_validation_selector_does_not_use_test_accuracy():
    rows = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "family": "planted_nonabelian_holonomy",
                "method": "unlifted_c2m3_sync",
                "validation_accuracy": 0.9,
                "validation_loss": 0.2,
                "test_accuracy": 0.5,
                "test_loss": 1.0,
                "stable_group_action": True,
                "invariant_pooling_residual": 0.0,
            },
            {
                "run_id": "r0",
                "family": "planted_nonabelian_holonomy",
                "method": "branch_regular_lift_with_invariant_pooling",
                "validation_accuracy": 0.8,
                "validation_loss": 0.3,
                "test_accuracy": 0.99,
                "test_loss": 0.01,
                "stable_group_action": True,
                "invariant_pooling_residual": 0.0,
            },
        ]
    )
    selected = controlled_nonabelian_safe_selector(rows, epsilon=0.0, loss_slack=float("inf"))
    assert selected.iloc[0]["selected_method"] == "unlifted_c2m3_sync"
    assert bool(selected.iloc[0]["selector_no_test_leakage"])


def test_null_controls_do_not_activate_branch_lift_without_stable_group_action():
    rows = pd.DataFrame(
        [
            {
                "run_id": "null0",
                "family": "random_noncoherent_null",
                "method": "unlifted_c2m3_sync",
                "validation_accuracy": 0.7,
                "validation_loss": 0.5,
                "test_accuracy": 0.7,
                "test_loss": 0.5,
                "stable_group_action": False,
                "invariant_pooling_residual": 0.0,
            },
            {
                "run_id": "null0",
                "family": "random_noncoherent_null",
                "method": "branch_regular_lift_with_invariant_pooling",
                "validation_accuracy": 0.95,
                "validation_loss": 0.1,
                "test_accuracy": 0.95,
                "test_loss": 0.1,
                "stable_group_action": False,
                "invariant_pooling_residual": 0.0,
            },
        ]
    )
    selected = controlled_nonabelian_safe_selector(rows, epsilon=0.0, loss_slack=float("inf"))
    assert not bool(selected.iloc[0]["selected_branch_lift"])


def test_trivial_coboundary_does_not_activate_branch_without_obstruction():
    rows = pd.DataFrame(
        [
            {
                "run_id": "trivial0",
                "family": "trivial_coboundary",
                "method": "unlifted_c2m3_sync",
                "validation_accuracy": 0.9,
                "validation_loss": 0.2,
                "test_accuracy": 0.9,
                "test_loss": 0.2,
                "stable_group_action": True,
                "ordinary_sync_residual": 0.0,
                "invariant_pooling_residual": 0.0,
            },
            {
                "run_id": "trivial0",
                "family": "trivial_coboundary",
                "method": "branch_regular_lift_with_invariant_pooling",
                "validation_accuracy": 0.95,
                "validation_loss": 0.1,
                "test_accuracy": 0.95,
                "test_loss": 0.1,
                "stable_group_action": True,
                "ordinary_sync_residual": 0.0,
                "invariant_pooling_residual": 0.0,
            },
        ]
    )
    selected = controlled_nonabelian_safe_selector(rows, epsilon=0.0, loss_slack=float("inf"))
    assert not bool(selected.iloc[0]["selected_branch_lift"])
