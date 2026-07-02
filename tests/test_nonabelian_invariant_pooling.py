import numpy as np

from src.controlled_nonabelian_holonomy import planted_case
from src.nonabelian_branch_lift import apply_branch_action, gamma_branch_lift
from src.nonabelian_invariant_pooling import (
    invariant_pool,
    naive_representation_residual,
    pooling_residual,
    regular_action_permutation,
)


def test_naive_regular_representation_keeps_nontrivial_holonomy():
    case = planted_case("S3", "planted_nonabelian_holonomy")
    perm = regular_action_permutation(case.group, case.holonomy)
    assert case.holonomy != case.group.identity
    assert naive_representation_residual(perm) > 0.0


def test_invariant_pooling_kills_branch_permutation():
    case = planted_case("D4", "planted_nonabelian_holonomy")
    perm = regular_action_permutation(case.group, case.holonomy)
    assert pooling_residual(perm, feature_dim=3) < 1e-12


def test_branch_action_preserves_pooled_features():
    case = planted_case("S3", "planted_nonabelian_holonomy")
    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(5, 4))
    lifted = gamma_branch_lift(hidden, case.group)
    moved = apply_branch_action(lifted, case.group, case.holonomy)
    np.testing.assert_allclose(invariant_pool(lifted), invariant_pool(moved), atol=1e-12)
