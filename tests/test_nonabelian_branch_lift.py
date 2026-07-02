import numpy as np

from src.controlled_nonabelian_holonomy import planted_case, synthetic_teacher_logits
from src.nonabelian_branch_lift import (
    branch_lift_with_invariant_pooling,
    oracle_true_branch_lift_logits,
    random_same_branch_count_control_logits,
)


def test_branch_lift_recovers_hidden_features_in_noiseless_case():
    case = planted_case("S3", "planted_nonabelian_holonomy")
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(8, 6))
    pooled = branch_lift_with_invariant_pooling(hidden, case.group, case.holonomy)
    np.testing.assert_allclose(pooled, hidden, atol=1e-12)


def test_oracle_branch_lift_preserves_teacher_logits():
    logits, _labels = synthetic_teacher_logits(seed=2, input_dim=5, hidden_width=6, n_samples=12)
    np.testing.assert_allclose(oracle_true_branch_lift_logits(logits), logits)


def test_random_branch_control_does_not_trivially_recover():
    logits, _labels = synthetic_teacher_logits(seed=3, input_dim=5, hidden_width=6, n_samples=20)
    rng = np.random.default_rng(4)
    random_logits = random_same_branch_count_control_logits(logits, branch_count=6, rng=rng, noise_scale=5.0)
    assert np.linalg.norm(random_logits - logits) > 1.0
