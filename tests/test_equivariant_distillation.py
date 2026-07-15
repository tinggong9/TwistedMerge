import numpy as np

from experiments.equivariant_distillation import action_logits, distillation_targets
from experiments.strong_compositional_baselines import build_group


def test_distillation_targets_are_label_or_teacher_derived():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]]); labels = np.array([0, 1])
    for objective in ["kl", "supervised", "chart_action", "group_law_consistency", "mixed"]:
        targets = distillation_targets(logits, labels, objective)
        assert targets.shape == logits.shape
        assert np.allclose(targets.sum(1), 1.0)


def test_action_logits_has_expected_shape():
    group = build_group("S3"); base = np.arange(18).reshape(3, 6); actions = np.array([0, 1, 2])
    assert action_logits(base, group, actions).shape == base.shape
