from __future__ import annotations

import numpy as np

from src.twist_distillation import distill_linear_student


def test_distillation_reduces_teacher_kl() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(600, 4))
    teacher_logits = features @ rng.normal(size=(4, 3))
    student, history = distill_linear_student(features, teacher_logits, steps=700, learning_rate=0.2)
    assert history[-1] < history[0] * 0.02
    assert student.logits(features).shape == teacher_logits.shape
