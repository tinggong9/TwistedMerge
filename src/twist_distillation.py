"""Label-independent logit distillation into a single linear student."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.twist_router import softmax


@dataclass
class LinearDistilledStudent:
    weights: np.ndarray
    bias: np.ndarray

    def logits(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features) @ self.weights + self.bias

    def probabilities(self, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.logits(features) / temperature)


def kl_divergence(teacher_probabilities: np.ndarray, student_probabilities: np.ndarray) -> float:
    teacher = np.clip(np.asarray(teacher_probabilities, dtype=float), 1e-12, 1.0)
    student = np.clip(np.asarray(student_probabilities, dtype=float), 1e-12, 1.0)
    return float(np.mean(np.sum(teacher * (np.log(teacher) - np.log(student)), axis=1)))


def distill_linear_student(
    features: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    temperature: float = 1.0,
    steps: int = 1000,
    learning_rate: float = 0.1,
    l2: float = 1e-5,
    seed: int = 0,
) -> tuple[LinearDistilledStudent, list[float]]:
    x = np.asarray(features, dtype=float)
    teacher = softmax(np.asarray(teacher_logits, dtype=float) / temperature)
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.01, size=(x.shape[1], teacher.shape[1]))
    bias = np.zeros(teacher.shape[1])
    history = []
    for step in range(steps):
        student = softmax((x @ weights + bias) / temperature)
        error = (student - teacher) / (len(x) * temperature)
        weights -= learning_rate * (x.T @ error + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)
        if step in {0, steps - 1}:
            history.append(kl_divergence(teacher, student))
    model = LinearDistilledStudent(weights, bias)
    history[-1] = kl_divergence(teacher, model.probabilities(x, temperature))
    return model, history
