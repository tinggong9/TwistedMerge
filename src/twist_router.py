"""Small validation-trained router and representation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class LinearTwistRouter:
    n_features: int
    n_branches: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(scale=0.01, size=(self.n_features, self.n_branches))
        self.bias = np.zeros(self.n_branches)

    def fit(
        self, features: np.ndarray, branch_targets: np.ndarray, *, steps: int = 500, learning_rate: float = 0.1, l2: float = 1e-4
    ) -> "LinearTwistRouter":
        x = np.asarray(features, dtype=float)
        y = np.asarray(branch_targets, dtype=int)
        if x.ndim != 2 or x.shape[1] != self.n_features or y.shape != (x.shape[0],):
            raise ValueError("invalid router training shapes")
        target = np.eye(self.n_branches)[y]
        for _ in range(steps):
            probabilities = softmax(x @ self.weights + self.bias)
            error = (probabilities - target) / len(x)
            self.weights -= learning_rate * (x.T @ error + l2 * self.weights)
            self.bias -= learning_rate * error.sum(axis=0)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        return softmax(x @ self.weights + self.bias)

    def combine(self, features: np.ndarray, branch_logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(branch_logits, dtype=float)
        if logits.ndim != 3 or logits.shape[1] != self.n_branches:
            raise ValueError("branch_logits must have shape [examples, branches, classes]")
        return np.einsum("nb,nbc->nc", self.predict_proba(features), logits)


def representation_loss(
    representation: dict[object, np.ndarray], products: list[tuple[object, object, object]], projection: np.ndarray, generators: list[object]
) -> float:
    loss = 0.0
    for left, right, product in products:
        loss += float(np.linalg.norm(representation[product] - representation[left] @ representation[right]) ** 2)
    for generator in generators:
        loss += float(np.linalg.norm(projection @ representation[generator] - projection) ** 2)
    return loss


def invariant_projection(generator_matrices: list[np.ndarray], *, tolerance: float = 1e-8) -> np.ndarray:
    if not generator_matrices:
        raise ValueError("at least one generator is required")
    d = generator_matrices[0].shape[0]
    constraints = np.concatenate([(matrix - np.eye(d)).T for matrix in generator_matrices], axis=0)
    _, singular, vt = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.sum(singular > tolerance))
    null = vt[rank:]
    return null.T @ null if null.size else np.zeros((d, d))
