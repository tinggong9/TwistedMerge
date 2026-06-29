"""Synthetic local classification tasks for obstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import rotate_weights


@dataclass(frozen=True)
class LocalDataset:
    x_val: list[np.ndarray]
    y_val: list[np.ndarray]
    x_test: list[np.ndarray]
    y_test: list[np.ndarray]


def make_base_weight(dim: int, rng: np.random.Generator) -> np.ndarray:
    weight = rng.normal(size=dim)
    return weight / np.linalg.norm(weight)


def make_mu2_local_weights(
    base_weight: np.ndarray,
    true_gauges: np.ndarray,
    rng: np.random.Generator,
    model_noise: float,
) -> np.ndarray:
    weights = true_gauges.reshape(-1, 1) * base_weight.reshape(1, -1)
    if model_noise > 0:
        weights = weights + rng.normal(scale=model_noise, size=weights.shape)
    return weights


def make_u1_local_weights(
    base_weight: np.ndarray,
    true_phases: np.ndarray,
    rng: np.random.Generator,
    model_noise: float,
) -> np.ndarray:
    weights = rotate_weights(np.repeat(base_weight.reshape(1, -1), len(true_phases), axis=0), true_phases)
    if model_noise > 0:
        weights = weights + rng.normal(scale=model_noise, size=weights.shape)
    return weights


def labels_from_weight(x: np.ndarray, weight: np.ndarray, rng: np.random.Generator, label_noise: float) -> np.ndarray:
    logits = x @ weight
    y = (logits >= 0.0).astype(np.int64)
    if label_noise > 0:
        flips = rng.random(size=y.shape[0]) < label_noise
        y = np.where(flips, 1 - y, y)
    return y


def make_local_datasets(
    local_weights: np.ndarray,
    rng: np.random.Generator,
    n_val: int = 256,
    n_test: int = 512,
    label_noise: float = 0.02,
) -> LocalDataset:
    x_val: list[np.ndarray] = []
    y_val: list[np.ndarray] = []
    x_test: list[np.ndarray] = []
    y_test: list[np.ndarray] = []
    dim = local_weights.shape[1]
    for weight in local_weights:
        xv = rng.normal(size=(n_val, dim))
        xt = rng.normal(size=(n_test, dim))
        x_val.append(xv)
        y_val.append(labels_from_weight(xv, weight, rng, label_noise))
        x_test.append(xt)
        y_test.append(labels_from_weight(xt, weight, rng, label_noise))
    return LocalDataset(x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
