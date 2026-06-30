"""Finite time-frequency benchmark utilities for period-index detection.

The signal space is ``(C^d)^{tensor k}``.  On one cyclic factor the time shift
and frequency modulation operators are

    (T x)[n] = x[n - 1 mod d],      (M x)[n] = zeta^n x[n],

where ``zeta = exp(2*pi*i/d)``.  With this convention ``M T = zeta T M``.
The relation is the standard finite Heisenberg time-frequency symmetry of
discrete signals, not a synthetic neural residual planted after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


TIME_FREQUENCY_SCOPE_NOTE = (
    "Finite time-frequency operators provide a natural known-operator chart "
    "benchmark for central projective period-index detection.  This is not a "
    "MNIST/CIFAR residual claim, and learned model chart transitions are not "
    "certified by this module."
)


@dataclass(frozen=True)
class TimeFrequencyGenerators:
    d: int
    k: int
    zeta: complex
    T: tuple[np.ndarray, ...]
    M: tuple[np.ndarray, ...]
    dimension_complex: int
    dimension_real: int
    convention: str = "M_i T_i = zeta T_i M_i"


@dataclass(frozen=True)
class TimeFrequencyRelationCheck:
    d: int
    k: int
    dimension_complex: int
    max_relation_residual: float
    relation_residuals: dict[str, float]
    all_relations_hold: bool
    convention: str


@dataclass(frozen=True)
class TimeFrequencyDataset:
    d: int
    k: int
    n_classes: int
    noise_level: float
    seed: int
    prototypes: np.ndarray
    train_x_complex: np.ndarray
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x_complex: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x_complex: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    label_note: str = "labels are prototype identities; random time-frequency shifts are nuisance transformations"

    @property
    def dimension_complex(self) -> int:
        return int(self.d**self.k)

    @property
    def dimension_real(self) -> int:
        return 2 * self.dimension_complex


def primitive_time_frequency_root(d: int) -> complex:
    if d <= 0:
        raise ValueError("d must be positive")
    return complex(np.exp(2j * np.pi / d))


def time_shift_operator(d: int) -> np.ndarray:
    """Return T with ``(T x)[n] = x[n - 1 mod d]``."""

    if d <= 0:
        raise ValueError("d must be positive")
    matrix = np.zeros((d, d), dtype=complex)
    for col in range(d):
        matrix[(col + 1) % d, col] = 1.0
    return matrix


def frequency_modulation_operator(d: int, zeta: complex | None = None) -> np.ndarray:
    """Return M with ``(M x)[n] = zeta^n x[n]``."""

    if d <= 0:
        raise ValueError("d must be positive")
    root = primitive_time_frequency_root(d) if zeta is None else complex(zeta)
    return np.diag([root**n for n in range(d)]).astype(complex)


def complex_to_real_block_matrix(matrix: np.ndarray) -> np.ndarray:
    """Real block representation of a complex-linear matrix.

    For ``A = B + i C`` and realification ``z -> [Re z, Im z]``, the real
    matrix is ``[[B, -C], [C, B]]``.
    """

    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return np.block([[arr.real, -arr.imag], [arr.imag, arr.real]]).astype(float)


def complex_vector_to_real(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=complex)
    return np.concatenate([arr.real, arr.imag], axis=-1).astype(float)


def real_vector_to_complex(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    if arr.shape[-1] % 2 != 0:
        raise ValueError("last vector dimension must be even")
    half = arr.shape[-1] // 2
    return arr[..., :half] + 1j * arr[..., half:]


def tensor_on_factor(matrix: np.ndarray, factor: int, k: int, d: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    if arr.shape != (d, d):
        raise ValueError("matrix must have shape (d, d)")
    if k <= 0:
        raise ValueError("k must be positive")
    if factor < 0 or factor >= k:
        raise ValueError("factor must satisfy 0 <= factor < k")

    out = np.array([[1.0 + 0.0j]])
    identity = np.eye(d, dtype=complex)
    for idx in range(k):
        out = np.kron(out, arr if idx == factor else identity)
    return out


def time_frequency_generators(d: int, k: int) -> TimeFrequencyGenerators:
    if d <= 0:
        raise ValueError("d must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    zeta = primitive_time_frequency_root(d)
    T0 = time_shift_operator(d)
    M0 = frequency_modulation_operator(d, zeta)
    T = tuple(tensor_on_factor(T0, factor, k, d) for factor in range(k))
    M = tuple(tensor_on_factor(M0, factor, k, d) for factor in range(k))
    dimension = int(d**k)
    return TimeFrequencyGenerators(
        d=d,
        k=k,
        zeta=zeta,
        T=T,
        M=M,
        dimension_complex=dimension,
        dimension_real=2 * dimension,
    )


def time_frequency_generator_dict(d: int, k: int, *, realified: bool = False) -> dict[str, np.ndarray]:
    system = time_frequency_generators(d, k)
    generators: dict[str, np.ndarray] = {}
    for idx in range(k):
        generators[f"T{idx + 1}"] = system.T[idx]
        generators[f"M{idx + 1}"] = system.M[idx]
    if realified:
        return {name: complex_to_real_block_matrix(matrix) for name, matrix in generators.items()}
    return generators


def time_frequency_chart_operators(d: int, k: int) -> dict[str, np.ndarray]:
    system = time_frequency_generators(d, k)
    charts: dict[str, np.ndarray] = {"I": np.eye(system.dimension_complex, dtype=complex)}
    for idx in range(k):
        charts[f"T{idx + 1}"] = system.T[idx]
        charts[f"M{idx + 1}"] = system.M[idx]
        charts[f"T{idx + 1}M{idx + 1}"] = system.T[idx] @ system.M[idx]
    return charts


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(right, ord="fro")), 1e-12)
    return float(np.linalg.norm(left - right, ord="fro") / denom)


def check_time_frequency_relations(
    generators: TimeFrequencyGenerators,
    tolerance: float = 1e-10,
) -> TimeFrequencyRelationCheck:
    residuals: dict[str, float] = {}
    identity = np.eye(generators.dimension_complex, dtype=complex)
    for idx in range(generators.k):
        T = generators.T[idx]
        M = generators.M[idx]
        residuals[f"M{idx + 1}T{idx + 1}=zetaT{idx + 1}M{idx + 1}"] = _relative_residual(
            M @ T,
            generators.zeta * (T @ M),
        )
        commutator = T @ M @ np.linalg.inv(T) @ np.linalg.inv(M)
        residuals[f"commutator_T{idx + 1}_M{idx + 1}=zeta^-1"] = _relative_residual(
            commutator,
            np.conjugate(generators.zeta) * identity,
        )

    for i in range(generators.k):
        for j in range(generators.k):
            if i == j:
                continue
            residuals[f"T{i + 1}M{j + 1}_commutes"] = _relative_residual(
                generators.T[i] @ generators.M[j],
                generators.M[j] @ generators.T[i],
            )
    for i in range(generators.k):
        for j in range(i + 1, generators.k):
            residuals[f"T{i + 1}T{j + 1}_commutes"] = _relative_residual(
                generators.T[i] @ generators.T[j],
                generators.T[j] @ generators.T[i],
            )
            residuals[f"M{i + 1}M{j + 1}_commutes"] = _relative_residual(
                generators.M[i] @ generators.M[j],
                generators.M[j] @ generators.M[i],
            )
    max_residual = float(max(residuals.values(), default=0.0))
    return TimeFrequencyRelationCheck(
        d=generators.d,
        k=generators.k,
        dimension_complex=generators.dimension_complex,
        max_relation_residual=max_residual,
        relation_residuals=residuals,
        all_relations_hold=max_residual <= tolerance,
        convention=generators.convention,
    )


def _coordinate_grid(d: int, k: int) -> np.ndarray:
    return np.array(list(product(range(d), repeat=k)), dtype=float)


def chirp_gabor_prototypes(d: int, k: int, n_classes: int, seed: int = 0) -> np.ndarray:
    """Build normalized finite chirp/Gabor-like class prototypes."""

    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    rng = np.random.default_rng(seed)
    coords = _coordinate_grid(d, k)
    sigma = max(float(d) / 4.0, 0.75)
    prototypes: list[np.ndarray] = []
    root = primitive_time_frequency_root(d)
    for label in range(n_classes):
        centers = np.array([(label + 2 * factor) % d for factor in range(k)], dtype=float)
        raw_delta = np.abs(coords - centers)
        circular_delta = np.minimum(raw_delta, d - raw_delta)
        envelope = np.exp(-0.5 * np.sum(circular_delta**2, axis=1) / (sigma**2))
        quadratic = np.zeros(coords.shape[0], dtype=float)
        linear = np.zeros(coords.shape[0], dtype=float)
        for factor in range(k):
            quadratic += (label + factor + 1) * coords[:, factor] ** 2
            linear += (2 * label + factor + 1) * coords[:, factor]
        chirp = root ** ((quadratic + linear).astype(int) % d)
        jitter = 0.03 * (rng.normal(size=coords.shape[0]) + 1j * rng.normal(size=coords.shape[0]))
        proto = envelope * chirp + jitter
        norm = max(float(np.linalg.norm(proto)), 1e-12)
        prototypes.append((proto / norm).astype(complex))
    return np.asarray(prototypes, dtype=complex)


def _operator_from_exponents(system: TimeFrequencyGenerators, shifts: Iterable[int], modulations: Iterable[int]) -> np.ndarray:
    operator = np.eye(system.dimension_complex, dtype=complex)
    for idx, (shift, modulation) in enumerate(zip(shifts, modulations, strict=True)):
        factor_operator = (
            np.linalg.matrix_power(system.T[idx], int(shift) % system.d)
            @ np.linalg.matrix_power(system.M[idx], int(modulation) % system.d)
        )
        operator = factor_operator @ operator
    return operator


def apply_time_frequency_shift(
    vector: np.ndarray,
    d: int,
    k: int,
    shifts: Iterable[int],
    modulations: Iterable[int],
) -> np.ndarray:
    system = time_frequency_generators(d, k)
    return _operator_from_exponents(system, shifts, modulations) @ np.asarray(vector, dtype=complex)


def _sample_split(
    *,
    system: TimeFrequencyGenerators,
    prototypes: np.ndarray,
    n_samples: int,
    noise_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, prototypes.shape[0], size=n_samples)
    x = np.zeros((n_samples, system.dimension_complex), dtype=complex)
    for row, label in enumerate(labels):
        shifts = rng.integers(0, system.d, size=system.k)
        modulations = rng.integers(0, system.d, size=system.k)
        signal = _operator_from_exponents(system, shifts, modulations) @ prototypes[int(label)]
        if noise_level > 0:
            noise = rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
            signal = signal + (float(noise_level) / np.sqrt(2.0)) * noise
        x[row] = signal
    return x, labels.astype(int)


def generate_time_frequency_dataset(
    d: int,
    k: int,
    *,
    n_classes: int = 3,
    train_samples: int = 2000,
    validation_samples: int = 500,
    test_samples: int = 1000,
    noise_level: float = 0.0,
    seed: int = 0,
) -> TimeFrequencyDataset:
    if min(train_samples, validation_samples, test_samples) < 0:
        raise ValueError("sample counts must be nonnegative")
    if noise_level < 0:
        raise ValueError("noise_level must be nonnegative")
    system = time_frequency_generators(d, k)
    rng = np.random.default_rng(seed)
    prototypes = chirp_gabor_prototypes(d, k, n_classes, seed=seed)
    train_x_complex, train_y = _sample_split(
        system=system,
        prototypes=prototypes,
        n_samples=train_samples,
        noise_level=noise_level,
        rng=rng,
    )
    validation_x_complex, validation_y = _sample_split(
        system=system,
        prototypes=prototypes,
        n_samples=validation_samples,
        noise_level=noise_level,
        rng=rng,
    )
    test_x_complex, test_y = _sample_split(
        system=system,
        prototypes=prototypes,
        n_samples=test_samples,
        noise_level=noise_level,
        rng=rng,
    )
    return TimeFrequencyDataset(
        d=d,
        k=k,
        n_classes=n_classes,
        noise_level=float(noise_level),
        seed=int(seed),
        prototypes=prototypes,
        train_x_complex=train_x_complex,
        train_x=complex_vector_to_real(train_x_complex),
        train_y=train_y,
        validation_x_complex=validation_x_complex,
        validation_x=complex_vector_to_real(validation_x_complex),
        validation_y=validation_y,
        test_x_complex=test_x_complex,
        test_x=complex_vector_to_real(test_x_complex),
        test_y=test_y,
    )


def _orbit_templates(d: int, k: int, prototypes: np.ndarray) -> list[np.ndarray]:
    system = time_frequency_generators(d, k)
    templates: list[np.ndarray] = []
    for prototype in prototypes:
        class_templates = []
        for shifts in product(range(d), repeat=k):
            for modulations in product(range(d), repeat=k):
                signal = _operator_from_exponents(system, shifts, modulations) @ prototype
                class_templates.append(signal / max(float(np.linalg.norm(signal)), 1e-12))
        templates.append(np.asarray(class_templates, dtype=complex))
    return templates


def orbit_invariant_prototype_accuracy(
    dataset: TimeFrequencyDataset,
    split: str = "test",
) -> float:
    """Nearest-orbit prototype accuracy for the natural nuisance-shift task."""

    if split == "train":
        x_complex, y = dataset.train_x_complex, dataset.train_y
    elif split == "validation":
        x_complex, y = dataset.validation_x_complex, dataset.validation_y
    elif split == "test":
        x_complex, y = dataset.test_x_complex, dataset.test_y
    else:
        raise ValueError("split must be one of train, validation, or test")
    if len(y) == 0:
        return float("nan")
    templates = _orbit_templates(dataset.d, dataset.k, dataset.prototypes)
    predictions = []
    for signal in x_complex:
        normalized = signal / max(float(np.linalg.norm(signal)), 1e-12)
        scores = [float(np.max(np.abs(class_templates @ normalized.conj()))) for class_templates in templates]
        predictions.append(int(np.argmax(scores)))
    return float(np.mean(np.asarray(predictions, dtype=int) == y))
