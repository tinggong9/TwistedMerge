"""Prototype TwistedMerge algorithm.

This module implements a small, auditable version of the algorithm described in
the paper plan.  It works on NumPy vector classifiers and matrix-valued
alignments, and includes a concrete mu_2 doubled representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Mapping, Sequence

import numpy as np

from .simplicial_mu2 import Face, LinearLocalModel, canonical_face


IndexPair = tuple[int, int]
Triple = tuple[int, int, int]


@dataclass(frozen=True)
class TwistedMergeConfig:
    rank_lift_q: int = 2
    tolerance: float = 1e-6
    central_tolerance: float = 1e-5
    reference_index: int = 0


@dataclass(frozen=True)
class GaugeSynchronizationResult:
    success: bool
    gauges: dict[int, np.ndarray]
    residual: float
    max_residual: float


@dataclass(frozen=True)
class VectorModel:
    weight: np.ndarray
    name: str = "vector"

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.logits(x) >= 0.0).astype(np.int64)


@dataclass(frozen=True)
class TwistedVectorModel:
    """Rank-lifted vector model with explicit mu_2 branch action."""

    branch_weights: np.ndarray
    central_matrices: dict[int, np.ndarray]
    context_twist: Mapping[Face, int] | None = None

    def branch_for_context(self, context: Face | None) -> int:
        if context is None or self.context_twist is None:
            return 0
        sign = int(self.context_twist[canonical_face(context)])
        return 0 if sign > 0 else 1

    def logits(self, x: np.ndarray, context: Face | None = None) -> np.ndarray:
        return x @ self.branch_weights[self.branch_for_context(context)]

    def predict(self, x: np.ndarray, context: Face | None = None) -> np.ndarray:
        return (self.logits(x, context) >= 0.0).astype(np.int64)


@dataclass(frozen=True)
class TwistedMergeResult:
    status: str
    ordinary_model: VectorModel
    cycle_consistent_model: VectorModel
    twisted_model: TwistedVectorModel | None
    ensemble_models: list[VectorModel]
    defects: dict[Triple, np.ndarray]
    defect_signs: dict[Triple, int]
    cycle_score: float
    twist_residual: float | None
    gauge: GaugeSynchronizationResult
    lifted_transition_maps: dict[IndexPair, np.ndarray]
    notes: list[str]


def _as_weight(model: np.ndarray | LinearLocalModel | VectorModel) -> np.ndarray:
    if isinstance(model, np.ndarray):
        return np.asarray(model, dtype=float)
    if isinstance(model, LinearLocalModel):
        return np.asarray(model.weight, dtype=float)
    if isinstance(model, VectorModel):
        return np.asarray(model.weight, dtype=float)
    if hasattr(model, "weight"):
        return np.asarray(model.weight, dtype=float)
    raise TypeError(f"unsupported local model type: {type(model)!r}")


def local_weights(local_models: Sequence[np.ndarray | LinearLocalModel | VectorModel]) -> np.ndarray:
    weights = np.stack([_as_weight(model) for model in local_models], axis=0)
    if weights.ndim != 2:
        raise ValueError("TwistedMerge vector prototype expects a list of 1D weights.")
    return weights


def complete_alignment_maps(
    alignments: Mapping[IndexPair, np.ndarray],
    n_models: int,
    dim: int,
) -> dict[IndexPair, np.ndarray]:
    completed: dict[IndexPair, np.ndarray] = {}
    eye = np.eye(dim)
    for i in range(n_models):
        completed[(i, i)] = eye
    for (i, j), matrix in alignments.items():
        completed[(i, j)] = np.asarray(matrix, dtype=float)
        if (j, i) not in completed:
            completed[(j, i)] = np.linalg.pinv(completed[(i, j)])
    missing = [(i, j) for i, j in product(range(n_models), repeat=2) if (i, j) not in completed]
    if missing:
        raise ValueError(f"missing alignment maps for pairs: {missing[:5]}")
    return completed


def estimate_pairwise_alignment_maps(
    local_models: Sequence[np.ndarray | LinearLocalModel | VectorModel],
    pairwise_alignments: Mapping[IndexPair, np.ndarray] | None = None,
) -> dict[IndexPair, np.ndarray]:
    """Estimate pairwise maps when none are supplied.

    The fallback estimator is intentionally simple and only detects a global
    mu_2 sign between vector classifiers.  Image-model permutation estimation is
    implemented separately in `src/model_merging_benchmark.py`.
    """
    weights = local_weights(local_models)
    n_models, dim = weights.shape
    if pairwise_alignments is not None:
        return complete_alignment_maps(pairwise_alignments, n_models, dim)
    alignments: dict[IndexPair, np.ndarray] = {}
    for i, j in product(range(n_models), repeat=2):
        if i == j:
            sign = 1.0
        else:
            sign = 1.0 if float(weights[i] @ weights[j]) >= 0.0 else -1.0
        alignments[(i, j)] = sign * np.eye(dim)
    return alignments


def compute_triangle_defects(
    g: Mapping[IndexPair, np.ndarray],
    triples: Sequence[Triple] | None = None,
) -> dict[Triple, np.ndarray]:
    if triples is None:
        n_models = max(max(i, j) for i, j in g) + 1
        triples = list(combinations(range(n_models), 3))
    return {
        tuple(triple): g[(triple[0], triple[1])] @ g[(triple[1], triple[2])] @ g[(triple[2], triple[0])]
        for triple in triples
    }


def central_sign(matrix: np.ndarray) -> int:
    return 1 if float(np.trace(matrix)) >= 0.0 else -1


def defect_signs(defects: Mapping[Triple, np.ndarray]) -> dict[Triple, int]:
    return {triple: central_sign(matrix) for triple, matrix in defects.items()}


def cycle_obstruction_score(defects: Mapping[Triple, np.ndarray]) -> float:
    values = []
    for matrix in defects.values():
        eye = np.eye(matrix.shape[0])
        values.append(np.linalg.norm(matrix - eye, ord="fro") / max(np.linalg.norm(eye, ord="fro"), 1e-12))
    return float(np.mean(values)) if values else 0.0


def twist_residual_score(
    defects: Mapping[Triple, np.ndarray],
    alpha: Mapping[Triple, int] | Mapping[Face, int] | None,
) -> float | None:
    if alpha is None:
        return None
    values = []
    for triple, matrix in defects.items():
        sign = lookup_twist(alpha, triple)
        target = sign * np.eye(matrix.shape[0])
        values.append(np.linalg.norm(matrix - target, ord="fro") / max(np.linalg.norm(target, ord="fro"), 1e-12))
    return float(np.mean(values)) if values else 0.0


def lookup_twist(alpha: Mapping[Triple, int] | Mapping[Face, int], triple: Triple) -> int:
    if triple in alpha:
        return int(alpha[triple])  # type: ignore[index]
    face = canonical_face(triple)
    if face in alpha:
        return int(alpha[face])  # type: ignore[index]
    raise KeyError(f"twist is missing face/triple {triple}")


def try_global_gauge_synchronization(
    g: Mapping[IndexPair, np.ndarray],
    tolerance: float = 1e-6,
    reference_index: int = 0,
) -> GaugeSynchronizationResult:
    n_models = max(max(i, j) for i, j in g) + 1
    dim = next(iter(g.values())).shape[0]
    gauges: dict[int, np.ndarray] = {}
    for i in range(n_models):
        if i == reference_index:
            gauges[i] = np.eye(dim)
        else:
            gauges[i] = g[(reference_index, i)]
    residuals = []
    for i, j in product(range(n_models), repeat=2):
        predicted = gauges[j] @ np.linalg.pinv(gauges[i])
        denom = max(np.linalg.norm(g[(i, j)], ord="fro"), 1e-12)
        residuals.append(float(np.linalg.norm(g[(i, j)] - predicted, ord="fro") / denom))
    residual = float(np.mean(residuals)) if residuals else 0.0
    max_residual = float(np.max(residuals)) if residuals else 0.0
    return GaugeSynchronizationResult(
        success=max_residual <= tolerance,
        gauges=gauges,
        residual=residual,
        max_residual=max_residual,
    )


def align_weights_to_reference(weights: np.ndarray, gauges: Mapping[int, np.ndarray]) -> np.ndarray:
    aligned = []
    for i, weight in enumerate(weights):
        aligned.append(np.linalg.pinv(gauges[i]) @ weight)
    return np.stack(aligned, axis=0)


def ordinary_merge(weights: np.ndarray) -> VectorModel:
    return VectorModel(weight=weights.mean(axis=0), name="ordinary_merge")


def cycle_consistent_merge(weights: np.ndarray, gauge: GaugeSynchronizationResult) -> VectorModel:
    aligned = align_weights_to_reference(weights, gauge.gauges)
    return VectorModel(weight=aligned.mean(axis=0), name="cycle_consistent_merge")


def mu2_doubled_central_matrices() -> dict[int, np.ndarray]:
    """Concrete 2x2 mu_2 action for a doubled branch representation.

    The basis stores the + and - sign branches.  The nontrivial central element
    swaps the two branches, so applying it changes which classifier is active.
    """
    return {
        1: np.eye(2),
        -1: np.array([[0.0, 1.0], [1.0, 0.0]]),
    }


def lift_mu2_transition(base_alignment: np.ndarray, central_sign_value: int) -> np.ndarray:
    central = mu2_doubled_central_matrices()[1 if central_sign_value >= 0 else -1]
    return np.kron(central, base_alignment)


def build_mu2_rank_lifted_model(
    ordinary: VectorModel,
    alpha: Mapping[Triple, int] | Mapping[Face, int] | None,
    rank_lift_q: int,
) -> TwistedVectorModel | None:
    if rank_lift_q < 2 or alpha is None:
        return None
    base = ordinary.weight
    branches = np.stack([base, -base], axis=0)
    if rank_lift_q > 2:
        extra = np.repeat(base.reshape(1, -1), rank_lift_q - 2, axis=0)
        branches = np.concatenate([branches, extra], axis=0)
    return TwistedVectorModel(
        branch_weights=branches,
        central_matrices=mu2_doubled_central_matrices(),
        context_twist={canonical_face(face): int(sign) for face, sign in alpha.items()},
    )


def evaluate_vector_model(
    model: VectorModel | TwistedVectorModel,
    datasets: Mapping[Face | None, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    losses = []
    accuracies = []
    for context, (x, y) in datasets.items():
        if isinstance(model, TwistedVectorModel):
            pred = model.predict(x, context)
        else:
            pred = model.predict(x)
        losses.append(float(np.mean(pred != y)))
        accuracies.append(float(np.mean(pred == y)))
    return {
        "zero_one_loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
    }


def evaluate_ensemble(
    models: Sequence[VectorModel],
    datasets: Mapping[Face | None, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    losses = []
    accuracies = []
    for _context, (x, y) in datasets.items():
        logits = np.stack([model.logits(x) for model in models], axis=0).mean(axis=0)
        pred = (logits >= 0.0).astype(np.int64)
        losses.append(float(np.mean(pred != y)))
        accuracies.append(float(np.mean(pred == y)))
    return {
        "zero_one_loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
    }


class TwistedMerge:
    """Prototype cocycle-aware merge algorithm for vector classifiers."""

    def __init__(self, config: TwistedMergeConfig | None = None):
        self.config = config or TwistedMergeConfig()

    def run(
        self,
        local_models_input: Sequence[np.ndarray | LinearLocalModel | VectorModel],
        pairwise_alignments: Mapping[IndexPair, np.ndarray] | None = None,
        alpha: Mapping[Triple, int] | Mapping[Face, int] | None = None,
        triples: Sequence[Triple] | None = None,
    ) -> TwistedMergeResult:
        weights = local_weights(local_models_input)
        g = estimate_pairwise_alignment_maps(local_models_input, pairwise_alignments)
        defects = compute_triangle_defects(g, triples)
        signs = defect_signs(defects)
        cycle_score = cycle_obstruction_score(defects)
        residual_to_twist = twist_residual_score(defects, alpha)
        gauge = try_global_gauge_synchronization(
            g,
            tolerance=self.config.tolerance,
            reference_index=self.config.reference_index,
        )
        ordinary = ordinary_merge(weights)
        cycle_model = cycle_consistent_merge(weights, gauge)
        notes: list[str] = []
        status = "ordinary"
        twisted_model: TwistedVectorModel | None = None
        if gauge.success:
            notes.append("Pairwise alignments gauge-trivialized within tolerance.")
        elif alpha is not None and residual_to_twist is not None and residual_to_twist <= self.config.central_tolerance:
            twisted_model = build_mu2_rank_lifted_model(ordinary, alpha, self.config.rank_lift_q)
            if twisted_model is not None:
                status = "twisted_rank_lifted"
                notes.append("Defects match the supplied finite central twist; built doubled mu_2 representation.")
            else:
                status = "failed"
                notes.append("Defects match a twist, but rank_lift_q < 2 so no doubled representation was built.")
        else:
            status = "failed"
            notes.append("Gauge trivialization failed and defects were not close to a supplied central twist.")
        lifted_maps: dict[IndexPair, np.ndarray] = {}
        if twisted_model is not None and alpha is not None:
            for pair, matrix in g.items():
                lifted_maps[pair] = lift_mu2_transition(matrix, 1)
        ensemble = [VectorModel(weight=weight, name=f"local_{i}") for i, weight in enumerate(weights)]
        return TwistedMergeResult(
            status=status,
            ordinary_model=ordinary,
            cycle_consistent_model=cycle_model,
            twisted_model=twisted_model,
            ensemble_models=ensemble,
            defects=defects,
            defect_signs=signs,
            cycle_score=cycle_score,
            twist_residual=residual_to_twist,
            gauge=gauge,
            lifted_transition_maps=lifted_maps,
            notes=notes,
        )

    def evaluate(
        self,
        result: TwistedMergeResult,
        datasets: Mapping[Face | None, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, dict[str, float]]:
        metrics = {
            "ordinary_merge": evaluate_vector_model(result.ordinary_model, datasets),
            "cycle_consistent_merge": evaluate_vector_model(result.cycle_consistent_model, datasets),
            "ensemble": evaluate_ensemble(result.ensemble_models, datasets),
        }
        if result.twisted_model is not None:
            metrics["twisted_merge"] = evaluate_vector_model(result.twisted_model, datasets)
        return metrics


def finite_central_twist_close(
    defects: Mapping[Triple, np.ndarray],
    alpha: Mapping[Triple, int] | Mapping[Face, int],
    tolerance: float = 1e-5,
) -> bool:
    residual = twist_residual_score(defects, alpha)
    return residual is not None and residual <= tolerance


def pseudocode() -> str:
    return """TwistedMerge(M_i, g_ij=None, alpha_ijk=None, q=2):
  1. If g_ij is absent, estimate pairwise alignments from local models.
  2. Compute c_ijk = g_ij g_jk g_ki on each triangle.
  3. Try to find gauges h_i with g_ij ~= h_j h_i^{-1}.
  4. If the residual is small, align all M_i by h_i^{-1} and average.
  5. Otherwise compare c_ijk with the supplied finite central twist alpha_ijk.
  6. If c_ijk ~= alpha_ijk I and q >= 2 for mu_2, form a doubled branch
     representation with branches (w, -w) and central action rho(-1) =
     [[0, 1], [1, 0]].
  7. Evaluate ordinary merge, cycle-consistent merge, twisted merge, and
     ensemble; report the cycle score and twist residual."""
