"""Controlled neural overlap benchmark with exact central twists.

The construction uses one-hidden-layer ReLU MLPs whose hidden units occur in
paired ``ReLU(z), ReLU(-z)`` blocks.  The nontrivial ``mu_2`` element is the
exact permutation that swaps every pair.  This gives a small neural benchmark
whose local models are exact and accurate, while the overlap/triangle data can
carry prescribed central or noncentral defects.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .simplicial_mu2 import Face, canonical_face, coboundary_witness_mu2, is_coboundary_mu2, tetrahedral_sphere


Edge = tuple[int, int]
Perm = np.ndarray


METHODS = (
    "ordinary_weight_average",
    "git_rebasin_pairwise",
    "c2m3_synchronized",
    "twisted_q2_branch",
    "random_branch_ensemble",
    "validation_selected_branch_ensemble",
    "c2m3_cluster_branch_ensemble",
    "ensemble_upper_bound",
)

EXTRA_CONTROL_ALIASES = {
    "wrong_twist": "wrong_twist_control",
    "wrong_twist_control": "wrong_twist_control",
    "wrong_context": "wrong_context_control",
    "wrong_context_control": "wrong_context_control",
    "learned_router": "learned_context_router",
    "learned_context_router": "learned_context_router",
    "distilled_single": "distilled_twisted_single_model",
    "distilled_twisted_single_model": "distilled_twisted_single_model",
    "parameter_matched_wide": "parameter_matched_wide_control",
    "parameter_matched_wide_control": "parameter_matched_wide_control",
    "no_twist_branch": "no_twist_branch_control",
    "no_twist_branch_control": "no_twist_branch_control",
}

EXTRA_CONTROL_METHODS = tuple(dict.fromkeys(EXTRA_CONTROL_ALIASES.values()))


@dataclass(frozen=True)
class ControlledMLP:
    """A tiny one-hidden-layer ReLU MLP stored as NumPy arrays."""

    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: float = 0.0
    model_index: int = -1
    hidden_permutation: np.ndarray | None = None

    @property
    def width(self) -> int:
        return int(self.hidden_weight.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.hidden_weight.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(self.hidden_weight.size + self.hidden_bias.size + self.output_weight.size + 1)

    def hidden_features(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x @ self.hidden_weight.T + self.hidden_bias, 0.0)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.hidden_features(x) @ self.output_weight + self.output_bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.logits(x) >= 0.0).astype(np.int64)


@dataclass(frozen=True)
class ControlledCase:
    family: str
    seed: int
    width: int
    n_models: int
    branch_count: int
    base_model: ControlledMLP
    local_models: list[ControlledMLP]
    local_hidden_permutations: dict[int, np.ndarray]
    alpha_signs: dict[Face, int]
    target_signs: dict[Face, int]
    true_edge_gauges: dict[Face, dict[Edge, np.ndarray]]
    observed_edge_gauges: dict[Face, dict[Edge, np.ndarray]]
    true_twist_permutations: dict[Face, np.ndarray]
    overlap_ids: dict[Face, str]
    chart_data: dict[int, tuple[np.ndarray, np.ndarray]]
    val_face_data: dict[Face, tuple[np.ndarray, np.ndarray]]
    test_face_data: dict[Face, tuple[np.ndarray, np.ndarray]]
    central_twist_claim_allowed: bool
    is_coboundary: bool | None
    notes: str


def canonical_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def identity_perm(width: int) -> np.ndarray:
    return np.arange(width, dtype=int)


def central_swap_perm(width: int) -> np.ndarray:
    if width % 2 != 0:
        raise ValueError("controlled overlap widths must be even")
    perm = np.arange(width, dtype=int)
    perm[0::2] = np.arange(1, width, 2)
    perm[1::2] = np.arange(0, width, 2)
    return perm


def noncentral_perm(width: int) -> np.ndarray:
    perm = np.arange(width, dtype=int)
    if width < 4:
        perm = np.roll(perm, 1)
    else:
        perm[:4] = np.array([1, 2, 3, 0])
    return perm


def invert_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def compose_perm(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the permutation corresponding to applying first, then second."""

    return second[first]


def perm_matrix(perm: np.ndarray) -> np.ndarray:
    mat = np.zeros((len(perm), len(perm)), dtype=float)
    mat[np.arange(len(perm)), perm] = 1.0
    return mat


def perm_residual(observed: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.asarray(observed, dtype=int) != np.asarray(target, dtype=int)))


def make_base_mlp(width: int, rng: np.random.Generator) -> ControlledMLP:
    if width % 2 != 0:
        raise ValueError("width must be even because hidden units are ReLU sign pairs")
    input_dim = width // 2
    hidden = np.zeros((width, input_dim), dtype=float)
    out = np.zeros(width, dtype=float)
    score_weight = rng.normal(size=input_dim)
    score_weight /= max(float(np.linalg.norm(score_weight)), 1e-12)
    for idx, coeff in enumerate(score_weight):
        hidden[2 * idx, idx] = 1.0
        hidden[2 * idx + 1, idx] = -1.0
        out[2 * idx] = coeff
        out[2 * idx + 1] = -coeff
    return ControlledMLP(
        hidden_weight=hidden,
        hidden_bias=np.zeros(width, dtype=float),
        output_weight=out,
        output_bias=0.0,
        model_index=-1,
        hidden_permutation=identity_perm(width),
    )


def permute_hidden(model: ControlledMLP, perm: np.ndarray, model_index: int) -> ControlledMLP:
    return ControlledMLP(
        hidden_weight=model.hidden_weight[perm].copy(),
        hidden_bias=model.hidden_bias[perm].copy(),
        output_weight=model.output_weight[perm].copy(),
        output_bias=float(model.output_bias),
        model_index=model_index,
        hidden_permutation=perm.copy(),
    )


def align_hidden_to_base(model: ControlledMLP) -> ControlledMLP:
    if model.hidden_permutation is None:
        return model
    inv = invert_perm(model.hidden_permutation)
    return ControlledMLP(
        hidden_weight=model.hidden_weight[inv].copy(),
        hidden_bias=model.hidden_bias[inv].copy(),
        output_weight=model.output_weight[inv].copy(),
        output_bias=float(model.output_bias),
        model_index=model.model_index,
        hidden_permutation=identity_perm(model.width),
    )


def average_models(models: list[ControlledMLP], model_index: int = -1) -> ControlledMLP:
    if not models:
        raise ValueError("cannot average an empty model list")
    return ControlledMLP(
        hidden_weight=np.stack([model.hidden_weight for model in models]).mean(axis=0),
        hidden_bias=np.stack([model.hidden_bias for model in models]).mean(axis=0),
        output_weight=np.stack([model.output_weight for model in models]).mean(axis=0),
        output_bias=float(np.mean([model.output_bias for model in models])),
        model_index=model_index,
        hidden_permutation=None,
    )


def local_vertex_permutations(width: int, n_models: int) -> dict[int, np.ndarray]:
    swap = central_swap_perm(width)
    return {idx: (identity_perm(width) if idx % 2 == 0 else swap.copy()) for idx in range(n_models)}


def make_alpha_signs(family: str) -> dict[Face, int]:
    faces = tuple(canonical_face(face) for face in tetrahedral_sphere().faces)
    if family == "mu2_coboundary":
        signs = {face: 1 for face in faces}
        signs[(0, 1, 2)] = -1
        signs[(0, 1, 3)] = -1
        if not is_coboundary_mu2(signs, tetrahedral_sphere()):
            raise AssertionError("internal coboundary alpha is not a coboundary")
        return signs
    if family == "mu2_nontrivial_h2":
        signs = {face: 1 for face in faces}
        signs[(0, 1, 2)] = -1
        if is_coboundary_mu2(signs, tetrahedral_sphere()):
            raise AssertionError("internal H2 alpha unexpectedly became coboundary")
        return signs
    if family == "random_noncentral":
        return {face: 0 for face in faces}
    raise ValueError(f"unknown twist family: {family}")


def target_signs_for_family(family: str, alpha_signs: Mapping[Face, int]) -> dict[Face, int]:
    if family == "mu2_nontrivial_h2":
        return {face: int(sign) for face, sign in alpha_signs.items()}
    return {face: 1 for face in alpha_signs}


def coboundary_edge_signs(alpha_signs: Mapping[Face, int]) -> dict[Edge, int] | None:
    witness = coboundary_witness_mu2(alpha_signs, tetrahedral_sphere())
    if witness is None:
        return None
    return {canonical_edge(*edge): int(sign) for edge, sign in witness.items()}


def edge_sign_to_perm(sign: int, width: int) -> np.ndarray:
    return identity_perm(width) if int(sign) >= 0 else central_swap_perm(width)


def make_face_gauges(family: str, width: int, alpha_signs: Mapping[Face, int]) -> tuple[dict[Face, dict[Edge, np.ndarray]], dict[Face, np.ndarray]]:
    gauges: dict[Face, dict[Edge, np.ndarray]] = {}
    true_twists: dict[Face, np.ndarray] = {}
    identity = identity_perm(width)
    central = central_swap_perm(width)
    noncentral = noncentral_perm(width)
    if family == "mu2_coboundary":
        edge_signs = coboundary_edge_signs(alpha_signs)
        if edge_signs is None:
            raise ValueError("mu2_coboundary family requires coboundary signs")
        for face in alpha_signs:
            i, j, k = face
            gauges[face] = {
                (i, j): edge_sign_to_perm(edge_signs[canonical_edge(i, j)], width),
                (j, k): edge_sign_to_perm(edge_signs[canonical_edge(j, k)], width),
                (k, i): edge_sign_to_perm(edge_signs[canonical_edge(k, i)], width),
            }
            true_twists[face] = edge_sign_to_perm(alpha_signs[face], width)
        return gauges, true_twists
    if family == "mu2_nontrivial_h2":
        for face, sign in alpha_signs.items():
            i, j, k = face
            gauges[face] = {
                (i, j): identity.copy(),
                (j, k): identity.copy(),
                (k, i): identity.copy() if sign >= 0 else central.copy(),
            }
            true_twists[face] = identity.copy() if sign >= 0 else central.copy()
        return gauges, true_twists
    if family == "random_noncentral":
        for face in alpha_signs:
            i, j, k = face
            gauges[face] = {
                (i, j): identity.copy(),
                (j, k): identity.copy(),
                (k, i): noncentral.copy(),
            }
            true_twists[face] = noncentral.copy()
        return gauges, true_twists
    raise ValueError(family)


def triangle_defect_perm(face_gauge: Mapping[Edge, np.ndarray], face: Face) -> np.ndarray:
    i, j, k = face
    return compose_perm(compose_perm(face_gauge[(i, j)], face_gauge[(j, k)]), face_gauge[(k, i)])


def defect_rows_for_case(case: ControlledCase) -> list[dict[str, object]]:
    rows = []
    identity = identity_perm(case.width)
    central = central_swap_perm(case.width)
    for face in sorted(case.observed_edge_gauges):
        observed = triangle_defect_perm(case.observed_edge_gauges[face], face)
        target = case.true_twist_permutations[face]
        centrality_residual = min(perm_residual(observed, identity), perm_residual(observed, central))
        if perm_residual(observed, identity) == 0.0:
            observed_sign = 1
            defect_type = "central_identity"
        elif perm_residual(observed, central) == 0.0:
            observed_sign = -1
            defect_type = "central_mu2"
        else:
            observed_sign = 0
            defect_type = "noncentral_permutation"
        rows.append(
            {
                "face": "-".join(map(str, face)),
                "i": face[0],
                "j": face[1],
                "k": face[2],
                "true_alpha_sign": int(case.alpha_signs[face]),
                "target_sign": int(case.target_signs[face]),
                "observed_triangle_sign": observed_sign,
                "defect_type": defect_type,
                "centrality_residual": centrality_residual,
                "defect_to_true_twist_residual": perm_residual(observed, target),
                "observed_triangle_perm": observed.astype(int).tolist(),
                "true_twist_perm": target.astype(int).tolist(),
            }
        )
    return rows


def generate_dataset(
    model: ControlledMLP,
    rng: np.random.Generator,
    n_samples: int,
    sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n_samples, model.input_dim))
    logits = sign * model.logits(x)
    y = (logits >= 0.0).astype(np.int64)
    return x, y


def build_controlled_case(
    family: str,
    width: int,
    n_models: int,
    seed: int,
    samples_per_chart: int,
    samples_per_overlap: int,
    branch_count: int = 2,
) -> ControlledCase:
    if n_models != 4:
        raise ValueError("the controlled tetrahedral benchmark currently uses n_models=4")
    family_offset = {"mu2_coboundary": 17, "mu2_nontrivial_h2": 37, "random_noncentral": 59}[family]
    rng = np.random.default_rng(seed + 1009 * width + 9176 * family_offset)
    base = make_base_mlp(width, rng)
    vertex_perms = local_vertex_permutations(width, n_models)
    local_models = [permute_hidden(base, vertex_perms[idx], idx) for idx in range(n_models)]
    alpha = make_alpha_signs(family)
    targets = target_signs_for_family(family, alpha)
    gauges, true_twists = make_face_gauges(family, width, alpha)
    chart_data = {
        idx: generate_dataset(base, np.random.default_rng(seed + 3001 * (idx + 1) + width), samples_per_chart, sign=1)
        for idx in range(n_models)
    }
    val_face_data = {
        face: generate_dataset(base, np.random.default_rng(seed + 7001 * (face[0] + 1) + 97 * face[1] + width), samples_per_overlap // 2, targets[face])
        for face in alpha
    }
    test_face_data = {
        face: generate_dataset(base, np.random.default_rng(seed + 9001 * (face[2] + 1) + 193 * face[1] + width), samples_per_overlap, targets[face])
        for face in alpha
    }
    is_cob = None if family == "random_noncentral" else is_coboundary_mu2(alpha, tetrahedral_sphere())
    notes = {
        "mu2_coboundary": "Nontrivial central face signs are an edge coboundary; C2M3 can absorb them as a single model.",
        "mu2_nontrivial_h2": "The central face signs represent the nonzero tetrahedral H2(mu2) class; only context-aware q=2 branch prediction is claimed.",
        "random_noncentral": "Triangle defects are noncentral permutations and are a negative control, not central-twist evidence.",
    }[family]
    return ControlledCase(
        family=family,
        seed=seed,
        width=width,
        n_models=n_models,
        branch_count=branch_count,
        base_model=base,
        local_models=local_models,
        local_hidden_permutations=vertex_perms,
        alpha_signs=alpha,
        target_signs=targets,
        true_edge_gauges=gauges,
        observed_edge_gauges={face: {edge: perm.copy() for edge, perm in face_gauges.items()} for face, face_gauges in gauges.items()},
        true_twist_permutations=true_twists,
        overlap_ids={face: f"{family}_seed{seed}_width{width}_face{'-'.join(map(str, face))}" for face in alpha},
        chart_data=chart_data,
        val_face_data=val_face_data,
        test_face_data=test_face_data,
        central_twist_claim_allowed=family in {"mu2_coboundary", "mu2_nontrivial_h2"},
        is_coboundary=is_cob,
        notes=notes,
    )


def binary_loss_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    signed = (2 * y.astype(float) - 1.0) * logits
    return float(np.mean(np.logaddexp(0.0, -signed)))


def evaluate_face_predictor(
    datasets: Mapping[Face, tuple[np.ndarray, np.ndarray]],
    predict_logits: Callable[[Face, np.ndarray], np.ndarray],
) -> dict[str, float]:
    losses = []
    accuracies = []
    for face, (x, y) in datasets.items():
        logits = predict_logits(face, x)
        pred = (logits >= 0.0).astype(np.int64)
        losses.append(binary_loss_from_logits(logits, y))
        accuracies.append(float(np.mean(pred == y)))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
    }


def evaluate_local_models(case: ControlledCase) -> dict[str, float]:
    accuracies = []
    losses = []
    for idx, model in enumerate(case.local_models):
        x, y = case.chart_data[idx]
        logits = model.logits(x)
        accuracies.append(float(np.mean((logits >= 0.0).astype(np.int64) == y)))
        losses.append(binary_loss_from_logits(logits, y))
    return {"accuracy": float(np.mean(accuracies)), "loss": float(np.mean(losses))}


def branch_sign_from_assignment(branch: int) -> int:
    return 1 if int(branch) == 0 else -1


def global_validation_branch_assignment(case: ControlledCase, base_model: ControlledMLP) -> dict[Face, int]:
    scores = {}
    for branch in (0, 1):
        sign = branch_sign_from_assignment(branch)
        metrics = evaluate_face_predictor(case.val_face_data, lambda _face, x, sign=sign: sign * base_model.logits(x))
        scores[branch] = metrics["accuracy"]
    best = max(scores, key=lambda branch: (scores[branch], -branch))
    return {face: int(best) for face in case.alpha_signs}


def random_branch_assignment(case: ControlledCase, rng: np.random.Generator) -> dict[Face, int]:
    branch = int(rng.integers(0, case.branch_count))
    return {face: branch for face in case.alpha_signs}


def twisted_branch_assignment(case: ControlledCase) -> dict[Face, int]:
    if case.family == "mu2_nontrivial_h2":
        return {face: (0 if sign >= 0 else 1) for face, sign in case.alpha_signs.items()}
    return {face: 0 for face in case.alpha_signs}


def no_twist_branch_assignment(case: ControlledCase) -> dict[Face, int]:
    return {face: 0 for face in case.alpha_signs}


def wrong_twist_assignment(case: ControlledCase) -> dict[Face, int]:
    faces = sorted(case.alpha_signs)
    correct = twisted_branch_assignment(case)
    if case.family == "mu2_nontrivial_h2":
        negative_faces = [face for face in faces if correct[face] == 1]
        if not negative_faces:
            return {face: 1 - branch for face, branch in correct.items()}
        true_negative = negative_faces[0]
        wrong_negative = faces[(faces.index(true_negative) + 1) % len(faces)]
        return {face: (1 if face == wrong_negative else 0) for face in faces}
    return {face: 1 - int(correct[face]) for face in faces}


def wrong_context_assignment(case: ControlledCase, rng: np.random.Generator) -> dict[Face, int]:
    faces = sorted(case.alpha_signs)
    correct = twisted_branch_assignment(case)
    permuted = faces.copy()
    for _ in range(16):
        rng.shuffle(permuted)
        assignment = {face: int(correct[source]) for face, source in zip(faces, permuted)}
        if assignment != correct:
            return assignment
    return {face: int(correct[faces[(idx + 1) % len(faces)]]) for idx, face in enumerate(faces)}


def learned_context_router_assignment(
    case: ControlledCase,
    base_model: ControlledMLP,
) -> tuple[dict[Face, int], dict[str, object]]:
    assignment: dict[Face, int] = {}
    branch_scores: dict[str, dict[str, float]] = {}
    correct = 0
    total = 0
    for face, (x, y) in case.val_face_data.items():
        scores = {}
        for branch in range(case.branch_count):
            sign = branch_sign_from_assignment(branch)
            logits = sign * base_model.logits(x)
            scores[branch] = float(np.mean((logits >= 0.0).astype(np.int64) == y))
        best = max(scores, key=lambda branch: (scores[branch], -branch))
        assignment[face] = int(best)
        branch_scores["-".join(map(str, face))] = {str(branch): float(score) for branch, score in scores.items()}
        correct += int(round(scores[best] * len(y)))
        total += int(len(y))
    return assignment, {
        "router_type": "validation_face_table",
        "router_train_accuracy": float(correct / max(total, 1)),
        "router_branch_scores": branch_scores,
    }


def distilled_twisted_scale(case: ControlledCase, base_model: ControlledMLP) -> tuple[float, dict[str, object]]:
    assignment = twisted_branch_assignment(case)
    numerator = 0.0
    denominator = 0.0
    teacher_losses = []
    for face, (x, _y) in case.val_face_data.items():
        z = base_model.logits(x)
        teacher = branch_sign_from_assignment(assignment[face]) * z
        numerator += float(np.sum(z * teacher))
        denominator += float(np.sum(z * z))
        teacher_losses.append(float(np.mean((teacher - z) ** 2)))
    scale = numerator / max(denominator, 1e-12)
    return float(scale), {
        "distillation_target": "twisted_q2_branch_logits",
        "distillation_scale": float(scale),
        "distillation_teacher_mse_against_base": float(np.mean(teacher_losses)) if teacher_losses else float("nan"),
    }


def make_parameter_matched_wide_model(base_model: ControlledMLP, target_parameter_count: int) -> ControlledMLP:
    input_dim = base_model.input_dim
    width = base_model.width
    while width * input_dim + 2 * width + 1 < target_parameter_count:
        width += 2
    repeats = int(np.ceil(width / base_model.width))
    hidden = np.tile(base_model.hidden_weight, (repeats, 1))[:width].copy()
    bias = np.tile(base_model.hidden_bias, repeats)[:width].copy()
    out = np.tile(base_model.output_weight / repeats, repeats)[:width].copy()
    return ControlledMLP(
        hidden_weight=hidden,
        hidden_bias=bias,
        output_weight=out,
        output_bias=float(base_model.output_bias),
        model_index=-2,
        hidden_permutation=identity_perm(width),
    )


def evaluate_branch_assignment(
    case: ControlledCase,
    base_model: ControlledMLP,
    datasets: Mapping[Face, tuple[np.ndarray, np.ndarray]],
    assignment: Mapping[Face, int],
) -> dict[str, float]:
    return evaluate_face_predictor(
        datasets,
        lambda face, x: branch_sign_from_assignment(assignment[canonical_face(face)]) * base_model.logits(x),
    )


def method_capacity_metadata(
    method: str,
    case: ControlledCase,
    base_model: ControlledMLP,
    *,
    parameter_count: int | None = None,
) -> dict[str, object]:
    base_params = base_model.parameter_count
    if method in {
        "ordinary_weight_average",
        "git_rebasin_pairwise",
        "c2m3_synchronized",
        "distilled_twisted_single_model",
    }:
        branch_count = 1
        params = int(parameter_count or base_params)
        param_mult = float(params / max(base_params, 1))
        infer_mult = 1.0
        single = True
        branch_model = False
    elif method == "parameter_matched_wide_control":
        branch_count = 1
        params = int(parameter_count or round(base_params * case.branch_count))
        param_mult = float(params / max(base_params, 1))
        infer_mult = 1.0
        single = True
        branch_model = False
    elif method == "ensemble_upper_bound":
        branch_count = case.n_models
        params = int(round(base_params * case.n_models))
        param_mult = float(params / max(base_params, 1))
        infer_mult = float(case.n_models)
        single = False
        branch_model = False
    else:
        branch_count = case.branch_count
        params = int(parameter_count or round(base_params * case.branch_count))
        param_mult = float(params / max(base_params, 1))
        infer_mult = float(case.branch_count)
        single = False
        branch_model = True
    return {
        "parameter_count": int(params),
        "branch_count": int(branch_count),
        "parameter_multiplier": float(param_mult),
        "inference_time_multiplier": float(infer_mult),
        "is_single_model": bool(single),
        "is_branch_model": bool(branch_model),
        "capacity_matched_to_weight_average": bool(param_mult == 1.0),
        "capacity_matched_to_rank_lift": bool(
            method != "ensemble_upper_bound"
            and np.isclose(param_mult, float(case.branch_count), rtol=0.05, atol=0.05)
        ),
        "uses_validation_data": method in {
            "validation_selected_branch_ensemble",
            "learned_context_router",
            "distilled_twisted_single_model",
        },
        "uses_obstruction_residual": method == "twisted_q2_branch",
        "uses_triangle_context": method in {
            "twisted_q2_branch",
            "wrong_twist_control",
            "wrong_context_control",
            "learned_context_router",
        },
        "exact_controlled_relu_symmetry": method in {
            "git_rebasin_pairwise",
            "c2m3_synchronized",
            "twisted_q2_branch",
            "wrong_twist_control",
            "wrong_context_control",
            "learned_context_router",
            "no_twist_branch_control",
        },
        "uses_distillation": method == "distilled_twisted_single_model",
        "uses_wrong_twist": method == "wrong_twist_control",
        "uses_wrong_context": method == "wrong_context_control",
    }


def normalize_extra_controls(extra_controls: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not extra_controls:
        return ()
    normalized = []
    for control in extra_controls:
        if not control:
            continue
        key = str(control).strip()
        if key not in EXTRA_CONTROL_ALIASES:
            raise ValueError(f"unknown extra control: {control}")
        normalized.append(EXTRA_CONTROL_ALIASES[key])
    return tuple(dict.fromkeys(normalized))


def evaluate_methods(case: ControlledCase, extra_controls: tuple[str, ...] | list[str] | None = None) -> list[dict[str, object]]:
    extra_controls = normalize_extra_controls(extra_controls)
    aligned_models = [align_hidden_to_base(model) for model in case.local_models]
    ordinary = average_models(case.local_models)
    aligned = average_models(aligned_models)
    local_metrics = evaluate_local_models(case)
    rng = np.random.default_rng(case.seed + 991 * case.width)
    assignments = {
        "twisted_q2_branch": twisted_branch_assignment(case),
        "random_branch_ensemble": random_branch_assignment(case, rng),
        "validation_selected_branch_ensemble": global_validation_branch_assignment(case, aligned),
        "c2m3_cluster_branch_ensemble": {face: 0 for face in case.alpha_signs},
    }
    control_details: dict[str, dict[str, object]] = {}
    if "wrong_twist_control" in extra_controls:
        assignments["wrong_twist_control"] = wrong_twist_assignment(case)
        control_details["wrong_twist_control"] = {"control_type": "wrong_twist"}
    if "wrong_context_control" in extra_controls:
        assignments["wrong_context_control"] = wrong_context_assignment(case, rng)
        control_details["wrong_context_control"] = {"control_type": "wrong_context"}
    if "learned_context_router" in extra_controls:
        assignment, details = learned_context_router_assignment(case, aligned)
        assignments["learned_context_router"] = assignment
        control_details["learned_context_router"] = {"control_type": "learned_router", **details}
    if "no_twist_branch_control" in extra_controls:
        assignments["no_twist_branch_control"] = no_twist_branch_assignment(case)
        control_details["no_twist_branch_control"] = {"control_type": "no_twist_branch"}
    distilled_scale = 1.0
    if "distilled_twisted_single_model" in extra_controls:
        distilled_scale, details = distilled_twisted_scale(case, aligned)
        control_details["distilled_twisted_single_model"] = {"control_type": "distilled_single", **details}
    wide_model = None
    if "parameter_matched_wide_control" in extra_controls:
        wide_model = make_parameter_matched_wide_model(aligned, aligned.parameter_count * case.branch_count)
        control_details["parameter_matched_wide_control"] = {
            "control_type": "parameter_matched_wide",
            "wide_hidden_width": wide_model.width,
        }
    predictors: dict[str, Callable[[Face, np.ndarray], np.ndarray]] = {
        "ordinary_weight_average": lambda _face, x: ordinary.logits(x),
        "git_rebasin_pairwise": lambda _face, x: aligned.logits(x),
        "c2m3_synchronized": lambda _face, x: aligned.logits(x),
        "ensemble_upper_bound": lambda _face, x: np.stack([model.logits(x) for model in case.local_models]).mean(axis=0),
    }
    if "distilled_twisted_single_model" in extra_controls:
        predictors["distilled_twisted_single_model"] = lambda _face, x: distilled_scale * aligned.logits(x)
    if "parameter_matched_wide_control" in extra_controls:
        assert wide_model is not None
        predictors["parameter_matched_wide_control"] = lambda _face, x: wide_model.logits(x)
    rows = []
    methods = tuple(METHODS) + tuple(control for control in extra_controls if control not in METHODS)
    for method in methods:
        if method in assignments:
            val_metrics = evaluate_branch_assignment(case, aligned, case.val_face_data, assignments[method])
            test_metrics = evaluate_branch_assignment(case, aligned, case.test_face_data, assignments[method])
            branch_assignment = assignments[method]
        else:
            val_metrics = evaluate_face_predictor(case.val_face_data, predictors[method])
            test_metrics = evaluate_face_predictor(case.test_face_data, predictors[method])
            branch_assignment = {}
        parameter_count = wide_model.parameter_count if method == "parameter_matched_wide_control" and wide_model is not None else None
        meta = method_capacity_metadata(method, case, aligned, parameter_count=parameter_count)
        if case.family == "random_noncentral" and method == "twisted_q2_branch":
            claim_role = "noncentral_control_not_mu2_claim"
        elif case.family == "mu2_nontrivial_h2" and method == "twisted_q2_branch":
            claim_role = "controlled_central_h2_branch_evidence"
        elif case.family == "mu2_coboundary" and method == "c2m3_synchronized":
            claim_role = "controlled_coboundary_single_model_evidence"
        elif method == "learned_context_router":
            claim_role = "validation_only_router_diagnostic"
        elif method in EXTRA_CONTROL_METHODS:
            claim_role = "hardening_control"
        else:
            claim_role = "baseline_or_diagnostic"
        details = control_details.get(method, {})
        rows.append(
            {
                "method": method,
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "local_model_accuracy": local_metrics["accuracy"],
                "local_model_loss": local_metrics["loss"],
                "branch_assignment": {("-".join(map(str, face))): int(branch) for face, branch in branch_assignment.items()},
                "claim_role": claim_role,
                "is_extra_control": method in EXTRA_CONTROL_METHODS,
                "control_type": details.get("control_type", ""),
                "router_type": details.get("router_type", ""),
                "router_train_accuracy": details.get("router_train_accuracy", float("nan")),
                "router_branch_scores_json": details.get("router_branch_scores", {}),
                "distillation_scale": details.get("distillation_scale", float("nan")),
                "distillation_target": details.get("distillation_target", ""),
                "distillation_teacher_mse_against_base": details.get(
                    "distillation_teacher_mse_against_base",
                    float("nan"),
                ),
                "wide_hidden_width": details.get("wide_hidden_width", float("nan")),
                **meta,
            }
        )
    lookup = {row["method"]: row for row in rows}
    for row in rows:
        row["delta_vs_weight_average"] = float(row["test_accuracy"] - lookup["ordinary_weight_average"]["test_accuracy"])
        row["delta_vs_c2m3"] = float(row["test_accuracy"] - lookup["c2m3_synchronized"]["test_accuracy"])
        row["delta_vs_random_branch"] = float(row["test_accuracy"] - lookup["random_branch_ensemble"]["test_accuracy"])
        row["delta_vs_validation_branch"] = float(row["test_accuracy"] - lookup["validation_selected_branch_ensemble"]["test_accuracy"])
        row["delta_vs_c2m3_cluster_branch"] = float(row["test_accuracy"] - lookup["c2m3_cluster_branch_ensemble"]["test_accuracy"])
        for control_method in EXTRA_CONTROL_METHODS:
            key = f"delta_vs_{control_method}"
            row[key] = (
                float(row["test_accuracy"] - lookup[control_method]["test_accuracy"])
                if control_method in lookup
                else float("nan")
            )
    return rows


def pairwise_rows_for_case(case: ControlledCase) -> list[dict[str, object]]:
    rows = []
    for face, face_gauges in sorted(case.observed_edge_gauges.items()):
        for edge, observed in sorted(face_gauges.items()):
            true = case.true_edge_gauges[face][edge]
            rows.append(
                {
                    "family": case.family,
                    "seed": case.seed,
                    "width": case.width,
                    "n_models": case.n_models,
                    "face": "-".join(map(str, face)),
                    "edge": f"{edge[0]}->{edge[1]}",
                    "overlap_id": case.overlap_ids[face],
                    "true_gauge_perm": true.astype(int).tolist(),
                    "observed_gauge_perm": observed.astype(int).tolist(),
                    "pairwise_alignment_residual": perm_residual(observed, true),
                    "pairwise_alignment_accuracy": 1.0 - perm_residual(observed, true),
                }
            )
    return rows


def checkpoint_metadata(case: ControlledCase, model: ControlledMLP, checkpoint_path: Path) -> dict[str, object]:
    return {
        "family": case.family,
        "seed": case.seed,
        "width": case.width,
        "n_models": case.n_models,
        "model_index": model.model_index,
        "input_dim": model.input_dim,
        "hidden_width": model.width,
        "checkpoint_path": str(checkpoint_path),
        "hidden_permutation": (model.hidden_permutation.astype(int).tolist() if model.hidden_permutation is not None else []),
    }


def save_local_checkpoints(case: ControlledCase, checkpoint_dir: Path) -> list[dict[str, object]]:
    rows = []
    for model in case.local_models:
        path = checkpoint_dir / case.family / f"width{case.width}" / f"seed{case.seed}_model{model.model_index}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = checkpoint_metadata(case, model, path)
        np.savez_compressed(
            path,
            hidden_weight=model.hidden_weight,
            hidden_bias=model.hidden_bias,
            output_weight=model.output_weight,
            output_bias=np.asarray([model.output_bias]),
            hidden_permutation=model.hidden_permutation if model.hidden_permutation is not None else np.array([], dtype=int),
            metadata=np.asarray([repr(metadata)]),
        )
        rows.append(metadata)
    return rows


def bootstrap_mean_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or samples <= 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    estimates = []
    for _ in range(samples):
        idx = rng.integers(0, len(values), len(values))
        estimates.append(float(values[idx].mean()))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))
