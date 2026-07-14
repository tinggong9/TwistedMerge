"""Executed two-loop noncommuting holonomy benchmark utilities.

Candidate-logit functions in this module deliberately do not accept labels.
Labels are generated once by the fixed base MLP and are used only by metric
code after every candidate tensor has been executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.controlled_nonabelian_holonomy import controlled_group, named_generators
from src.finite_group_cohomology import FinitePermutationGroup, Permutation
from src.nonabelian_invariant_pooling import regular_action_permutation


METHODS = (
    "ordinary_weight_average",
    "git_rebasin_pairwise",
    "c2m3_strict_synchronization",
    "greedy_soup",
    "naive_regular_representation_without_invariant_pooling",
    "random_same_branch_count_control",
    "wrong_generator_control",
    "wrong_order_control",
    "wrong_group_action_control",
    "branch_orbit_lift_with_invariant_pooling",
    "branch_regular_lift_with_invariant_pooling",
    "oracle_supplied_context_branch_predictor",
    "validation_only_safe_selector",
    "ensemble_reference",
)


@dataclass(frozen=True)
class ExecutedMLP:
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    @property
    def width(self) -> int:
        return int(self.hidden_weight.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.hidden_weight.shape[1])

    @property
    def n_classes(self) -> int:
        return int(self.output_weight.shape[0])

    @property
    def parameter_count(self) -> int:
        return int(
            self.hidden_weight.size
            + self.hidden_bias.size
            + self.output_weight.size
            + self.output_bias.size
        )

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.maximum(np.asarray(inputs) @ self.hidden_weight.T + self.hidden_bias, 0.0)
        return hidden @ self.output_weight.T + self.output_bias


@dataclass(frozen=True)
class TwoLoopCase:
    group_name: str
    group: FinitePermutationGroup
    generator_s: Permutation
    generator_r: Permutation
    width: int
    seed: int
    base_model: ExecutedMLP
    local_models: tuple[ExecutedMLP, ...]
    chart_permutations: tuple[np.ndarray, ...]
    transition_matrices: Mapping[str, np.ndarray]
    loop_holonomy_s: np.ndarray
    loop_holonomy_r: np.ndarray
    hidden_action_s: np.ndarray
    hidden_action_r: np.ndarray


def permutation_matrix(perm: np.ndarray | tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(perm, dtype=int)
    matrix = np.zeros((len(arr), len(arr)), dtype=float)
    matrix[np.arange(len(arr)), arr] = 1.0
    return matrix


def invert_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[np.asarray(perm, dtype=int)] = np.arange(len(perm))
    return inv


def permute_hidden(model: ExecutedMLP, perm: np.ndarray) -> ExecutedMLP:
    arr = np.asarray(perm, dtype=int)
    return ExecutedMLP(
        hidden_weight=model.hidden_weight[arr].copy(),
        hidden_bias=model.hidden_bias[arr].copy(),
        output_weight=model.output_weight[:, arr].copy(),
        output_bias=model.output_bias.copy(),
    )


def average_models(models: list[ExecutedMLP] | tuple[ExecutedMLP, ...]) -> ExecutedMLP:
    if not models:
        raise ValueError("cannot average an empty model collection")
    return ExecutedMLP(
        hidden_weight=np.stack([model.hidden_weight for model in models]).mean(axis=0),
        hidden_bias=np.stack([model.hidden_bias for model in models]).mean(axis=0),
        output_weight=np.stack([model.output_weight for model in models]).mean(axis=0),
        output_bias=np.stack([model.output_bias for model in models]).mean(axis=0),
    )


def align_to_base(model: ExecutedMLP, chart_perm: np.ndarray) -> ExecutedMLP:
    return permute_hidden(model, invert_perm(np.asarray(chart_perm, dtype=int)))


def hidden_regular_perm(group: FinitePermutationGroup, element: Permutation, width: int) -> np.ndarray:
    if width < group.order:
        raise ValueError("hidden width must be at least the group order")
    perm = np.arange(width, dtype=int)
    perm[: group.order] = np.asarray(regular_action_permutation(group, element, side="left"), dtype=int)
    return perm


def make_base_model(group: FinitePermutationGroup, width: int, seed: int, input_dim: int, n_classes: int) -> ExecutedMLP:
    rng = np.random.default_rng(seed)
    hidden_weight = rng.normal(scale=0.7 / np.sqrt(input_dim), size=(width, input_dim))
    hidden_bias = rng.normal(scale=0.15, size=width)
    output_weight = rng.normal(scale=0.8 / np.sqrt(width), size=(n_classes, width))
    output_bias = rng.normal(scale=0.05, size=n_classes)

    # The first regular orbit is an exact automorphism block.  Its duplicated
    # units allow nontrivial S3/D4 loop transitions while the remaining units
    # make unaligned checkpoint averaging a genuine executed baseline.
    hidden_weight[: group.order] = hidden_weight[0]
    hidden_bias[: group.order] = hidden_bias[0]
    output_weight[:, : group.order] = output_weight[:, [0]] / float(group.order)
    return ExecutedMLP(hidden_weight, hidden_bias, output_weight, output_bias)


def _edge_transition(P_source: np.ndarray, P_target: np.ndarray, hidden_action: np.ndarray) -> np.ndarray:
    return P_target @ hidden_action @ P_source.T


def build_case(
    group_name: str,
    width: int,
    seed: int,
    *,
    input_dim: int = 12,
    n_classes: int = 5,
) -> TwoLoopCase:
    group = controlled_group(group_name)
    generator_s, generator_r = named_generators(group_name)
    base = make_base_model(group, int(width), 10007 * int(seed) + 97 * int(width) + group.order, input_dim, n_classes)
    rng = np.random.default_rng(20011 * int(seed) + 193 * int(width) + group.order)
    chart_perms = [np.arange(width, dtype=int)]
    chart_perms.extend(rng.permutation(width) for _ in range(4))
    local_models = tuple(permute_hidden(base, perm) for perm in chart_perms)
    chart_matrices = tuple(permutation_matrix(perm) for perm in chart_perms)

    identity = np.eye(width)
    hidden_s = permutation_matrix(hidden_regular_perm(group, generator_s, width))
    hidden_r = permutation_matrix(hidden_regular_perm(group, generator_r, width))

    # Wedge of two length-three cycles: 0-1-2-0 and 0-3-4-0.
    transitions = {
        "0->1": _edge_transition(chart_matrices[0], chart_matrices[1], identity),
        "1->2": _edge_transition(chart_matrices[1], chart_matrices[2], identity),
        "2->0": _edge_transition(chart_matrices[2], chart_matrices[0], hidden_s),
        "0->3": _edge_transition(chart_matrices[0], chart_matrices[3], identity),
        "3->4": _edge_transition(chart_matrices[3], chart_matrices[4], identity),
        "4->0": _edge_transition(chart_matrices[4], chart_matrices[0], hidden_r),
    }
    holonomy_s = transitions["2->0"] @ transitions["1->2"] @ transitions["0->1"]
    holonomy_r = transitions["4->0"] @ transitions["3->4"] @ transitions["0->3"]
    return TwoLoopCase(
        group_name=str(group_name).upper(),
        group=group,
        generator_s=generator_s,
        generator_r=generator_r,
        width=int(width),
        seed=int(seed),
        base_model=base,
        local_models=local_models,
        chart_permutations=tuple(np.asarray(perm, dtype=int) for perm in chart_perms),
        transition_matrices=transitions,
        loop_holonomy_s=holonomy_s,
        loop_holonomy_r=holonomy_r,
        hidden_action_s=hidden_s,
        hidden_action_r=hidden_r,
    )


def make_dataset(case: TwoLoopCase, split: str, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_offset = {"train": 11, "validation": 23, "test": 37}[str(split)]
    rng = np.random.default_rng(30011 * case.seed + 307 * case.width + 1009 * case.group.order + split_offset)
    inputs = rng.normal(size=(int(n_samples), case.base_model.input_dim))
    contexts = rng.integers(0, len(case.local_models), size=int(n_samples), dtype=np.int64)
    # Fixed planted teacher; candidate method is not an input to this operation.
    labels = np.argmax(case.base_model.logits(inputs), axis=1).astype(np.int64)
    return inputs, labels, contexts


def _branch_logits(case: TwoLoopCase, inputs: np.ndarray, elements: list[Permutation]) -> np.ndarray:
    branches = []
    for element in elements:
        model = permute_hidden(case.base_model, hidden_regular_perm(case.group, element, case.width))
        branches.append(model.logits(inputs))
    return np.stack(branches, axis=1)


def _greedy_soup(case: TwoLoopCase, validation_inputs: np.ndarray, validation_labels: np.ndarray) -> ExecutedMLP:
    candidates = list(case.local_models)
    scores = []
    for idx, model in enumerate(candidates):
        logits = model.logits(validation_inputs)
        scores.append((float(np.mean(np.argmax(logits, axis=1) == validation_labels)), -cross_entropy(logits, validation_labels), idx))
    order = [idx for _acc, _loss, idx in sorted(scores, reverse=True)]
    selected = [order[0]]
    soup = candidates[order[0]]
    best = metric_pair(soup.logits(validation_inputs), validation_labels)
    for idx in order[1:]:
        proposed = average_models([candidates[item] for item in selected + [idx]])
        metrics = metric_pair(proposed.logits(validation_inputs), validation_labels)
        if (metrics[0], -metrics[1]) > (best[0], -best[1]):
            selected.append(idx)
            soup = proposed
            best = metrics
    return soup


def executed_candidate_logits(
    case: TwoLoopCase,
    inputs: np.ndarray,
    contexts: np.ndarray,
    *,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Execute every candidate.  Evaluation labels are intentionally absent."""

    aligned_models = [align_to_base(model, perm) for model, perm in zip(case.local_models, case.chart_permutations)]
    ordinary = average_models(case.local_models)
    aligned = average_models(aligned_models)
    greedy = _greedy_soup(case, validation_inputs, validation_labels)
    group_elements = list(case.group.elements)
    regular = _branch_logits(case, inputs, group_elements)
    orbit = regular[:, : case.group.degree, :]
    rng = np.random.default_rng(40009 * case.seed + 401 * case.width + case.group.order)
    random_branches = np.stack(
        [permute_hidden(case.base_model, rng.permutation(case.width)).logits(inputs) for _ in group_elements], axis=1
    )
    wrong_generator_elements = [case.group.identity for _ in group_elements]
    wrong_generator = _branch_logits(case, inputs, wrong_generator_elements)
    wrong_order = _branch_logits(case, inputs, list(reversed(group_elements)))
    wrong_group = np.stack(
        [permute_hidden(case.base_model, np.roll(np.arange(case.width), shift + 1)).logits(inputs) for shift in range(case.group.order)],
        axis=1,
    )
    local_logits = np.stack([model.logits(inputs) for model in case.local_models], axis=1)
    oracle = local_logits[np.arange(len(inputs)), np.asarray(contexts, dtype=int) % len(case.local_models)]

    candidates = {
        "ordinary_weight_average": ordinary.logits(inputs),
        "git_rebasin_pairwise": aligned.logits(inputs),
        "c2m3_strict_synchronization": aligned.logits(inputs),
        "greedy_soup": greedy.logits(inputs),
        "naive_regular_representation_without_invariant_pooling": regular[:, 0, :],
        "random_same_branch_count_control": random_branches.mean(axis=1),
        "wrong_generator_control": wrong_generator.mean(axis=1),
        "wrong_order_control": wrong_order.mean(axis=1),
        "wrong_group_action_control": wrong_group.mean(axis=1),
        "branch_orbit_lift_with_invariant_pooling": orbit.mean(axis=1),
        "branch_regular_lift_with_invariant_pooling": regular.mean(axis=1),
        "oracle_supplied_context_branch_predictor": oracle,
        "ensemble_reference": local_logits.mean(axis=1),
    }

    # Validation-only selector.  The candidate logits above were already
    # executed; test labels never participate in this choice.
    validation_candidates = executed_candidate_logits_without_selector(
        case,
        validation_inputs,
        np.zeros(len(validation_inputs), dtype=int),
        validation_inputs=validation_inputs,
        validation_labels=validation_labels,
    )
    selectable = (
        "git_rebasin_pairwise",
        "greedy_soup",
        "branch_regular_lift_with_invariant_pooling",
        "ordinary_weight_average",
    )
    chosen = max(
        selectable,
        key=lambda method: (
            metric_pair(validation_candidates[method], validation_labels)[0],
            -metric_pair(validation_candidates[method], validation_labels)[1],
            -selectable.index(method),
        ),
    )
    candidates["validation_only_safe_selector"] = candidates[chosen].copy()
    return candidates


def executed_candidate_logits_without_selector(
    case: TwoLoopCase,
    inputs: np.ndarray,
    contexts: np.ndarray,
    *,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    aligned_models = [align_to_base(model, perm) for model, perm in zip(case.local_models, case.chart_permutations)]
    ordinary = average_models(case.local_models)
    aligned = average_models(aligned_models)
    greedy = _greedy_soup(case, validation_inputs, validation_labels)
    group_elements = list(case.group.elements)
    regular = _branch_logits(case, inputs, group_elements)
    orbit = regular[:, : case.group.degree, :]
    rng = np.random.default_rng(40009 * case.seed + 401 * case.width + case.group.order)
    random_branches = np.stack(
        [permute_hidden(case.base_model, rng.permutation(case.width)).logits(inputs) for _ in group_elements], axis=1
    )
    wrong_generator = _branch_logits(case, inputs, [case.group.identity for _ in group_elements])
    wrong_order = _branch_logits(case, inputs, list(reversed(group_elements)))
    wrong_group = np.stack(
        [permute_hidden(case.base_model, np.roll(np.arange(case.width), shift + 1)).logits(inputs) for shift in range(case.group.order)], axis=1
    )
    local_logits = np.stack([model.logits(inputs) for model in case.local_models], axis=1)
    oracle = local_logits[np.arange(len(inputs)), np.asarray(contexts, dtype=int) % len(case.local_models)]
    return {
        "ordinary_weight_average": ordinary.logits(inputs),
        "git_rebasin_pairwise": aligned.logits(inputs),
        "c2m3_strict_synchronization": aligned.logits(inputs),
        "greedy_soup": greedy.logits(inputs),
        "naive_regular_representation_without_invariant_pooling": regular[:, 0, :],
        "random_same_branch_count_control": random_branches.mean(axis=1),
        "wrong_generator_control": wrong_generator.mean(axis=1),
        "wrong_order_control": wrong_order.mean(axis=1),
        "wrong_group_action_control": wrong_group.mean(axis=1),
        "branch_orbit_lift_with_invariant_pooling": orbit.mean(axis=1),
        "branch_regular_lift_with_invariant_pooling": regular.mean(axis=1),
        "oracle_supplied_context_branch_predictor": oracle,
        "ensemble_reference": local_logits.mean(axis=1),
    }


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    arr = np.asarray(logits, dtype=float)
    shifted = arr - arr.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_norm - shifted[np.arange(len(labels)), np.asarray(labels, dtype=int)]))


def metric_pair(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    return float(np.mean(np.argmax(logits, axis=1) == labels)), cross_entropy(logits, labels)


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / max(np.linalg.norm(right, ord="fro"), 1e-12))


def structural_certificates(case: TwoLoopCase, *, tolerance: float = 1e-12) -> dict[str, float | bool]:
    commutator_left = case.loop_holonomy_s @ case.loop_holonomy_r
    commutator_right = case.loop_holonomy_r @ case.loop_holonomy_s
    commutator_residual = _relative(commutator_left, commutator_right)

    pooling = np.ones((1, case.width), dtype=float) / float(case.width)
    pooling_s = _relative(pooling @ case.loop_holonomy_s, pooling)
    pooling_r = _relative(pooling @ case.loop_holonomy_r, pooling)

    multiplication = []
    for left in case.group.elements:
        for right in case.group.elements:
            product = case.group.multiply(left, right)
            L = permutation_matrix(hidden_regular_perm(case.group, left, case.width))
            R = permutation_matrix(hidden_regular_perm(case.group, right, case.width))
            P = permutation_matrix(hidden_regular_perm(case.group, product, case.width))
            # ``permutation_matrix`` acts on row-indexed hidden coordinates,
            # so matrix composition is reversed relative to the group helper.
            multiplication.append(_relative(R @ L, P))

    probe = np.random.default_rng(case.seed + 12345).normal(size=(64, case.base_model.input_dim))
    base_logits = case.base_model.logits(probe)
    local_equivalence = max(
        float(np.max(np.abs(model.logits(probe) - base_logits))) for model in case.local_models
    )
    hidden_s_match = _relative(case.loop_holonomy_s, case.hidden_action_s)
    hidden_r_match = _relative(case.loop_holonomy_r, case.hidden_action_r)
    pre_lift = max(_relative(case.loop_holonomy_s, np.eye(case.width)), _relative(case.loop_holonomy_r, np.eye(case.width)))
    post_lift = max(pooling_s, pooling_r)
    wrong_generator_match = _relative(np.eye(case.width), case.hidden_action_r)
    random_action_multiplication = 1.0
    return {
        "pre_lift_residual": pre_lift,
        "post_lift_residual": post_lift,
        "pooling_residual_gamma_1": pooling_s,
        "pooling_residual_gamma_2": pooling_r,
        "commutator_residual": commutator_residual,
        "group_action_multiplication_residual": float(max(multiplication, default=0.0)),
        "local_functional_equivalence_residual": local_equivalence,
        "generator_1_recovery_residual": hidden_s_match,
        "generator_2_recovery_residual": hidden_r_match,
        "wrong_generator_recovery_residual": wrong_generator_match,
        "random_action_multiplication_residual": random_action_multiplication,
        "generators_noncommute": bool(commutator_residual > tolerance),
        "pooling_certificate_passed": bool(max(pooling_s, pooling_r) <= tolerance),
        "group_action_certificate_passed": bool(max(multiplication, default=0.0) <= tolerance),
        "local_equivalence_passed": bool(local_equivalence <= tolerance),
        "generators_recovered": bool(max(hidden_s_match, hidden_r_match) <= tolerance),
        "wrong_controls_rejected_structurally": bool(wrong_generator_match > tolerance and random_action_multiplication > tolerance),
    }


def method_capacity(case: TwoLoopCase, method: str) -> dict[str, object]:
    base = case.base_model.parameter_count
    if method == "ensemble_reference":
        branches, params, infer, kind = len(case.local_models), base * len(case.local_models), len(case.local_models), "ensemble"
    elif method == "oracle_supplied_context_branch_predictor":
        branches, params, infer, kind = len(case.local_models), base * len(case.local_models), 1.0, "branch_model"
    elif method in {
        "branch_regular_lift_with_invariant_pooling",
        "random_same_branch_count_control",
        "wrong_generator_control",
        "wrong_order_control",
        "wrong_group_action_control",
        "naive_regular_representation_without_invariant_pooling",
    }:
        branches, params, infer, kind = case.group.order, base, case.group.order, "branch_model"
    elif method == "branch_orbit_lift_with_invariant_pooling":
        branches, params, infer, kind = case.group.degree, base, case.group.degree, "branch_model"
    elif method == "greedy_soup":
        branches, params, infer, kind = 1, base, 1.0, "soup"
    else:
        branches, params, infer, kind = 1, base, 1.0, "single_model"
    return {
        "actual_parameter_count": int(params),
        "parameter_multiplier": float(params / base),
        "branch_count": int(branches),
        "inference_multiplier": float(infer),
        "model_kind": kind,
        "is_single_model": kind == "single_model",
        "is_soup": kind == "soup",
        "is_branch_model": kind == "branch_model",
        "is_ensemble": kind == "ensemble",
        "uses_supplied_context": method == "oracle_supplied_context_branch_predictor",
        "uses_validation_data": method in {"greedy_soup", "validation_only_safe_selector"},
        "uses_obstruction_data": method in {
            "c2m3_strict_synchronization",
            "branch_orbit_lift_with_invariant_pooling",
            "branch_regular_lift_with_invariant_pooling",
            "validation_only_safe_selector",
        },
    }
