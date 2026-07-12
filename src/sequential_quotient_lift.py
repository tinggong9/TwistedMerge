"""Certified quotient chains and auditable prediction-level branch lifts.

The routines here are deliberately conservative.  They certify C2/C3 quotient
factors from exact multiplication tables, represent the lift by the coset action
on ``Gamma / K_j``, and expose residuals that are recomputed from the current
kernel rather than hard-coded to zero.  Truncated groups may use the ambient
permutation sign character as a first C2 certificate, but the code does not
pretend to know the exact kernel order or recurse into an incomplete kernel.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from src.finite_group_cohomology import (
    FinitePermutationGroup,
    Permutation,
    close_permutation_group,
    compose_permutations,
    cyclic_group,
    dihedral_group_4,
    identity_permutation,
    invert_permutation,
    klein_four_group,
    normalize_permutation,
    symmetric_group_3,
)


@dataclass(frozen=True)
class QuotientCertificate:
    quotient_name: str
    quotient_order: int
    assignment: dict[Permutation, int]
    homomorphism_residual: float
    kernel: tuple[Permutation, ...]
    kernel_order: int | None
    kernel_normal: bool | None
    image_size: int
    certified: bool
    certification_method: str
    rejection_reason: str = ""


@dataclass(frozen=True)
class CosetActionRepresentation:
    cosets: tuple[tuple[Permutation, ...], ...]
    action: dict[Permutation, Permutation]
    law_residual: float
    kernel: tuple[Permutation, ...]
    kernel_order: int
    stabilizer_matches_subgroup: bool
    permutation_conjugate_to_regular: bool


@dataclass(frozen=True)
class QuotientChainStage:
    depth: int
    source_group_order: int
    quotient: QuotientCertificate
    residual_before: float
    residual_after: float
    branch_multiplier: int
    residual_group_order: int | None = None
    coset_count: int | None = None
    coset_action_law_residual: float | None = None
    coset_action_kernel_order: int | None = None
    stabilizer_matches_kernel: bool | None = None
    final_regular_representation_verified: bool = False


@dataclass(frozen=True)
class SequentialQuotientChain:
    group_name: str
    group_order: int
    closure_status: str
    truncated: bool
    stages: tuple[QuotientChainStage, ...]
    final_kernel_order: int | None
    stopped_reason: str


def permutation_parity(perm: Sequence[int]) -> int:
    arr = normalize_permutation(perm)
    inv_count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            inv_count += int(arr[i] > arr[j])
    return inv_count % 2


def multiplication_table(group: FinitePermutationGroup) -> dict[tuple[Permutation, Permutation], Permutation]:
    return {(left, right): group.multiply(left, right) for left in group.elements for right in group.elements}


def verify_closure(group: FinitePermutationGroup) -> bool:
    if group.truncated:
        return False
    elements = set(group.elements)
    return all(group.multiply(left, right) in elements for left in group.elements for right in group.elements)


def homomorphism_residual(group: FinitePermutationGroup, assignment: Mapping[Permutation, int], q: int) -> float:
    failures = 0
    total = 0
    for left, right in product(group.elements, repeat=2):
        product_element = group.multiply(left, right)
        if product_element not in assignment:
            failures += 1
            total += 1
            continue
        total += 1
        lhs = (int(assignment.get(left, 0)) + int(assignment.get(right, 0))) % int(q)
        rhs = int(assignment.get(product_element, 0)) % int(q)
        failures += int(lhs != rhs)
    return float(failures / max(1, total))


def kernel_is_normal(group: FinitePermutationGroup, kernel: Iterable[Permutation]) -> bool:
    kernel_set = set(kernel)
    if not kernel_set:
        return False
    for g in group.elements:
        g_inv = group.inverse(g)
        for k in kernel_set:
            if group.multiply(group.multiply(g, k), g_inv) not in kernel_set:
                return False
    return True


def _certificate(group: FinitePermutationGroup, q: int, assignment: dict[Permutation, int], method: str) -> QuotientCertificate:
    residual = homomorphism_residual(group, assignment, q)
    kernel = tuple(sorted(element for element in group.elements if int(assignment.get(element, 0)) % int(q) == 0))
    image = {int(value) % int(q) for value in assignment.values()}
    normal = kernel_is_normal(group, kernel)
    certified = bool(residual == 0.0 and len(image) == int(q) and normal)
    return QuotientCertificate(
        quotient_name=f"C{int(q)}",
        quotient_order=int(q),
        assignment={element: int(value) % int(q) for element, value in assignment.items()},
        homomorphism_residual=float(residual),
        kernel=kernel,
        kernel_order=len(kernel),
        kernel_normal=bool(normal),
        image_size=len(image),
        certified=certified,
        certification_method=method,
        rejection_reason="" if certified else "homomorphism_not_surjective_or_kernel_not_normal",
    )


def _bruteforce_homomorphisms(group: FinitePermutationGroup, q: int, max_assignments: int) -> list[QuotientCertificate]:
    if group.truncated:
        return []
    identity = group.identity
    nonidentity = [element for element in group.elements if element != identity]
    total = int(q) ** len(nonidentity)
    if total > int(max_assignments):
        return []
    out: list[QuotientCertificate] = []
    for values in product(range(int(q)), repeat=len(nonidentity)):
        if not any(values):
            continue
        assignment = {identity: 0}
        assignment.update({element: int(value) for element, value in zip(nonidentity, values)})
        cert = _certificate(group, q, assignment, "exact_multiplication_table_bruteforce")
        if cert.certified:
            out.append(cert)
    out.sort(key=lambda cert: (cert.quotient_order, cert.kernel_order or 10**9, tuple(cert.assignment[e] for e in group.elements)))
    return out


def _sign_certificate(group: FinitePermutationGroup) -> QuotientCertificate:
    assignment = {element: permutation_parity(element) for element in group.elements}
    image = set(assignment.values())
    if group.truncated:
        kernel = tuple(sorted(element for element in group.elements if assignment[element] == 0))
        return QuotientCertificate(
            quotient_name="C2",
            quotient_order=2,
            assignment=assignment,
            homomorphism_residual=0.0,
            kernel=kernel,
            kernel_order=None,
            kernel_normal=None,
            image_size=len(image),
            certified=len(image) == 2,
            certification_method="ambient_permutation_sign_character_truncated_group",
            rejection_reason="" if len(image) == 2 else "group_has_no_observed_odd_permutation",
        )
    cert = _certificate(group, 2, assignment, "permutation_sign_character")
    if cert.image_size < 2:
        return QuotientCertificate(
            quotient_name="C2",
            quotient_order=2,
            assignment=assignment,
            homomorphism_residual=0.0,
            kernel=tuple(group.elements),
            kernel_order=group.order,
            kernel_normal=True,
            image_size=cert.image_size,
            certified=False,
            certification_method="permutation_sign_character",
            rejection_reason="group_has_no_observed_odd_permutation",
        )
    return cert


def certified_cyclic_quotients(
    group: FinitePermutationGroup,
    target_orders: Sequence[int] = (2, 3),
    max_exact_order: int = 64,
    max_assignments: int = 100_000,
) -> list[QuotientCertificate]:
    """Return certified C2/C3 quotients without element-order heuristics."""

    out: list[QuotientCertificate] = []
    targets = tuple(int(q) for q in target_orders)
    if 2 in targets:
        sign = _sign_certificate(group)
        if sign.certified:
            out.append(sign)
    if not group.truncated and group.order <= int(max_exact_order):
        for q in targets:
            for cert in _bruteforce_homomorphisms(group, q, max_assignments=max_assignments):
                duplicate = any(
                    cert.assignment == existing.assignment and cert.quotient_order == existing.quotient_order
                    for existing in out
                )
                if not duplicate:
                    out.append(cert)
    out.sort(key=lambda cert: (cert.quotient_order, cert.kernel_order or 10**9, cert.certification_method))
    return out


def subgroup_from_elements(elements: Sequence[Permutation], parent: FinitePermutationGroup) -> FinitePermutationGroup:
    elems = tuple(sorted(set(elements)))
    if not elems:
        elems = (parent.identity,)
    return FinitePermutationGroup(
        elements=elems,
        generators=elems,
        closure_status="exact_kernel_subgroup",
        truncated=False,
    )


def _coset_key(coset: Iterable[Permutation]) -> frozenset[Permutation]:
    return frozenset(coset)


def left_cosets(group: FinitePermutationGroup, subgroup: Sequence[Permutation]) -> tuple[tuple[Permutation, ...], ...]:
    subgroup_tuple = tuple(sorted(set(subgroup)))
    if not subgroup_tuple:
        raise ValueError("subgroup cannot be empty")
    remaining = set(group.elements)
    cosets: list[tuple[Permutation, ...]] = []
    while remaining:
        representative = min(remaining)
        coset = tuple(sorted(group.multiply(representative, element) for element in subgroup_tuple))
        cosets.append(coset)
        remaining.difference_update(coset)
    return tuple(cosets)


def coset_action_representation(
    group: FinitePermutationGroup,
    subgroup: Sequence[Permutation],
) -> CosetActionRepresentation:
    """Return the left action of ``group`` on cosets ``group / subgroup``."""

    subgroup_tuple = tuple(sorted(set(subgroup)))
    cosets = left_cosets(group, subgroup_tuple)
    coset_index = {_coset_key(coset): idx for idx, coset in enumerate(cosets)}
    action: dict[Permutation, Permutation] = {}
    for element in group.elements:
        image = []
        for coset in cosets:
            acted = tuple(sorted(group.multiply(element, member) for member in coset))
            image.append(coset_index[_coset_key(acted)])
        action[element] = tuple(image)

    failures = 0
    total = 0
    for left, right in product(group.elements, repeat=2):
        total += 1
        product_action = action[group.multiply(left, right)]
        # The repository's permutation convention composes as ``right after
        # left``.  The coset action is a left action, so the action of the
        # product uses the reversed tuple-composition order here.
        composed = compose_permutations(action[right], action[left])
        failures += int(product_action != composed)
    law_residual = float(failures / max(1, total))
    identity_action = identity_permutation(len(cosets))
    kernel = tuple(sorted(element for element in group.elements if action[element] == identity_action))
    subgroup_set = set(subgroup_tuple)
    kernel_matches = set(kernel) == subgroup_set
    regular_verified = bool(len(cosets) == group.order and len(kernel) == 1 and law_residual == 0.0)
    return CosetActionRepresentation(
        cosets=cosets,
        action=action,
        law_residual=law_residual,
        kernel=kernel,
        kernel_order=len(kernel),
        stabilizer_matches_subgroup=kernel_matches,
        permutation_conjugate_to_regular=regular_verified,
    )


def triangle_residual(elements: Sequence[Permutation], group: FinitePermutationGroup) -> float:
    identity = group.identity
    vals = [element != identity for element in elements]
    return float(np.mean(vals)) if vals else 0.0


def quotient_residual(elements: Sequence[Permutation], cert: QuotientCertificate) -> float:
    vals = [
        int(cert.assignment.get(element, 0)) % cert.quotient_order != 0
        for element in elements
        if element in cert.assignment
    ]
    return float(np.mean(vals)) if vals else 0.0


def build_successive_quotient_chain(
    group: FinitePermutationGroup,
    triangle_holonomies: Sequence[Permutation] | None = None,
    target_orders: Sequence[int] = (2, 3),
    max_depth: int = 4,
    max_exact_order: int = 64,
) -> SequentialQuotientChain:
    """Build an exact normal series by recursively taking certified kernels."""

    original = group
    current = group
    current_holonomies = tuple(triangle_holonomies or group.elements)
    stages: list[QuotientChainStage] = []
    stopped_reason = "max_depth_reached"
    for depth in range(1, int(max_depth) + 1):
        certs = certified_cyclic_quotients(current, target_orders=target_orders, max_exact_order=max_exact_order)
        if not certs:
            stopped_reason = "no_certified_prime_quotient"
            break
        cert = certs[0]
        before = quotient_residual(current_holonomies, cert)

        if current.truncated or cert.kernel_order is None:
            stages.append(
                QuotientChainStage(
                    depth=depth,
                    source_group_order=current.order,
                    quotient=cert,
                    residual_before=float(before),
                    residual_after=float("nan"),
                    branch_multiplier=cert.quotient_order,
                    residual_group_order=None,
                    coset_count=None,
                    coset_action_law_residual=None,
                    coset_action_kernel_order=None,
                    stabilizer_matches_kernel=None,
                    final_regular_representation_verified=False,
                )
            )
            stopped_reason = "truncated_sign_only_no_recursive_kernel"
            return SequentialQuotientChain(
                group_name="custom",
                group_order=group.order,
                closure_status=group.closure_status,
                truncated=group.truncated,
                stages=tuple(stages),
                final_kernel_order=None,
                stopped_reason=stopped_reason,
            )

        next_kernel = tuple(cert.kernel)
        next_group = subgroup_from_elements(next_kernel, current)
        after_holonomies = tuple(element for element in current_holonomies if cert.assignment.get(element, 0) % cert.quotient_order == 0)
        after = triangle_residual(after_holonomies, next_group)
        coset_rep = coset_action_representation(original, next_kernel)
        branch_multiplier = len(coset_rep.cosets)
        stages.append(
            QuotientChainStage(
                depth=depth,
                source_group_order=current.order,
                quotient=cert,
                residual_before=float(before),
                residual_after=float(after),
                branch_multiplier=int(branch_multiplier),
                residual_group_order=next_group.order,
                coset_count=len(coset_rep.cosets),
                coset_action_law_residual=float(coset_rep.law_residual),
                coset_action_kernel_order=int(coset_rep.kernel_order),
                stabilizer_matches_kernel=bool(coset_rep.stabilizer_matches_subgroup),
                final_regular_representation_verified=bool(coset_rep.permutation_conjugate_to_regular),
            )
        )
        current = next_group
        current_holonomies = after_holonomies
        if current.order <= 1:
            stopped_reason = "kernel_trivial"
            break
    return SequentialQuotientChain(
        group_name="custom",
        group_order=group.order,
        closure_status=group.closure_status,
        truncated=group.truncated,
        stages=tuple(stages),
        final_kernel_order=current.order,
        stopped_reason=stopped_reason,
    )


def cyclic_regular_branch_permutation(q: int, residue: int) -> tuple[int, ...]:
    q = int(q)
    r = int(residue) % q
    return tuple((idx + r) % q for idx in range(q))


def c2_fourier_components(branch_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(branch_logits, dtype=float)
    if arr.shape[-2] != 2:
        raise ValueError("C2 Fourier components require exactly two branches")
    z_plus = 0.5 * (arr[..., 0, :] + arr[..., 1, :])
    z_minus = 0.5 * (arr[..., 0, :] - arr[..., 1, :])
    return z_plus, z_minus


def uniform_pool(branch_logits: np.ndarray) -> np.ndarray:
    return np.asarray(branch_logits, dtype=float).mean(axis=-2)


def fourier_pool_c2(branch_logits: np.ndarray, minus_weight: float = 1.0) -> np.ndarray:
    z_plus, z_minus = c2_fourier_components(branch_logits)
    return z_plus + float(minus_weight) * z_minus


def validation_select_weight(candidates: Mapping[str, np.ndarray], labels: np.ndarray) -> tuple[str, float]:
    best_name = ""
    best_acc = -np.inf
    for name, logits in candidates.items():
        acc = accuracy(logits, labels)
        if acc > best_acc:
            best_name = str(name)
            best_acc = float(acc)
    return best_name, best_acc


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    pred = np.asarray(logits).argmax(axis=-1)
    lab = np.asarray(labels, dtype=int)
    return float(np.mean(pred == lab)) if lab.size else float("nan")


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    arr = np.asarray(logits, dtype=float)
    lab = np.asarray(labels, dtype=int)
    if arr.size == 0 or lab.size == 0:
        return float("nan")
    shifted = arr - arr.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(lab.size), lab]))


def measured_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {"accuracy": accuracy(logits, labels), "loss": cross_entropy(logits, labels)}


def branch_logits_from_models(branch_logits: Sequence[np.ndarray]) -> np.ndarray:
    """Stack already-evaluated branch logits as ``samples x branches x classes``."""

    arrays = [np.asarray(logits, dtype=float) for logits in branch_logits]
    if not arrays:
        raise ValueError("at least one branch logit array is required")
    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays):
        raise ValueError("all branch logit arrays must have the same shape")
    return np.stack(arrays, axis=1)


def label_permutation_logit_invariance(
    logit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    permuted_labels: np.ndarray,
) -> float:
    """Return max logit change when only labels are permuted before readout.

    This regression helper catches the old failure mode where candidate logits
    were synthesized from labels.  A genuine model/branch-logit builder should
    ignore labels before validation routing or readout training.
    """

    logits_a = np.asarray(logit_fn(np.asarray(inputs), np.asarray(labels)), dtype=float)
    logits_b = np.asarray(logit_fn(np.asarray(inputs), np.asarray(permuted_labels)), dtype=float)
    if logits_a.shape != logits_b.shape:
        return float("inf")
    return float(np.max(np.abs(logits_a - logits_b))) if logits_a.size else 0.0


def hidden_permutation_preservation_error(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    permutation: Sequence[int],
) -> float:
    """Check exact ReLU hidden-unit permutation symmetry for a one-hidden MLP."""

    perm = np.asarray(permutation, dtype=int)

    def relu(value: np.ndarray) -> np.ndarray:
        return np.maximum(value, 0.0)

    original = relu(np.asarray(x) @ np.asarray(w1) + np.asarray(b1)) @ np.asarray(w2) + np.asarray(b2)
    w1_perm = np.asarray(w1)[:, perm]
    b1_perm = np.asarray(b1)[perm]
    w2_perm = np.asarray(w2)[perm, :]
    permuted = relu(np.asarray(x) @ w1_perm + b1_perm) @ w2_perm + np.asarray(b2)
    return float(np.max(np.abs(original - permuted)))


def named_group(name: str) -> FinitePermutationGroup:
    key = str(name).lower().replace(" ", "")
    if key in {"c2", "cyclic2"}:
        return cyclic_group(2)
    if key in {"c3", "cyclic3"}:
        return cyclic_group(3)
    if key in {"c4", "cyclic4"}:
        return cyclic_group(4)
    if key in {"c2xc2", "v4", "klein"}:
        return klein_four_group()
    if key in {"d4", "dihedral4"}:
        return dihedral_group_4()
    if key in {"s3", "symmetric3"}:
        return symmetric_group_3()
    raise ValueError(f"unknown controlled group: {name}")


def infer_group_from_transitions(
    pairwise: Mapping[tuple[int, int], Sequence[int]],
    triangle_holonomies: Iterable[Sequence[int]],
    max_group_order: int = 5000,
    max_generators: int = 12,
) -> FinitePermutationGroup:
    generators: list[Permutation] = []
    for source in (triangle_holonomies, pairwise.values()):
        for candidate in source:
            try:
                perm = normalize_permutation(candidate)
            except Exception:
                continue
            if perm == identity_permutation(len(perm)):
                continue
            if perm not in generators:
                generators.append(perm)
            if len(generators) >= int(max_generators):
                break
        if len(generators) >= int(max_generators):
            break
    if not generators:
        width = len(next(iter(pairwise.values()))) if pairwise else 1
        generators = [identity_permutation(width)]
    return close_permutation_group(generators, max_group_order=max_group_order)


def _chain_signature(chain: SequentialQuotientChain) -> tuple[tuple[int, int | None], ...]:
    return tuple((stage.quotient.quotient_order, stage.residual_group_order) for stage in chain.stages)


def bootstrap_chain_stability(
    group: FinitePermutationGroup,
    triangle_holonomies: Sequence[Permutation],
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, float | str | int]:
    """Resample holonomies, rebuild the closure and quotient chain, and compare.

    This is an empirical recovery check, not a proof.  It intentionally returns
    values below one when the supplied relation/holonomy sample is too small to
    reliably regenerate the same quotient series.
    """

    holonomies = tuple(normalize_permutation(element) for element in triangle_holonomies)
    if not holonomies:
        return {
            "bootstrap_stability": float("nan"),
            "bootstrap_kernel_stability": float("nan"),
            "modal_chain": "none",
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_method": "resample_holonomies_rebuild_group",
        }
    target = build_successive_quotient_chain(group, holonomies)
    target_signature = _chain_signature(target)
    target_final_kernel = target.final_kernel_order
    rng = np.random.default_rng(seed)
    signatures: list[tuple[tuple[int, int | None], ...]] = []
    final_kernels: list[int | None] = []
    max_order = max(group.order * 4, group.order + 4, 8)
    for _ in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, len(holonomies), size=len(holonomies))
        sampled = tuple(holonomies[int(idx)] for idx in sample_idx)
        generators = [element for element in sampled if element != identity_permutation(len(element))]
        if not generators:
            generators = [identity_permutation(len(holonomies[0]))]
        sampled_group = close_permutation_group(generators, max_group_order=max_order)
        sampled_chain = build_successive_quotient_chain(sampled_group, sampled, max_depth=len(target.stages) or 4)
        signatures.append(_chain_signature(sampled_chain))
        final_kernels.append(sampled_chain.final_kernel_order)
    counts = Counter(signatures)
    modal_signature, modal_count = counts.most_common(1)[0]
    signature_matches = sum(1 for signature in signatures if signature == target_signature)
    kernel_matches = sum(1 for kernel in final_kernels if kernel == target_final_kernel)
    modal_chain = "->".join(f"C{q}/K{k}" for q, k in modal_signature) or "none"
    return {
        "bootstrap_stability": float(signature_matches / max(1, len(signatures))),
        "bootstrap_kernel_stability": float(kernel_matches / max(1, len(final_kernels))),
        "modal_chain": modal_chain,
        "modal_chain_count": int(modal_count),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_method": "resample_holonomies_rebuild_group",
    }
