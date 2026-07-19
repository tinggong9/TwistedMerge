from __future__ import annotations

import numpy as np

from src.lineage_transport_sync import (
    TRANSITION_METHODS,
    bootstrap_loop_identity_distance,
    identity_distance,
    inverse_consistency,
    loop_product,
    loop_statistics,
    normalized_residual,
    select_transition,
    synchronize_frames,
)


def random_orthogonal(dimension: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(dimension, dimension))
    left, _singular, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def test_transition_selection_and_inverse_consistency_on_exact_map() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(300, 8))
    expected = random_orthogonal(8, 11)
    target = source @ expected.T
    selected, fits, validation = select_transition(
        source[:200], target[:200], source[200:], target[200:]
    )
    assert set(fits) == set(TRANSITION_METHODS)
    assert selected.method in TRANSITION_METHODS
    assert validation[selected.method] < 1e-8
    assert normalized_residual(source, target, selected.matrix) < 1e-8
    reverse, _reverse_fits, _reverse_validation = select_transition(
        target[:200], source[:200], target[200:], source[200:]
    )
    assert inverse_consistency(selected.matrix, reverse.matrix) < 1e-7


def test_consistent_connection_synchronizes_and_has_identity_loop() -> None:
    nodes = ("M0", "M_A", "M_B", "M_AB")
    frames = {node: random_orthogonal(5, index + 20) for index, node in enumerate(nodes)}
    transitions = {
        (source, target): frames[target] @ frames[source].T
        for source in nodes
        for target in nodes
        if source != target
    }
    result = synchronize_frames(transitions, nodes)
    assert result.connection_residual < 1e-7
    loop = ("M0", "M_A", "M_AB", "M_B", "M0")
    product = loop_product(transitions, loop)
    assert identity_distance(product) < 1e-10
    stats = loop_statistics(product)
    np.testing.assert_allclose(stats["spectral_radius"], 1.0, atol=1e-10)


def test_loop_bootstrap_uses_shared_anchor_resamples() -> None:
    rng = np.random.default_rng(19)
    base = rng.normal(size=(120, 4))
    nodes = ("x", "y", "z")
    frames = {node: random_orthogonal(4, index + 30) for index, node in enumerate(nodes)}
    representations = {node: base @ frames[node].T for node in nodes}
    loop = ("x", "y", "z", "x")
    methods = {edge: "orthogonal_procrustes" for edge in zip(loop[:-1], loop[1:], strict=True)}
    mean, low, high, samples = bootstrap_loop_identity_distance(
        representations, loop, methods, samples=20, seed=4
    )
    assert len(samples) == 20
    assert 0.0 <= low <= mean <= high < 1e-6
