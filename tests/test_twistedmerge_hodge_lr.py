from __future__ import annotations

import numpy as np

from src.twistedmerge_hodge_lr import (
    conservative_confidence_gate,
    cycle_residual,
    dispatch_correction,
    estimate_transition,
    inverse_consistency,
    weighted_hodge_decomposition,
)


def test_transition_recovers_permutation_and_inverse() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(400, 4))
    permutation = np.eye(4)[[2, 0, 3, 1]]
    target = source @ permutation.T
    forward = estimate_transition(source, target, family="permutation")
    reverse = estimate_transition(target, source, family="permutation")
    assert np.allclose(forward.matrix, permutation)
    assert forward.calibration_error < 1e-12
    assert inverse_consistency(forward.matrix, reverse.matrix) < 1e-12


def test_cycle_and_dispatch_reject_uncertified_residual() -> None:
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = cycle_residual(rotation, rotation)
    assert result.distance_to_identity > 1.0
    gate = conservative_confidence_gate([0.2, 0.3, 0.25, 0.4])
    decision = dispatch_correction(residual_norm=1.0, harmonic_norm=1.0, gate=gate)
    assert not decision.activate_lift
    assert decision.mode == "ordinary_validated_family"


def test_weighted_hodge_reconstructs_and_is_orthogonal() -> None:
    b1 = np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]], dtype=float)
    b2 = np.array([[1], [1], [1]], dtype=float)
    values = np.array([0.8, -0.1, 0.4])
    result = weighted_hodge_decomposition(b1, b2, values, edge_weights=[1, 2, 3])
    assert result.reconstruction_error < 1e-10
    assert abs(result.gradient_harmonic_inner) < 1e-10
    assert abs(result.gradient_coexact_inner) < 1e-10
    assert abs(result.harmonic_coexact_inner) < 1e-10
