from __future__ import annotations

import numpy as np
import torch

from src.holonomy_brauer_certificate import (
    coboundary_fit,
    gauge_transform_connection,
    nearest_root,
    scalar_centrality,
    tetrahedral_cocycle_rows,
    triangle_defect,
)


def orthogonal(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(dimension, dimension, generator=generator)).Q


def test_scalar_and_root_detectors_recognize_minus_identity() -> None:
    matrix = -torch.eye(6)
    centrality = scalar_centrality(matrix)
    root = nearest_root(matrix)
    assert centrality.centrality_residual < 1e-12
    assert root.order == 2 and root.exponent == 1
    assert root.residual < 1e-12


def test_noncentral_matrix_is_rejected_by_scalar_residual() -> None:
    matrix = torch.diag(torch.tensor([1.0, 1.0, -1.0, -1.0]))
    assert scalar_centrality(matrix).centrality_residual > 0.9


def test_trivial_connection_is_cocycle_and_coboundary() -> None:
    frames = [orthogonal(5, seed) for seed in range(4)]
    transitions = {
        (source, target): frames[target] @ frames[source].T
        for source in range(4)
        for target in range(4)
        if source != target
    }
    phases = {}
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                defect = triangle_defect(transitions, (i, j, k))
                diagnostics = scalar_centrality(defect)
                phases[(i, j, k)] = float(
                    np.angle(complex(diagnostics.scalar_real, diagnostics.scalar_imag))
                )
    rows = tetrahedral_cocycle_rows(phases, vertices=4)
    residual, _rephasings = coboundary_fit(phases, vertices=4)
    assert max(float(row["normalized_cocycle_residual"]) for row in rows) < 1e-6
    assert residual < 1e-6


def test_centrality_is_invariant_under_local_orthogonal_gauge() -> None:
    frames = [orthogonal(5, seed) for seed in range(3)]
    transitions = {
        (source, target): frames[target] @ frames[source].T
        for source in range(3)
        for target in range(3)
        if source != target
    }
    before = scalar_centrality(triangle_defect(transitions, (0, 1, 2)))
    changed = gauge_transform_connection(transitions, [orthogonal(5, seed + 20) for seed in range(3)])
    after = scalar_centrality(triangle_defect(changed, (0, 1, 2)))
    assert abs(before.centrality_residual - after.centrality_residual) < 1e-6
    assert abs(before.scalar_real - after.scalar_real) < 1e-5
