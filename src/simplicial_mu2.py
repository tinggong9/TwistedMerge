"""Explicit mu_2 cocycles on a small 2-dimensional simplicial complex.

The main example is the boundary of a tetrahedron, a triangulated S^2 with
H^2(S^2, mu_2) ~= mu_2.  A nontrivial class can be represented by assigning
-1 to exactly one face and +1 to the other three faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np


Vertex = int
Edge = tuple[int, int]
Face = tuple[int, int, int]


def canonical_edge(i: int, j: int) -> Edge:
    return (i, j) if i < j else (j, i)


def canonical_face(face: Face) -> Face:
    return tuple(sorted(face))  # type: ignore[return-value]


@dataclass(frozen=True)
class SimplicialComplex:
    vertices: tuple[Vertex, ...]
    edges: tuple[Edge, ...]
    faces: tuple[Face, ...]


@dataclass(frozen=True)
class Mu2TransitionSystem:
    """Pairwise edge transitions plus a central face twist.

    `edge_signs` stores the ordinary local pairwise alignments.  `face_twist`
    stores the prescribed central 2-cocycle.  The triangle defect is

        edge_sign(i,j) edge_sign(j,k) edge_sign(k,i) face_twist(i,j,k) I.

    If `face_twist` is non-coboundary, this is twisted descent data rather than
    an ordinary vector-bundle transition system.
    """

    complex: SimplicialComplex
    edge_signs: Mapping[Edge, int]
    rank: int
    face_twist: Mapping[Face, int]

    def edge_sign(self, i: int, j: int) -> int:
        return int(self.edge_signs[canonical_edge(i, j)])

    def transition_matrix(self, i: int, j: int) -> np.ndarray:
        return self.edge_sign(i, j) * np.eye(self.rank)


@dataclass(frozen=True)
class LinearLocalModel:
    weight: np.ndarray
    bias: float = 0.0

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.logits(x) >= 0.0).astype(np.int64)


@dataclass(frozen=True)
class TwistedSheafPredictor:
    base_weight: np.ndarray
    twist: Mapping[Face, int]
    can_absorb_twist: bool

    def predict(self, face: Face, x: np.ndarray) -> np.ndarray:
        sign = int(self.twist[canonical_face(face)]) if self.can_absorb_twist else 1
        return (sign * (x @ self.base_weight) >= 0.0).astype(np.int64)


def tetrahedral_sphere() -> SimplicialComplex:
    vertices = (0, 1, 2, 3)
    edges = tuple(canonical_edge(i, j) for i in vertices for j in vertices if i < j)
    faces: tuple[Face, ...] = (
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    )
    return SimplicialComplex(vertices=vertices, edges=edges, faces=faces)


def trivial_mu2_twist(complex_: SimplicialComplex) -> dict[Face, int]:
    return {canonical_face(face): 1 for face in complex_.faces}


def nontrivial_tetrahedral_mu2_twist(complex_: SimplicialComplex) -> dict[Face, int]:
    twist = trivial_mu2_twist(complex_)
    twist[canonical_face(complex_.faces[0])] = -1
    return twist


def make_mu2_transition_system(
    complex_: SimplicialComplex,
    rank: int,
    twist: Mapping[Face, int] | None = None,
    edge_signs: Mapping[Edge, int] | None = None,
) -> Mu2TransitionSystem:
    edge_payload = {edge: 1 for edge in complex_.edges}
    if edge_signs is not None:
        edge_payload.update({canonical_edge(*edge): int(sign) for edge, sign in edge_signs.items()})
    face_payload = trivial_mu2_twist(complex_)
    if twist is not None:
        face_payload.update({canonical_face(face): int(sign) for face, sign in twist.items()})
    return Mu2TransitionSystem(
        complex=complex_,
        edge_signs=edge_payload,
        rank=rank,
        face_twist=face_payload,
    )


def compute_triangle_defects(g: Mu2TransitionSystem) -> dict[Face, np.ndarray]:
    defects: dict[Face, np.ndarray] = {}
    for face in g.complex.faces:
        i, j, k = face
        sign = g.edge_sign(i, j) * g.edge_sign(j, k) * g.edge_sign(k, i)
        sign *= int(g.face_twist[canonical_face(face)])
        defects[canonical_face(face)] = sign * np.eye(g.rank)
    return defects


def triangle_defect_signs(g: Mu2TransitionSystem) -> dict[Face, int]:
    signs: dict[Face, int] = {}
    for face, matrix in compute_triangle_defects(g).items():
        signs[face] = 1 if float(np.trace(matrix)) >= 0.0 else -1
    return signs


def coboundary_witness_mu2(
    cocycle: Mapping[Face, int],
    complex_: SimplicialComplex,
) -> dict[Edge, int] | None:
    """Return an edge 1-cochain whose coboundary equals `cocycle`, if it exists."""
    face_signs = {canonical_face(face): int(sign) for face, sign in cocycle.items()}
    for assignment in product((-1, 1), repeat=len(complex_.edges)):
        edge_signs = dict(zip(complex_.edges, assignment, strict=True))
        matched = True
        for face in complex_.faces:
            i, j, k = face
            value = (
                edge_signs[canonical_edge(i, j)]
                * edge_signs[canonical_edge(j, k)]
                * edge_signs[canonical_edge(k, i)]
            )
            if value != face_signs[canonical_face(face)]:
                matched = False
                break
        if matched:
            return edge_signs
    return None


def is_coboundary_mu2(cocycle: Mapping[Face, int], complex_: SimplicialComplex) -> bool:
    return coboundary_witness_mu2(cocycle, complex_) is not None


def obstruction_score(g: Mu2TransitionSystem) -> float:
    signs = triangle_defect_signs(g)
    return float(np.mean([sign < 0 for sign in signs.values()])) if signs else 0.0


def try_global_gauge_synchronization(g: Mu2TransitionSystem) -> dict[str, object]:
    """Try to remove the face defects by a mu_2-valued edge gauge.

    In H^2 language this solves delta b = c.  It succeeds for coboundary face
    defects and fails for the nontrivial tetrahedral-sphere class.
    """
    signs = triangle_defect_signs(g)
    witness = coboundary_witness_mu2(signs, g.complex)
    return {
        "success": witness is not None,
        "edge_correction": witness,
        "obstruction_score": obstruction_score(g),
        "is_coboundary": witness is not None,
        "negative_faces": sum(1 for sign in signs.values() if sign < 0),
        "triangle_defects": signs,
    }


def binary_zero_one_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(pred.astype(np.int64) != target.astype(np.int64)))


def pairwise_alignment_loss(
    local_models: Mapping[int, LinearLocalModel],
    transitions: Mu2TransitionSystem,
) -> float:
    losses = []
    for i, j in transitions.complex.edges:
        aligned = transitions.edge_sign(i, j) * local_models[j].weight
        denom = max(float(np.linalg.norm(local_models[i].weight) ** 2), 1e-12)
        losses.append(float(np.linalg.norm(local_models[i].weight - aligned) ** 2 / denom))
    return float(np.mean(losses)) if losses else 0.0


def ordinary_global_prediction(
    local_models: Mapping[int, LinearLocalModel],
) -> LinearLocalModel:
    weights = np.stack([model.weight for _, model in sorted(local_models.items())], axis=0)
    return LinearLocalModel(weight=weights.mean(axis=0))


def twisted_sheaf_prediction(
    local_models: Mapping[int, LinearLocalModel],
    transitions: Mu2TransitionSystem,
    twist: Mapping[Face, int],
) -> TwistedSheafPredictor:
    """Build the synthetic twisted predictor.

    A nontrivial mu_2 class cannot be absorbed by the rank-1 ordinary branch in
    this toy model.  Rank >= 2 is treated as the doubled representation: the
    predictor can carry both sign sectors and select the sector prescribed by
    the face twist.
    """
    base = ordinary_global_prediction(local_models).weight
    twist_is_coboundary = is_coboundary_mu2(twist, transitions.complex)
    can_absorb_twist = transitions.rank >= 2 or twist_is_coboundary
    return TwistedSheafPredictor(
        base_weight=base,
        twist={canonical_face(face): int(sign) for face, sign in twist.items()},
        can_absorb_twist=can_absorb_twist,
    )
