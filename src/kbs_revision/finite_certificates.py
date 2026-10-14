"""Controlled finite certificates; no learned transitions or neural results.

Transport matrices map column vectors from source to target.  Thus the loop
``i -> j -> k -> i`` is ``g_ki @ g_jk @ g_ij``.  All matrix certificates below
use integers, whereas Hodge identities use separate real vector coefficients.
The only implemented finite coefficient theory is mu_2 = F_2.  Unsupported
coefficient theories are explicitly inapplicable, never treated as real signs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


class CertificateVerificationError(RuntimeError):
    """A required measured construction or identity check failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificateVerificationError(message)


def _integers(value: np.ndarray | Sequence, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biu":
        raise ValueError(f"{name} must contain exact integers")
    return array.astype(np.int64, copy=True)


def gf2_rank(matrix: np.ndarray | Sequence) -> int:
    """Exact binary rank, including empty rectangular matrices."""
    array = _integers(matrix, "matrix") % 2
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rank = 0
    for column in range(array.shape[1]):
        candidates = np.flatnonzero(array[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        array[[rank, pivot]] = array[[pivot, rank]]
        for row in range(rank + 1, array.shape[0]):
            if array[row, column]:
                array[row] ^= array[rank]
        rank += 1
        if rank == array.shape[0]:
            break
    return rank


def gf2_solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """Return an exact binary witness for matrix*x=target, or None."""
    lhs = _integers(matrix, "matrix") % 2
    rhs = _integers(target, "target") % 2
    if lhs.ndim != 2 or rhs.shape != (lhs.shape[0],):
        raise ValueError("matrix and target dimensions do not agree")
    augmented = np.column_stack((lhs, rhs))
    pivots = []
    row = 0
    for column in range(lhs.shape[1]):
        candidates = np.flatnonzero(augmented[row:, column])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        augmented[[row, pivot]] = augmented[[pivot, row]]
        for other in range(augmented.shape[0]):
            if other != row and augmented[other, column]:
                augmented[other] ^= augmented[row]
        pivots.append(column)
        row += 1
        if row == lhs.shape[0]:
            break
    if np.any(np.logical_and(~augmented[:, :-1].any(axis=1), augmented[:, -1])):
        return None
    witness = np.zeros(lhs.shape[1], dtype=np.int64)
    for index, column in enumerate(pivots):
        witness[column] = augmented[index, -1]
    return witness


@dataclass(frozen=True)
class SimplicialComplex:
    """An explicitly stored, downward-closed oriented simplicial complex."""

    simplices: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        for degree, level in enumerate(self.simplices):
            if len(set(level)) != len(level):
                raise ValueError("duplicate simplices")
            for simplex in level:
                if len(simplex) != degree + 1 or tuple(sorted(set(simplex))) != simplex:
                    raise ValueError("simplices must have distinct sorted vertices")
                if degree and any(face not in self.simplices[degree - 1] for face in combinations(simplex, degree)):
                    raise ValueError("complex is not downward closed")

    def level(self, degree: int) -> tuple[tuple[int, ...], ...]:
        return self.simplices[degree] if 0 <= degree < len(self.simplices) else ()

    def delta(self, degree: int) -> np.ndarray:
        """Signed simplicial coboundary C^degree -> C^(degree+1)."""
        source = self.level(degree)
        target = self.level(degree + 1)
        index = {simplex: column for column, simplex in enumerate(source)}
        matrix = np.zeros((len(target), len(source)), dtype=np.int64)
        for row, simplex in enumerate(target):
            for omitted in range(len(simplex)):
                matrix[row, index[simplex[:omitted] + simplex[omitted + 1:]]] = (-1) ** omitted
        return matrix

    def betti_f2(self) -> list[int]:
        ranks = [gf2_rank(self.delta(degree)) for degree in range(len(self.simplices))]
        return [len(level) - ranks[degree] - (ranks[degree - 1] if degree else 0)
                for degree, level in enumerate(self.simplices)]

    def to_json(self) -> dict:
        return {"simplices_by_degree": [[list(simplex) for simplex in level] for level in self.simplices],
                "signed_coboundaries": [self.delta(degree).tolist() for degree in range(len(self.simplices))]}


def _from_maximal(maximal: Sequence[Sequence[int]]) -> SimplicialComplex:
    if not maximal:
        raise ValueError("at least one maximal simplex is required")
    dimension = max(len(simplex) for simplex in maximal) - 1
    levels = [set() for _ in range(dimension + 1)]
    for simplex in maximal:
        canonical = tuple(sorted(simplex))
        if len(set(canonical)) != len(canonical):
            raise ValueError("repeated vertex")
        for size in range(1, len(canonical) + 1):
            levels[size - 1].update(combinations(canonical, size))
    return SimplicialComplex(tuple(tuple(sorted(level)) for level in levels))


def full_simplex(vertices: int = 4) -> SimplicialComplex:
    if vertices < 1:
        raise ValueError("vertices must be positive")
    return _from_maximal([tuple(range(vertices))])


def tetrahedral_boundary() -> SimplicialComplex:
    return _from_maximal(list(combinations(range(4), 3)))


I = np.eye(2, dtype=np.int64)
X = np.array([[0, 1], [1, 0]], dtype=np.int64)
Z = np.array([[1, 0], [0, -1]], dtype=np.int64)


def _inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = _integers(matrix, "transport")
    if matrix.shape != (2, 2):
        raise ValueError("transport must be a 2x2 integer matrix")
    a, b, c, d = map(int, matrix.ravel())
    determinant = a * d - b * c
    if abs(determinant) != 1:
        raise ValueError("exact transport/gauge must be unimodular")
    return np.array([[d, -b], [-c, a]], dtype=np.int64) * determinant


def build_torus(size: int = 3, *, trivial_control: bool = False) -> tuple[SimplicialComplex, dict, dict]:
    """Periodic SW-to-NE grid, with genuine stored edge transports.

    Returns (complex, directed_transports, construction_metadata).  Both
    directions are stored and all face defects must be computed from them.
    """
    if size < 3:
        raise ValueError("size >= 3 is required for a simplicial periodic grid")
    vertex = lambda x, y: (y % size) * size + (x % size)
    directed: dict[tuple[int, int], np.ndarray] = {}
    native_edges = []
    oriented_faces = []

    def add(source: int, target: int, matrix: np.ndarray, kind: str) -> None:
        if (source, target) in directed or (target, source) in directed:
            raise ValueError("periodic grid generated a repeated unoriented edge")
        directed[(source, target)] = matrix.copy()
        directed[(target, source)] = _inverse(matrix)
        native_edges.append({"source": source, "target": target, "kind": kind,
                             "matrix": matrix.tolist(), "reverse_matrix": directed[(target, source)].tolist()})

    vertical_seam = I if trivial_control else Z
    for y in range(size):
        for x in range(size):
            p, h, v, diagonal = vertex(x, y), vertex(x + 1, y), vertex(x, y + 1), vertex(x + 1, y + 1)
            horizontal = X if x == size - 1 else I
            vertical = vertical_seam if y == size - 1 else I
            add(p, h, horizontal, "horizontal")
            add(p, v, vertical, "vertical")
            add(p, diagonal, vertical @ horizontal, "diagonal_horizontal_then_vertical")
            oriented_faces.extend([(p, h, diagonal), (p, diagonal, v)])
    complex_ = _from_maximal(oriented_faces)
    metadata = {"size": size, "coordinates": [{"vertex": vertex(x, y), "x": x, "y": y}
                                               for y in range(size) for x in range(size)],
                "transport_convention": "column vectors; loop i-j-k-i = g_ki @ g_jk @ g_ij",
                "diagonal_convention": "horizontal then vertical: g_pd = g_hd @ g_ph",
                "native_edges": native_edges, "oriented_grid_faces": [list(face) for face in oriented_faces],
                "X": X.tolist(), "Z": vertical_seam.tolist(), "trivial_control": trivial_control}
    return complex_, directed, metadata


def face_cochain(complex_: SimplicialComplex, edges: Mapping) -> tuple[np.ndarray, list]:
    """Compute mu_2 labels from exact stored edge products; reject noncentrality."""
    cochain = []
    residuals = []
    for i, j, k in complex_.level(2):
        residual = edges[(k, i)] @ edges[(j, k)] @ edges[(i, j)]
        if np.array_equal(residual, I):
            label = 0
        elif np.array_equal(residual, -I):
            label = 1
        else:
            raise CertificateVerificationError(f"face {(i, j, k)} does not have an exact mu_2 residual")
        cochain.append(label)
        residuals.append({"face": [i, j, k], "matrix": residual.tolist(), "mu2_exponent": label})
    return np.asarray(cochain, dtype=np.int64), residuals


def _connection_checks(complex_: SimplicialComplex, edges: Mapping) -> dict:
    required = {(i, j) for i, j in complex_.level(1)} | {(j, i) for i, j in complex_.level(1)}
    unique = set(edges) == required
    inverse = unique and all(np.array_equal(edges[(j, i)] @ edges[(i, j)], I) for i, j in complex_.level(1))
    chain = all(not np.any(complex_.delta(degree + 1) @ complex_.delta(degree))
                for degree in range(len(complex_.simplices) - 1))
    return {"stored_edges_exactly_match_complex": unique, "inverse_consistent": inverse,
            "signed_delta_squared_zero": chain, "f2_delta_squared_zero": chain}


def _surface_checks(complex_: SimplicialComplex) -> dict:
    incidence = np.abs(complex_.delta(1)).sum(axis=0)
    links = []
    for (vertex,) in complex_.level(0):
        link_edges = [tuple(other for other in face if other != vertex)
                      for face in complex_.level(2) if vertex in face]
        neighbors = {other for edge in link_edges for other in edge}
        adjacency = {other: set() for other in neighbors}
        for left, right in link_edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        reached = set()
        pending = [next(iter(neighbors))] if neighbors else []
        while pending:
            current = pending.pop()
            if current not in reached:
                reached.add(current)
                pending.extend(adjacency[current] - reached)
        cycle = bool(neighbors) and reached == neighbors and all(len(value) == 2 for value in adjacency.values())
        links.append({"vertex": vertex, "link_edges": [list(edge) for edge in link_edges], "is_connected_cycle": cycle})
    return {"each_edge_incident_to_two_faces": bool(np.all(incidence == 2)),
            "edge_incidence_counts": incidence.tolist(), "vertex_links_are_cycles": all(link["is_connected_cycle"] for link in links),
            "vertex_links": links}


def _random_gauges(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    gauges = []
    for _ in range(count):
        upper = np.array([[1, int(rng.integers(-3, 4))], [0, 1]], dtype=np.int64)
        lower = np.array([[1, 0], [int(rng.integers(-3, 4)), 1]], dtype=np.int64)
        gauges.append(upper @ lower @ (X if rng.integers(2) else I) @ (-I if rng.integers(2) else I))
    return gauges


def _gauged(edges: Mapping, gauges: Sequence[np.ndarray]) -> dict:
    return {(i, j): gauges[j] @ matrix @ _inverse(gauges[i]) for (i, j), matrix in edges.items()}


def _central_perturb(complex_: SimplicialComplex, edges: Mapping, beta: np.ndarray) -> dict:
    signs = {edge: (-1) ** int(beta[column]) for column, edge in enumerate(complex_.level(1))}
    return {(i, j): signs[tuple(sorted((i, j)))] * matrix for (i, j), matrix in edges.items()}


def verify_torus(size: int = 3, *, trivial_control: bool = False) -> dict:
    complex_, edges, construction = build_torus(size, trivial_control=trivial_control)
    checks = _connection_checks(complex_, edges)
    checks.update(_surface_checks(complex_))
    c, residuals = face_cochain(complex_, edges)
    checks["all_face_residuals_exactly_mu2"] = True
    checks["closed_exact_f2"] = bool(not np.any(complex_.delta(2) @ c % 2))
    delta1 = complex_.delta(1)
    rank = gf2_rank(delta1)
    augmented = gf2_rank(np.column_stack((delta1, c)))
    betti = complex_.betti_f2()
    witness = gf2_solve(delta1, c)
    expected_exact = trivial_control
    required = ["stored_edges_exactly_match_complex", "inverse_consistent", "signed_delta_squared_zero",
                "each_edge_incident_to_two_faces", "vertex_links_are_cycles"]
    for name in required:
        _require(checks[name], f"torus size {size}: {name} failed")
    _require([len(complex_.level(degree)) for degree in range(3)] == [size**2, 3*size**2, 2*size**2], "torus simplex counts")
    _require(betti == [1, 2, 1], "torus Betti numbers")
    _require(rank == 2*size**2 - 1 and augmented == rank + (not trivial_control), "torus F2 ranks")
    _require(int(c.sum()) == (0 if trivial_control else 1), "torus central negative-face count")
    _require((witness is not None) == expected_exact, "torus coboundary membership")
    perturbations = []
    for seed in (11, 29, 47):
        gauges = _random_gauges(size**2, seed)
        changed = _gauged(edges, gauges)
        changed_c, _ = face_cochain(complex_, changed)
        gauge_invariant = np.array_equal(c, changed_c)
        _require(gauge_invariant, "torus exact vertex-gauge invariance")
        beta = np.random.default_rng(seed).integers(0, 2, len(complex_.level(1)), dtype=np.int64)
        shifted = _central_perturb(complex_, changed, beta)
        shifted_c, shifted_residuals = face_cochain(complex_, shifted)
        cochain_law = np.array_equal(shifted_c, (c + delta1 @ beta) % 2)
        class_unchanged = (gf2_solve(delta1, shifted_c) is not None) == expected_exact
        perturb_checks = _connection_checks(complex_, shifted)
        _require(cochain_law and class_unchanged and all(perturb_checks.values()), "torus central edge-cochain perturbation")
        perturbations.append({"seed": seed, "gauges": [gauge.tolist() for gauge in gauges],
                              "central_edge_cochain": beta.tolist(), "face_cochain": shifted_c.tolist(),
                              "face_residuals": shifted_residuals, "gauge_invariant": bool(gauge_invariant),
                              "cochain_law_verified": bool(cochain_law), "cohomology_class_unchanged": bool(class_unchanged),
                              "connection_checks": perturb_checks})
    return {"status": "controlled_algebraic", "scope": "exact stored-edge mu2 cohomology; no neural performance claim",
            "construction": {**complex_.to_json(), **construction}, "checks": checks,
            "face_residuals": residuals, "face_cochain": c.tolist(), "negative_faces": int(c.sum()),
            "rank_delta1_f2": rank, "rank_augmented_f2": augmented, "betti_f2": betti,
            "exact_trivial": witness is not None, "coboundary_witness": None if witness is None else witness.tolist(),
            "perturbations": perturbations}


def verify_simply_connected_no_go() -> list[dict]:
    """Construction-certified cases, not a test equating H1=0 with pi1=0.

    These systems are generated from exact vertex lifts and central edges.
    Gauge stripping recovers the central edge witness directly.  The generic
    simply connected no-go is a mathematical implication, not inferred from
    these sampled systems or from binary Betti numbers.
    """
    cases = [("tetrahedral_boundary", tetrahedral_boundary(), "boundary of a tetrahedron realizes S^2; S^2 is simply connected"),
             ("full_tetrahedron", full_simplex(4), "full simplex is contractible by straight-line contraction to a vertex"),
             ("full_4_simplex", full_simplex(5), "full simplex is contractible by straight-line contraction to a vertex")]
    rows = []
    for name, complex_, topology_certificate in cases:
        trials = []
        for seed in (3, 17, 41):
            gauges = _random_gauges(len(complex_.level(0)), seed)
            beta = np.random.default_rng(seed).integers(0, 2, len(complex_.level(1)), dtype=np.int64)
            edges = {}
            for column, (i, j) in enumerate(complex_.level(1)):
                edges[(i, j)] = (-1)**int(beta[column]) * gauges[j] @ _inverse(gauges[i])
                edges[(j, i)] = _inverse(edges[(i, j)])
            c, residuals = face_cochain(complex_, edges)
            stripped = [_inverse(gauges[j]) @ edges[(i, j)] @ gauges[i] for i, j in complex_.level(1)]
            recovered = np.asarray([0 if np.array_equal(matrix, I) else 1 for matrix in stripped], dtype=np.int64)
            central = all(np.array_equal(matrix, I) or np.array_equal(matrix, -I) for matrix in stripped)
            witness = gf2_solve(complex_.delta(1), c)
            checks = _connection_checks(complex_, edges)
            checks.update({"stripped_edges_central": central,
                           "recovered_edge_witness_matches": bool(np.array_equal(recovered, beta)),
                           "central_coboundary_witness_verified": bool(np.array_equal(complex_.delta(1) @ recovered % 2, c)),
                           "closed_exact_f2": bool(not np.any(complex_.delta(2) @ c % 2)),
                           "exact_trivial": witness is not None})
            _require(all(checks.values()), f"simply connected construction {name} failed")
            trials.append({"seed": seed, "gauges": [gauge.tolist() for gauge in gauges], "edge_cochain": beta.tolist(),
                           "stored_edges": [{"source": i, "target": j, "matrix": matrix.tolist()} for (i, j), matrix in sorted(edges.items())],
                           "face_residuals": residuals, "face_cochain": c.tolist(), "checks": checks,
                           "coboundary_witness": witness.tolist()})
        row = {"case": name, "status": "controlled_algebraic", "construction": complex_.to_json(),
                     "simple_connectedness_certificate": topology_certificate, "certificate_basis": "construction",
                     "betti_f2": complex_.betti_f2(), "decision": "nontrivial_central_class_ruled_out_under_model",
                     "model_assumptions": ["constant coefficients", "context-independent inverse-consistent G edges",
                                           "central mu2 triangle residuals"],
                     "no_go_proof_steps": ["reduce the edge connection modulo central A to a flat G/A connection",
                                           "simple connectedness makes that flat connection a vertex coboundary",
                                           "lift vertex gauges to G and strip them from the edges",
                                           "remaining edges lie in A, so their face residual is an A coboundary"],
                     "neural_merge_safety_claim": False, "trials": trials}
        if name == "tetrahedral_boundary":
            oracle = np.array([1, 0, 0, 0], dtype=np.int64)
            row["oracle_face_assignment_control"] = {
                "construction_type": "oracle_face_assignment", "face_cochain": oracle.tolist(),
                "closed_exact_f2": bool(not np.any(complex_.delta(2) @ oracle % 2)),
                "rank_delta1_f2": gf2_rank(complex_.delta(1)),
                "rank_augmented_f2": gf2_rank(np.column_stack((complex_.delta(1), oracle))),
                "exact_trivial": gf2_solve(complex_.delta(1), oracle) is not None,
                "edge_realizability_asserted": False,
                "edge_realizability_under_stated_model": "ruled_out_by_simply_connected_no_go",
                "scope": "abstract input cochain, not stored-edge residuals or neural transitions"}
            _require(row["oracle_face_assignment_control"]["closed_exact_f2"]
                     and not row["oracle_face_assignment_control"]["exact_trivial"], "boundary oracle distinction")
        rows.append(row)
    return rows


def vector_hodge(complex_: SimplicialComplex, degree: int, values: np.ndarray,
                 weights: Mapping[int, Sequence[float]] | None = None) -> dict[str, np.ndarray | float]:
    """Weighted real-vector Hodge decomposition, with the same complex/metric.

    The three columns (if supplied) are independent vector coefficients.
    Returns gradient, harmonic and coexact cochains in original coordinates.
    It makes no statement about torsion or discrete sign coefficients.
    """
    if not 0 <= degree < len(complex_.simplices):
        raise ValueError("degree is outside the complex")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(complex_.level(degree)) or not np.isfinite(values).all():
        raise ValueError("finite cochain must match the chosen degree")
    weights = {} if weights is None else weights

    def metric(d: int) -> np.ndarray:
        size = len(complex_.level(d))
        array = np.asarray(weights.get(d, np.ones(size)), dtype=float)
        if array.shape != (size,) or not np.isfinite(array).all() or np.any(array <= 0):
            raise ValueError("cochain weights must be finite and positive with matching sizes")
        return array

    current = metric(degree)
    root = np.sqrt(current)[:, None]
    previous = complex_.delta(degree - 1) if degree else np.zeros((len(current), 0))
    following = complex_.delta(degree)
    if np.any(following @ previous):
        raise ValueError("coboundaries fail delta squared zero")
    gradient_basis = root * previous
    coexact_basis = following.T * np.sqrt(metric(degree + 1))[None, :] / root
    whitened = root * values

    def project(basis: np.ndarray) -> np.ndarray:
        return basis @ np.linalg.lstsq(basis, whitened, rcond=1e-12)[0] if basis.shape[1] else np.zeros_like(whitened)

    gradient = project(gradient_basis)
    coexact = project(coexact_basis)
    harmonic = whitened - gradient - coexact
    distance_squared = float(np.sum((whitened - gradient)**2))
    return {"gradient": gradient / root, "harmonic": harmonic / root, "coexact": coexact / root,
            "distance_squared": distance_squared, "harmonic_energy": float(np.sum(harmonic**2)),
            "coexact_energy": float(np.sum(coexact**2)), "closure_norm": float(np.linalg.norm(following @ values)),
            "gradient_harmonic_inner": float(np.sum(gradient * harmonic)),
            "gradient_coexact_inner": float(np.sum(gradient * coexact)),
            "harmonic_coexact_inner": float(np.sum(harmonic * coexact)),
            "reconstruction_error": float(np.linalg.norm((gradient + harmonic + coexact) / root - values))}


def verify_hodge() -> list[dict]:
    torus, _, _ = build_torus(3)
    complexes = [("torus_3", torus, [1, 2]), ("tetrahedral_boundary", tetrahedral_boundary(), [1, 2]),
                 ("full_tetrahedron", full_simplex(4), [1, 2, 3]), ("full_4_simplex", full_simplex(5), [1, 2, 3])]
    rng = np.random.default_rng(101)
    rows = []
    for name, complex_, degrees in complexes:
        weights = {degree: np.linspace(0.7, 1.9, len(level)) for degree, level in enumerate(complex_.simplices)}
        for degree in degrees:
            for metric_name, metric in [("identity", None), ("diagonal_weighted", weights)]:
                arbitrary = rng.normal(size=(len(complex_.level(degree)), 3))
                raw = vector_hodge(complex_, degree, arbitrary, metric)
                # Project to ker(delta_degree) with this same cochain metric.
                closed = raw["gradient"] + raw["harmonic"]
                for label, cochain in [("arbitrary", arbitrary), ("closed", closed)]:
                    decomposition = vector_hodge(complex_, degree, cochain, metric)
                    distance = float(decomposition["distance_squared"])
                    harmonic = float(decomposition["harmonic_energy"])
                    coexact = float(decomposition["coexact_energy"])
                    error = abs(distance - harmonic - coexact)
                    tolerance = 1e-10 * max(1.0, distance)
                    _require(error <= tolerance, f"Hodge arbitrary identity {name} degree {degree}")
                    _require(max(abs(float(decomposition[key])) for key in ["gradient_harmonic_inner", "gradient_coexact_inner", "harmonic_coexact_inner"]) <= tolerance,
                             f"Hodge orthogonality {name} degree {degree}")
                    if label == "closed":
                        _require(float(decomposition["closure_norm"]) < 1e-10 and abs(distance - harmonic) <= tolerance,
                                 f"Hodge closed identity {name} degree {degree}")
                    rows.append({"complex": name, "degree": degree, "coefficient_theory": "real_vector_dimension_3",
                                 "metric": metric_name, "cochain_kind": label, "status": "controlled_algebraic",
                                 "cochain": cochain.tolist(), "weights": None if metric is None else {str(d): w.tolist() for d, w in metric.items()},
                                 "distance_squared": distance, "harmonic_energy": harmonic, "coexact_energy": coexact,
                                 "arbitrary_identity_absolute_error": error,
                                 "closed_identity_absolute_error": abs(distance - harmonic) if label == "closed" else None,
                                 "closure_norm": float(decomposition["closure_norm"]),
                                 "reconstruction_error": float(decomposition["reconstruction_error"]),
                                 "tolerance": tolerance, "passed": True})
    return rows


def certificate_decision(delta1: np.ndarray, delta2: np.ndarray, cochain: np.ndarray, *,
                         coefficient_theory: str = "mu2", common_coordinate: bool = True,
                         overlap_model: bool = True, structural_prerequisites: bool = True,
                         confidence_prerequisites: bool = True, exact_model_verified: bool = False,
                         rounded: bool = False, error_radius: float | None = None,
                         coefficient_separation: float | None = None,
                         metric_bound_prerequisites: bool = False, repair_upper_bound: float | None = None,
                         separation_lower_bound: float | None = None, a: float = 0.1, b: float = 0.2) -> dict:
    """Separate discrete exact truth from qualified real defect bounds.

    Prerequisite booleans are caller attestations, never inferred from small
    residuals.  Rounded mu2 labels have an exact *rounded-cochain* check, but
    recover underlying truth only with an exact latent finite model and a
    certified radius strictly below half the coefficient separation.
    Approximate bounds use a separately justified metric/cochain construction.
    Unsupported mu_n arithmetic returns inapplicable rather than a real fit.
    """
    if not np.isfinite([a, b]).all() or a < 0 or b <= a:
        raise ValueError("thresholds must satisfy 0 <= a < b")
    for bound in (repair_upper_bound, separation_lower_bound):
        if bound is not None and (not np.isfinite(bound) or bound < 0):
            raise ValueError("bounds must be finite and nonnegative")
    if repair_upper_bound is not None and separation_lower_bound is not None and separation_lower_bound > repair_upper_bound:
        raise ValueError("lower bound cannot exceed upper bound")
    result = {"exact_trivial": None, "repairable_to_tolerance": None, "separated_from_coboundaries": None,
              "uncertified": False, "inapplicable": False, "exact_rounded_trivial": None,
              "exact_model_verified": exact_model_verified, "coefficient_theory": coefficient_theory,
              "prerequisites": {"common_coordinate": common_coordinate, "overlap_model": overlap_model,
                                "structural": structural_prerequisites, "confidence": confidence_prerequisites,
                                "metric_bound": metric_bound_prerequisites},
              "rounded": rounded, "error_radius": error_radius, "coefficient_separation": coefficient_separation,
              "a": a, "b": b, "repair_upper_bound": repair_upper_bound, "separation_lower_bound": separation_lower_bound,
              "reason": ""}
    if not common_coordinate or not overlap_model or coefficient_theory != "mu2":
        result.update(inapplicable=True, reason="common coordinate/overlap model absent or coefficient theory unimplemented")
        return result
    d1, d2, c = _integers(delta1, "delta1"), _integers(delta2, "delta2"), _integers(cochain, "cochain")
    if d1.ndim != 2 or d2.ndim != 2 or c.shape != (d1.shape[0],) or d2.shape[1] != len(c):
        raise ValueError("cochain/coboundary dimensions do not agree")
    if np.any(d2 @ d1 % 2):
        raise ValueError("F2 coboundaries do not form a complex")
    if np.any((c != 0) & (c != 1)):
        raise ValueError("mu2 exponents must be exactly 0 or 1")
    closed = not np.any(d2 @ c % 2)
    exact_membership = gf2_solve(d1, c) is not None
    result["closed_exact_f2"] = bool(closed)
    if rounded:
        result["exact_rounded_trivial"] = exact_membership if closed else None
    recovery = not rounded or (error_radius is not None and coefficient_separation is not None
                               and np.isfinite([error_radius, coefficient_separation]).all()
                               and 0 <= error_radius < coefficient_separation / 2)
    result["recovery_prerequisites_passed"] = bool(recovery)
    prerequisite_pass = structural_prerequisites and confidence_prerequisites and closed
    if prerequisite_pass and exact_model_verified and recovery:
        result["exact_trivial"] = exact_membership
    if prerequisite_pass and metric_bound_prerequisites:
        if repair_upper_bound is not None:
            result["repairable_to_tolerance"] = bool(repair_upper_bound <= a)
        if separation_lower_bound is not None:
            result["separated_from_coboundaries"] = bool(separation_lower_bound >= b)
    result["uncertified"] = not prerequisite_pass or (rounded and not (exact_model_verified and recovery)) or not (
        result["exact_trivial"] is not None or result["repairable_to_tolerance"] is not None or result["separated_from_coboundaries"] is not None)
    result["reason"] = ("qualified exact/bound fields evaluated separately" if not result["uncertified"]
                        else "structural, confidence, closure, exact-model or recovery prerequisites unverified")
    return result


def verify_decisions() -> dict:
    complex_, edges, _ = build_torus(3)
    c, _ = face_cochain(complex_, edges)
    d1, d2 = complex_.delta(1), complex_.delta(2)
    cases = {
        "exact_nontrivial_torus": certificate_decision(d1, d2, c, exact_model_verified=True),
        "exact_trivial_control": certificate_decision(d1, d2, np.zeros_like(c), exact_model_verified=True),
        "positive_tolerance_not_exact_zero": certificate_decision(d1, d2, np.zeros_like(c), metric_bound_prerequisites=True, repair_upper_bound=0.05),
        "separated_metric_bound": certificate_decision(d1, d2, c, metric_bound_prerequisites=True, separation_lower_bound=0.3),
        "rounded_without_latent_model": certificate_decision(d1, d2, c, rounded=True, error_radius=0.01, coefficient_separation=2),
        "rounded_recovery_verified": certificate_decision(d1, d2, c, rounded=True, exact_model_verified=True, error_radius=0.01, coefficient_separation=2),
        "rounded_at_half_separation": certificate_decision(d1, d2, c, rounded=True, exact_model_verified=True, error_radius=1, coefficient_separation=2),
        "failed_structural_prerequisite": certificate_decision(d1, d2, c, exact_model_verified=True, structural_prerequisites=False),
        "absent_overlap_model": certificate_decision(d1, d2, c, overlap_model=False),
        "unsupported_mun": certificate_decision(d1, d2, c, coefficient_theory="mu4"),
    }
    _require(cases["exact_nontrivial_torus"]["exact_trivial"] is False, "exact torus decision")
    _require(cases["exact_trivial_control"]["exact_trivial"] is True, "exact control decision")
    _require(cases["positive_tolerance_not_exact_zero"]["exact_trivial"] is None and cases["positive_tolerance_not_exact_zero"]["repairable_to_tolerance"] is True, "positive tolerance cannot certify exact zero")
    _require(cases["separated_metric_bound"]["separated_from_coboundaries"] is True, "separation decision")
    _require(cases["rounded_without_latent_model"]["uncertified"] and cases["rounded_without_latent_model"]["exact_trivial"] is None, "rounded labels alone cannot establish latent model")
    _require(cases["rounded_recovery_verified"]["exact_trivial"] is False, "finite recovery prerequisites")
    _require(cases["rounded_at_half_separation"]["uncertified"], "half separation must be strict")
    _require(cases["failed_structural_prerequisite"]["uncertified"], "structural prerequisite")
    _require(cases["absent_overlap_model"]["inapplicable"] and cases["unsupported_mun"]["inapplicable"], "inapplicable decisions")
    return cases


def gate_ablation_controls() -> list[dict]:
    """Measured negative gate fixtures; no accuracy, training or latent claims.

    Raw face matrices are retained before nearest-sign projection.  Gate flags
    describe actual checks and can fail together.  A cochain fed directly to a
    closure test is explicitly distinguished from stored-edge realizations.
    Unsupported coefficient models and absent common coordinates are
    inapplicable even when a projected cochain has a binary membership result.
    """
    gate_names = ("inverse_consistency", "coefficient_identification", "projection_fidelity",
                  "centrality", "closure", "uncertainty_margin", "topology")

    def flat_edges(complex_: SimplicialComplex) -> dict:
        return {directed: I.copy() for i, j in complex_.level(1) for directed in [(i, j), (j, i)]}

    def matrix_measurements(complex_: SimplicialComplex, edges: Mapping) -> dict:
        inverse = max((float(np.linalg.norm(edges[(j, i)] @ edges[(i, j)] - I) / np.sqrt(2))
                       for i, j in complex_.level(1)), default=0.0)
        raw = []
        projected = []
        labels = []
        scalarities = []
        fidelities = []
        margins = []
        for i, j, k in complex_.level(2):
            matrix = edges[(k, i)] @ edges[(j, k)] @ edges[(i, j)]
            plus = float(np.linalg.norm(matrix - I) / np.sqrt(2))
            minus = float(np.linalg.norm(matrix + I) / np.sqrt(2))
            label = int(minus < plus)
            fitted = -I if label else I
            scalarity = float(np.linalg.norm(matrix - np.trace(matrix) / 2 * I) / np.sqrt(2))
            fidelity = min(plus, minus)
            raw.append({"face": [i, j, k], "matrix": matrix.tolist(), "centrality_residual": scalarity,
                        "distance_to_plus_identity": plus, "distance_to_minus_identity": minus})
            projected.append({"face": [i, j, k], "matrix": fitted.tolist(), "mu2_exponent": label,
                              "centrality_residual": 0.0})
            labels.append(label)
            scalarities.append(scalarity)
            fidelities.append(fidelity)
            margins.append(abs(plus - minus))
        c = np.asarray(labels, dtype=np.int64)
        projected_closed = bool(not np.any(complex_.delta(2) @ c % 2))
        projected_membership = gf2_solve(complex_.delta(1), c) is not None
        hodge = vector_hodge(complex_, 2, c.astype(float))
        return {"inverse_residual": inverse, "raw_face_residuals": raw, "projected_face_residuals": projected,
                "raw_centrality_residual": max(scalarities, default=0.0), "projected_centrality_residual": 0.0,
                "projection_fidelity_residual": max(fidelities, default=0.0),
                "minimum_rounding_margin": min(margins, default=0.0), "projected_cochain": c.tolist(),
                "projected_closed_f2": projected_closed,
                "exact_projected_cochain_trivial": bool(projected_membership) if projected_closed else None,
                "projected_real_cochain_distance": float(np.sqrt(hodge["distance_squared"])),
                "projected_real_distance_scope": "real-vector metric diagnostic, not a torsion certificate"}

    def connection_row(name: str, target: str, complex_: SimplicialComplex, edges: Mapping,
                       *, coefficient_correct: bool = True, common_model: bool = True,
                       radius_verified: bool = True, exact_truth: bool | None = None,
                       extra: dict | None = None) -> dict:
        measurements = matrix_measurements(complex_, edges)
        flags = {"inverse_consistency": measurements["inverse_residual"] == 0,
                 "coefficient_identification": coefficient_correct,
                 "projection_fidelity": measurements["projection_fidelity_residual"] == 0,
                 "centrality": measurements["raw_centrality_residual"] == 0,
                 "closure": measurements["projected_closed_f2"], "uncertainty_margin": radius_verified,
                 "topology": common_model}
        inapplicable = not coefficient_correct or not common_model
        return {"control_id": name, "target_gate": target, "status": "controlled_algebraic",
                "construction_type": "stored_edge_transition_error", "construction": complex_.to_json(),
                "stored_edges": [{"source": i, "target": j, "matrix": matrix.tolist()} for (i, j), matrix in sorted(edges.items())],
                **measurements, "prerequisite_flags": flags, "all_prerequisites_pass": all(flags.values()),
                "failed_gates": [gate for gate in gate_names if not flags[gate]],
                "inapplicable": inapplicable, "uncertified": not inapplicable and not all(flags.values()),
                "exact_underlying_transition_class_trivial": exact_truth,
                "exact_ground_truth_scope": "underlying admissible transition class only when stated; projected cochain truth separate",
                "task_metric_measured": False, "neural_training_performed": False, **(extra or {})}

    tetra = full_simplex(4)
    torus, torus_edges, torus_metadata = build_torus(3)
    rows = []
    reverse = flat_edges(tetra)
    # This reverse direction is absent from canonical sorted face loop products.
    # The canonical cochain remains zero, but the edge-model prerequisite fails.
    reverse[(1, 0)] = np.array([[1, 1], [0, 1]], dtype=np.int64)
    rows.append(connection_row("independently_fitted_reverse", "inverse_consistency", tetra, reverse,
                               extra={"changed_directed_edge": [1, 0], "source_construction": "flat full tetrahedron",
                                      "canonical_face_truth": "zero; not an inverse-consistent transport-class certificate"}))

    # -I is exactly outside mu3 because (-I)^3 != I.  No mu3 rounding or
    # modular membership algorithm is claimed by this mu2-only module.
    mu3_failures = [row for row in matrix_measurements(torus, torus_edges)["raw_face_residuals"]
                    if not np.array_equal(np.linalg.matrix_power(np.asarray(row["matrix"], dtype=np.int64), 3), I)]
    rows.append(connection_row("wrong_coefficient_mu3", "coefficient_identification", torus, torus_edges,
                               coefficient_correct=not mu3_failures,
                               extra={"claimed_coefficient_theory": "mu3", "actual_coefficient_theory": "mu2",
                                      "mu3_power_identity_failures": mu3_failures,
                                      "exact_mu2_ground_truth_trivial": False, "torus_metadata": torus_metadata,
                                      "requested_coefficient_theory_implemented": False}))

    noncentral = flat_edges(tetra)
    noncentral[(0, 1)] = np.array([[1, 2], [0, 1]], dtype=np.int64)
    noncentral[(1, 0)] = _inverse(noncentral[(0, 1)])
    rows.append(connection_row("noncentral_face_products", "centrality", tetra, noncentral,
                               extra={"central_model_exists_for_raw_edges": False}))

    trap = flat_edges(tetra)
    # Noncentral signed permutation has trace zero.  Nearest +/-I is ambiguous;
    # projection returns central labels while destroying the actual transport.
    trap[(0, 1)] = X.copy()
    trap[(1, 0)] = X.copy()
    rows.append(connection_row("projection_trap", "projection_fidelity", tetra, trap,
                               radius_verified=False,
                               extra={"central_model_exists_for_raw_edges": False,
                                      "projection_warning": "projected centrality zero does not validate raw transitions"}))

    # Two contexts offer incompatible coordinate dimensions for the same edge.
    # Their local inverses/centers exist, but no shared coordinate model has been
    # supplied.  The base torus matrices are retained as a local 2D diagnostic.
    rows.append(connection_row("incoherent_coordinate_context", "topology", torus, torus_edges,
                               common_model=False,
                               extra={"construction_type": "incompatible_context_coordinate_models",
                                      "common_coordinate_model": False, "application_grounded_overlap_model": False,
                                      "shared_edge_contexts": [{"context": "chart_2d", "edge": [0, 1], "dimension": 2, "matrix": I.tolist()},
                                                               {"context": "chart_3d", "edge": [0, 1], "dimension": 3, "matrix": np.eye(3, dtype=int).tolist()}],
                                      "measured_shared_edge_dimension_mismatch": True}))

    radius = 1.0
    separation = 2.0
    rows.append(connection_row("uncertified_radius", "uncertainty_margin", torus, torus_edges,
                               radius_verified=0 <= radius < separation / 2, exact_truth=False,
                               extra={"construction_type": "insufficient_estimator_error_bound",
                                      "exact_latent_finite_model_known_by_construction": True,
                                      "certified_error_radius": radius, "coefficient_minimum_separation": separation,
                                      "half_separation_margin": separation / 2,
                                      "radius_evidence_type": "supplied valid but non-sharp bound; not a measured neural error",
                                      "underlying_truth_known_by_construction_does_not_certify_estimator": True}))

    # Intentionally an input cochain rather than a realizable central G-edge
    # assignment: full K3 makes the one-negative-face input fail closure.
    c = np.array([1, 0, 0, 0], dtype=np.int64)
    delta2 = tetra.delta(2)
    closure_integer = delta2 @ c
    closure_f2 = closure_integer % 2
    hodge = vector_hodge(tetra, 2, c.astype(float))
    flags = {gate: gate != "closure" for gate in gate_names}
    rows.append({"control_id": "filled_tetrahedron_input_cochain_error", "target_gate": "closure",
                 "status": "controlled_algebraic", "construction_type": "input_cochain_error",
                 "construction": tetra.to_json(), "populated_K3": len(tetra.level(3)) > 0,
                 "edge_realizability_asserted": False, "face_input_cochain": c.tolist(),
                 "signed_delta2": delta2.tolist(), "closure_integer": closure_integer.tolist(),
                 "closure_f2": closure_f2.tolist(), "closure_residual_norm": float(np.linalg.norm(closure_integer)),
                 "raw_centrality_residual": None, "projected_centrality_residual": None,
                 "projection_fidelity_residual": None, "inverse_residual": None,
                 "prerequisite_flags": flags,
                 "prerequisite_scope": "all gates except closure are held fixed as valid rule inputs; no underlying G-edge certificate",
                 "all_prerequisites_pass": False, "failed_gates": ["closure"], "inapplicable": False, "uncertified": True,
                 "hodge_real_distance_squared": float(hodge["distance_squared"]),
                 "hodge_real_harmonic_energy": float(hodge["harmonic_energy"]),
                 "hodge_real_coexact_energy": float(hodge["coexact_energy"]),
                 "exact_input_cochain_closed": False, "exact_underlying_transition_class_trivial": None,
                 "exact_ground_truth_scope": "nonclosed input cochain has no H2 class",
                 "task_metric_measured": False, "neural_training_performed": False})

    _require(len(rows) == 7 and {row["target_gate"] for row in rows} == set(gate_names), "gate-control coverage")
    _require(rows[0]["inverse_residual"] > 0 and rows[0]["raw_centrality_residual"] == 0,
             "measured inverse-consistency control")
    _require(len(mu3_failures) == 1 and rows[1]["inapplicable"], "wrong coefficient control")
    _require(rows[2]["raw_centrality_residual"] > 0 and rows[2]["inverse_residual"] == 0,
             "measured noncentral-product control")
    _require(rows[3]["raw_centrality_residual"] > 0 and rows[3]["projected_centrality_residual"] == 0
             and rows[3]["projection_fidelity_residual"] > 0, "measured projection trap")
    _require(rows[4]["inapplicable"] and rows[4]["measured_shared_edge_dimension_mismatch"], "coordinate context control")
    _require(not rows[5]["prerequisite_flags"]["uncertainty_margin"], "uncertified radius control")
    _require(rows[6]["populated_K3"] and np.any(closure_f2) and hodge["coexact_energy"] > 0,
             "measured populated-K3 closure control")
    return rows


def verify_all() -> dict:
    """Execute bounded checks; return JSON data, raising on any failed check."""
    return {"schema_version": 1, "status": "controlled_algebraic", "passed": True,
            "scope": "finite stored-edge certificates and vector Hodge identities; not natural/learned neural evidence",
            "reference_verifier": "verification/verify_revision_examples.py absent; independent implementation",
            "coefficient_support": {"exact": ["mu2/F2"], "hodge": ["real vector"],
                                    "other_mu_n": "inapplicable; no real-sign reinterpretation"},
            "torus": [verify_torus(size, trivial_control=control) for size in (3, 4) for control in (False, True)],
            "simply_connected_no_go": verify_simply_connected_no_go(),
            "hodge": verify_hodge(), "decisions": verify_decisions(), "gate_ablation_controls": gate_ablation_controls()}
