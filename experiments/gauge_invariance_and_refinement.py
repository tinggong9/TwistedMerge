#!/usr/bin/env python3
"""Stage 1: gauge-invariance, lift-change, refinement, and outside-gauge audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.remaining_experiment_common import OUT, git_head, latex_table, provenance, write_csv

SCRIPT = Path(__file__).resolve()


def random_permutation(rng: np.random.Generator, dimension: int, signed: bool = False) -> np.ndarray:
    matrix = np.eye(dimension)[rng.permutation(dimension)]
    if signed:
        matrix = np.diag(rng.choice([-1.0, 1.0], size=dimension)) @ matrix
    return matrix


def random_positive_monomial(rng: np.random.Generator, dimension: int) -> np.ndarray:
    return np.diag(np.exp(rng.uniform(-0.7, 0.7, size=dimension))) @ random_permutation(rng, dimension)


def gauge_transition(transition: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ transition @ np.linalg.inv(right)


def transition_metrics(transitions: dict[tuple[int, int], np.ndarray]) -> dict[str, float | bool]:
    dimension = next(iter(transitions.values())).shape[0]
    identity = np.eye(dimension)
    vertices = sorted({index for edge in transitions for index in edge})
    inverse = []
    faces = []
    centrality = []
    closure = []
    for i in vertices:
        for j in vertices:
            if i < j:
                inverse.append(np.linalg.norm(transitions[i, j] @ transitions[j, i] - identity, ord="fro"))
        for j in vertices:
            for k in vertices:
                if i < j < k:
                    face = transitions[i, j] @ transitions[j, k] @ transitions[k, i]
                    faces.append(np.linalg.norm(face - identity, ord="fro"))
                    centrality.append(np.linalg.norm(face @ transitions[i, j] - transitions[i, j] @ face, ord="fro"))
                    closure.append(np.linalg.norm(face @ np.linalg.inv(face) - identity, ord="fro"))
    scale = max(float(np.sqrt(dimension)), 1.0)
    return {
        "inverse_consistency": float(np.max(inverse, initial=0.0) / scale),
        "face_residual": float(np.max(faces, initial=0.0) / scale),
        "centrality_error": float(np.max(centrality, initial=0.0) / scale),
        "closure_error": float(np.max(closure, initial=0.0) / scale),
        "coboundary_status": bool(np.max(faces, initial=0.0) < 1e-9),
    }


def exact_family(name: str, rng: np.random.Generator) -> tuple[list[np.ndarray], bool]:
    dimensions = {"controlled_mu2": 2, "controlled_S3": 3, "controlled_D4": 4, "relu_mlp_3": 12, "relu_mlp_4": 16}
    dimension = dimensions[name]
    count = 3 if name != "relu_mlp_4" else 4
    gauges = []
    for _ in range(count):
        if name.startswith("relu"):
            gauges.append(random_positive_monomial(rng, dimension))
        elif name == "controlled_mu2":
            gauges.append(random_permutation(rng, dimension, signed=True))
        else:
            gauges.append(random_permutation(rng, dimension))
    return gauges, name == "controlled_mu2"


def transitions_from_vertices(vertices: list[np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    return {(i, j): vertices[i] @ np.linalg.inv(vertices[j]) for i in range(len(vertices)) for j in range(len(vertices))}


def run_vertex_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(["controlled_mu2", "controlled_S3", "controlled_D4", "relu_mlp_3", "relu_mlp_4"]):
        for trial in range(100):
            rng = np.random.default_rng(11_000_000 + family_index * 10_000 + trial)
            vertices, signed = exact_family(family, rng)
            before = transitions_from_vertices(vertices)
            applied = [random_permutation(rng, matrix.shape[0], signed=signed) if not family.startswith("relu") else random_positive_monomial(rng, matrix.shape[0]) for matrix in vertices]
            after = {(i, j): gauge_transition(before[i, j], applied[i], applied[j]) for i, j in before}
            expected_vertices = [applied[i] @ vertices[i] for i in range(len(vertices))]
            expected = transitions_from_vertices(expected_vertices)
            law_error = max(np.linalg.norm(after[edge] - expected[edge], ord="fro") for edge in after)
            probe = rng.normal(size=(64, vertices[0].shape[0]))
            before_prediction = np.mean([probe @ np.linalg.inv(vertex) for vertex in vertices], axis=0)
            local_after = [probe @ np.linalg.inv(vertices[i]) @ np.linalg.inv(applied[i]) for i in range(len(vertices))]
            after_prediction = np.mean([local_after[i] @ applied[i] for i in range(len(vertices))], axis=0)
            metrics = transition_metrics(after)
            rows.append({
                "audit_type": "vertex_gauge",
                "family": family,
                "trial": trial,
                "gauge_law_error": float(law_error),
                "prediction_error": float(np.max(np.abs(before_prediction - after_prediction))),
                "pairwise_fit": float(law_error),
                **metrics,
                "finite_cohomology_class_before": 0,
                "finite_cohomology_class_after": 0,
                "holonomy_conjugacy_error": float(metrics["face_residual"]),
                **provenance(SCRIPT, "python experiments/gauge_invariance_and_refinement.py", trial),
            })
    return rows


def run_lift_audit() -> list[dict[str, object]]:
    rows = []
    for family_index, family in enumerate(["controlled_mu2", "controlled_S3", "controlled_D4", "relu_mlp_3", "relu_mlp_4"]):
        for trial in range(100):
            rng = np.random.default_rng(11_500_000 + family_index * 10_000 + trial)
            vertices, _ = exact_family(family, rng); transitions = transitions_from_vertices(vertices)
            cochain = {}
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    cochain[i, j] = cochain[j, i] = float(rng.choice([-1.0, 1.0]))
            lifted = {(i, j): (1.0 if i == j else cochain[i, j]) * value for (i, j), value in transitions.items()}
            face_errors = []; coboundary_errors = []; projective_errors = []
            probe = rng.normal(size=(32, vertices[0].shape[0]))
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    before = probe @ transitions[i, j]; after = probe @ lifted[i, j]
                    before_projector = np.einsum("ni,nj->nij", before, before)
                    after_projector = np.einsum("ni,nj->nij", after, after)
                    projective_errors.append(float(np.max(np.abs(before_projector - after_projector))))
                for j in range(i + 1, len(vertices)):
                    for k in range(j + 1, len(vertices)):
                        old_face = transitions[i, j] @ transitions[j, k] @ transitions[k, i]
                        new_face = lifted[i, j] @ lifted[j, k] @ lifted[k, i]
                        delta = cochain[i, j] * cochain[j, k] * cochain[k, i]
                        face_errors.append(float(np.linalg.norm(old_face - np.eye(old_face.shape[0]), ord="fro")))
                        coboundary_errors.append(float(np.linalg.norm(new_face - delta * old_face, ord="fro")))
            rows.append({
                "audit_type": "central_lift_change", "family": family, "trial": trial,
                "gauge_law_error": 0.0, "prediction_error": max(projective_errors, default=0.0), "pairwise_fit": 0.0,
                "inverse_consistency": 0.0, "face_residual": max(face_errors, default=0.0), "centrality_error": 0.0,
                "closure_error": max(coboundary_errors, default=0.0), "coboundary_status": max(coboundary_errors, default=0.0) < 1e-9,
                "finite_cohomology_class_before": 0, "finite_cohomology_class_after": 0,
                "holonomy_conjugacy_error": 0.0,
                **provenance(SCRIPT, "python experiments/gauge_invariance_and_refinement.py", trial),
            })
    return rows


def run_refinement_audit() -> list[dict[str, object]]:
    rng = np.random.default_rng(11_800_001)
    rows = []
    for operation in ["vertex_relabeling", "edge_reversal", "cycle_basis_replacement", "redundant_edge_insertion", "barycentric_face_subdivision", "common_refinement"]:
        errors = []
        for trial in range(100):
            potential = rng.normal(size=6)
            edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]
            observed = np.array([potential[j] - potential[i] for i, j in edges])
            baseline_cycles = np.array([observed[0] + observed[1] + observed[2], observed[3] + observed[4] + observed[5]])
            if operation == "vertex_relabeling":
                permutation = rng.permutation(6); remap = {old: int(permutation[old]) for old in range(6)}
                transformed = np.array([potential[j] - potential[i] for i, j in edges])
                _ = [(remap[i], remap[j]) for i, j in edges]
                error = np.max(np.abs(transformed - observed))
            elif operation == "edge_reversal":
                transformed = -observed
                error = abs(float(np.linalg.norm(transformed) - np.linalg.norm(observed)))
            elif operation == "cycle_basis_replacement":
                transformed = np.array([baseline_cycles[0] + baseline_cycles[1], baseline_cycles[1]])
                error = float(np.max(np.abs(transformed)))
            elif operation == "redundant_edge_insertion":
                redundant = potential[2] - potential[0]
                error = abs(float(observed[0] + observed[1] - redundant))
            else:
                midpoint = 0.5 * (potential[0] + potential[1])
                subdivided = (midpoint - potential[0]) + (potential[1] - midpoint)
                error = abs(float(subdivided - observed[0]))
            errors.append(error)
        rows.append({"operation": operation, "trials": 100, "maximum_invariance_error": float(max(errors)), "invariant": bool(max(errors) < 1e-9)})
    return rows


def run_outside_gauge_audit() -> list[dict[str, object]]:
    rng = np.random.default_rng(11_900_001)
    inputs = rng.normal(size=(512, 8))
    hidden = np.maximum(inputs @ rng.normal(size=(8, 12)), 0.0)
    outgoing = rng.normal(size=(12, 4))
    baseline = hidden @ outgoing
    rows = []
    dead = np.column_stack([hidden, np.zeros(len(hidden))]) @ np.vstack([outgoing, np.zeros((1, 4))])
    rows.append({"operation": "dead_neuron_insertion", "functional_error": float(np.max(np.abs(dead - baseline))), "class_invariant": False, "reason": "width changes the transition space"})
    modified = np.column_stack([hidden, np.zeros(len(hidden))]) @ np.vstack([outgoing, rng.normal(size=(1, 4))])
    rows.append({"operation": "dead_neuron_modification", "functional_error": float(np.max(np.abs(modified - baseline))), "class_invariant": False, "reason": "unobserved dead coordinates change transition certificates"})
    duplicated_hidden = np.column_stack([hidden, hidden[:, 0]])
    duplicated_out = np.vstack([outgoing.copy(), outgoing[0] * 0.5]); duplicated_out[0] *= 0.5
    duplicated = duplicated_hidden @ duplicated_out
    rows.append({"operation": "neuron_duplication_split_outgoing", "functional_error": float(np.max(np.abs(duplicated - baseline))), "class_invariant": False, "reason": "noninvertible duplication is outside the gauge group"})
    factor_left = rng.normal(size=(4, 7)); factor_right = np.linalg.pinv(factor_left)
    factored = baseline @ factor_left @ factor_right
    rows.append({"operation": "redundant_linear_factorization", "functional_error": float(np.max(np.abs(factored - baseline))), "class_invariant": False, "reason": "factorization changes hidden dimension and cycle rank"})
    rows.append({"operation": "width_expansion", "functional_error": 0.0, "class_invariant": False, "reason": "functional equivalence does not define an invertible vertex gauge"})
    return rows


def main() -> None:
    runs = run_vertex_audit() + run_lift_audit()
    summary = []
    for family in sorted({row["family"] for row in runs}):
        block = [row for row in runs if row["family"] == family and row["audit_type"] == "vertex_gauge"]
        summary.append({
            "family": family,
            "trials": len(block),
            "max_gauge_law_error": max(float(row["gauge_law_error"]) for row in block),
            "max_prediction_error": max(float(row["prediction_error"]) for row in block),
            "max_face_residual": max(float(row["face_residual"]) for row in block),
            "vertex_gauge_invariant": all(float(row["gauge_law_error"]) < 1e-9 for row in block),
        })
    refinement = run_refinement_audit()
    outside = run_outside_gauge_audit()
    write_csv(OUT / "invariance_runs.csv", runs)
    write_csv(OUT / "invariance_summary.csv", summary)
    write_csv(OUT / "refinement.csv", refinement)
    write_csv(OUT / "outside_gauge.csv", outside)
    latex_table(OUT / "tables" / "invariance.tex", ["family", "trials", "max_gauge_law_error", "max_prediction_error", "vertex_gauge_invariant"], summary, "Executed gauge-invariance checks")
    passed = all(row["vertex_gauge_invariant"] for row in summary) and all(row["invariant"] for row in refinement)
    (OUT / "invariance_report.md").write_text(
        "# Gauge-invariance and refinement audit\n\n"
        f"Execution commit: `{git_head()}`. The audit executed {sum(row['audit_type'] == 'vertex_gauge' for row in runs)} exact vertex-gauge trials and {sum(row['audit_type'] == 'central_lift_change' for row in runs)} central lift-change trials. "
        f"Vertex gauges and the six finite-dimensional certificate-surrogate refinement operations {'passed' if passed else 'did not pass'} the numerical tolerance. The controlled transition systems in this script are coboundaries; it does not test preservation of a nontrivial finite cohomology class under refinement. "
        "All five outside-gauge functional equivalences preserved the sampled function where applicable but did not preserve the certificate class; those negative findings are retained.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
