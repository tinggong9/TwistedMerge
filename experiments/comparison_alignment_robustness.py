#!/usr/bin/env python3
"""C4/C5: comparison-complex and alignment-family robustness audits."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import OUT, git_head, modular_rank, write_csv

DEST = OUT / "extended"


def incidence(vertices: int, edges: list[tuple[int, int]], faces: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    b1 = np.zeros((vertices, len(edges))); edge_index = {}
    for index, (left, right) in enumerate(edges):
        if left > right: left, right = right, left
        edge_index[left, right] = index; b1[left, index] = -1; b1[right, index] = 1
    b2 = np.zeros((len(edges), len(faces)))
    for face_index, (a, b, c) in enumerate(faces):
        for left, right, sign in ((a, b, 1), (b, c, 1), (a, c, -1)):
            if left > right: left, right, sign = right, left, -sign
            if (left, right) in edge_index: b2[edge_index[left, right], face_index] += sign
    return b1, b2


def hodge(edge_values: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> dict[str, float]:
    values = np.asarray(edge_values, dtype=float).reshape(-1, 1)
    exact = b1.T @ np.linalg.pinv(b1 @ b1.T) @ b1 @ values
    remainder = values - exact
    coexact = b2 @ np.linalg.pinv(b2.T @ b2) @ b2.T @ remainder if b2.shape[1] else np.zeros_like(values)
    harmonic = remainder - coexact; scale = max(float(np.linalg.norm(values)), 1e-12)
    return {"exact": float(np.linalg.norm(exact) / scale), "coexact": float(np.linalg.norm(coexact) / scale), "harmonic": float(np.linalg.norm(harmonic) / scale), "distance_to_coboundaries": float(np.linalg.norm(remainder))}


def complexes(seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "triangle_complex": (3, [(0, 1), (1, 2), (0, 2)], [(0, 1, 2)]),
        "wedge_of_cycles": (5, [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)], []),
        "theta_graph": (5, [(0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)], []),
        "sparse_random": (8, [(i, i + 1) for i in range(7)] + [(0, 4), (2, 7)], []),
        "dense_complex": (6, [(i, j) for i in range(6) for j in range(i + 1, 6)], [(i, j, k) for i in range(6) for j in range(i + 1, 6) for k in range(j + 1, 6)]),
        "missing_faces": (6, [(i, j) for i in range(6) for j in range(i + 1, 6)], [(0, 1, 2), (0, 2, 3), (0, 3, 4)]),
        "redundant_edges_refinement": (7, [(i, i + 1) for i in range(6)] + [(0, 3), (3, 6), (0, 6), (1, 5)], [(0, 3, 6)]),
    }


def harmonic_vector(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    laplacian = b1.T @ b1 + b2 @ b2.T
    values, vectors = np.linalg.eigh(laplacian)
    index = int(np.argmin(values))
    if values[index] < 1e-8: return vectors[:, index]
    # Contractible complexes have no nontrivial H1; retain an explicit coexact
    # control rather than manufacturing a persistent class.
    return b2[:, 0] if b2.shape[1] else vectors[:, index]


def run_comparison_complexes(trials: int = 100):
    rows = []; scaling = []; rng = np.random.default_rng(161_000_000)
    for name, (vertex_count, edges, faces) in complexes().items():
        b1, b2 = incidence(vertex_count, edges, faces); h1_dimension = len(edges) - np.linalg.matrix_rank(b1) - np.linalg.matrix_rank(b2)
        for class_type in ("trivial_exact", "nontrivial_harmonic"):
            for trial in range(trials):
                started = time.perf_counter()
                if class_type == "trivial_exact": values = b1.T @ rng.normal(size=vertex_count)
                else: values = harmonic_vector(b1, b2)
                noise = 0.01 * rng.normal(size=len(edges)); observed = values + noise
                components = hodge(observed, b1, b2); elapsed = time.perf_counter() - started
                classified_nontrivial = components["harmonic"] > 0.25 and h1_dimension > 0
                expected_nontrivial = class_type == "nontrivial_harmonic" and h1_dimension > 0
                rows.append({"complex": name, "class_type": class_type, "trial": trial, "vertices": vertex_count, "edges": len(edges), "faces": len(faces), "h1_dimension": h1_dimension, **components, "classified_nontrivial": classified_nontrivial, "expected_nontrivial": expected_nontrivial, "classification_correct": classified_nontrivial == expected_nontrivial, "runtime_seconds": elapsed})
                scaling.append({"complex": name, "vertices": vertex_count, "edges": len(edges), "faces": len(faces), "runtime_seconds": elapsed})
    summary = []
    for name in complexes():
        block = [row for row in rows if row["complex"] == name]
        trivial = [row for row in block if row["class_type"] == "trivial_exact"]
        nontrivial = [row for row in block if row["class_type"] == "nontrivial_harmonic" and row["h1_dimension"] > 0]
        summary.append({"complex": name, "trials": len(block), "class_stability": float(np.mean([row["classification_correct"] for row in block])), "false_positive_rate": float(np.mean([row["classified_nontrivial"] for row in trivial])), "nontrivial_detection_rate": float(np.mean([row["classified_nontrivial"] for row in nontrivial])) if nontrivial else "not_applicable_h1_trivial", "mean_distance_to_coboundaries": float(np.mean([row["distance_to_coboundaries"] for row in block])), "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in block]))})
    return rows, summary, scaling


def alignment_audit():
    path = OUT / "iclr" / "full_model_transitions.csv"
    if not path.exists(): raise FileNotFoundError("C5 requires B1 transition artifacts")
    with path.open(encoding="utf-8", newline="") as handle: source = list(csv.DictReader(handle))
    rows = []
    for gauge in sorted({row["gauge_family"] for row in source}):
        block = [row for row in source if row["gauge_family"] == gauge]
        fit = float(np.mean([float(row["heldout_pairwise_fit"]) for row in block])); cycle = float(np.mean([float(row["cycle_residual"]) for row in block])); inverse = float(np.mean([float(row["inverse_consistency"]) for row in block]))
        if fit > 0.35: classification = "poor_pairwise_fit"
        elif cycle < 0.05 and inverse < 0.05: classification = "removable_gauge_error"
        elif cycle < 0.15: classification = "noise"
        else: classification = "persistent_holonomy_candidate"
        rows.append({"alignment_family": gauge, "layers": len({row["layer"] for row in block}), "collections": len({row["collection"] for row in block}), "mean_pairwise_fit": fit, "mean_inverse_consistency": inverse, "mean_cycle_residual": cycle, "classification": classification})
    # LoRA basis maps are reported separately; no adapter subspace existed in C3.
    rows.append({"alignment_family": "LoRA_basis_maps", "layers": 0, "collections": 0, "mean_pairwise_fit": "unavailable", "mean_inverse_consistency": "unavailable", "mean_cycle_residual": "unavailable", "classification": "not_executed_no_nontrivial_adapter_mechanism"})
    return rows


def main() -> None:
    runs, summary, scaling = run_comparison_complexes(); alignment = alignment_audit()
    write_csv(DEST / "complex_runs.csv", runs); write_csv(DEST / "complex_summary.csv", summary); write_csv(DEST / "complex_scaling.csv", scaling); write_csv(DEST / "alignment_robustness.csv", alignment)
    (DEST / "complex_alignment_report.md").write_text(
        "# Comparison-complex and alignment robustness\n\n"
        f"Execution commit: `{git_head()}`. Seven complex families were evaluated with 100 trivial and 100 nontrivial-or-control "
        "trials each using measured Hodge projections and real runtime. False positives and contractible-complex H1 boundaries "
        "are explicit. Six full-model alignment families were classified from held-out fit, inverse consistency, and cycle "
        "residual; LoRA maps remain unavailable because C2's mechanism prerequisite did not pass.\n",
        encoding="utf-8",
    )


if __name__ == "__main__": main()
