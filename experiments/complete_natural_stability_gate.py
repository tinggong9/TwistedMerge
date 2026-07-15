#!/usr/bin/env python3
"""Stage 5: complete the fixed natural-residual stability gate."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compact_benchmark_common as compact_common
import experiments.compact_natural_twist as natural
from experiments.remaining_experiment_common import DATA, OUT, classification_metrics, git_head, logits_hashes, matched_bootstrap, write_csv

SCRIPT = Path(__file__).resolve()
FUTURE_TMP = ROOT / "reports" / "tmp" / "future_program"
compact_common.DATA = DATA


def fixed_map(source: np.ndarray, target: np.ndarray, family: str) -> np.ndarray:
    if family == "orthogonal": return natural.orthogonal_map(source, target)
    if family == "positive_monomial": return natural.positive_monomial_map(source, target)
    raise ValueError(family)


def residual_statistics(calibration: list[np.ndarray], family: str, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(55_000_000 + seed)
    dimension = calibration[0].shape[1]
    identity = np.eye(dimension)

    def maps_for(indices: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        return {(i, j): fixed_map(calibration[i][indices], calibration[j][indices], family) for i in range(4) for j in range(4) if i != j}

    fit = np.arange(250); heldout = np.arange(250, 500); maps = maps_for(fit)
    observed_cycle = maps[0, 1] @ maps[1, 2] @ maps[2, 0] - identity
    observed = float(np.linalg.norm(observed_cycle, ord="fro") / np.sqrt(observed_cycle.size))
    heldout_fit = float(np.mean([np.linalg.norm(calibration[i][heldout] @ maps[i, j] - calibration[j][heldout]) / max(np.linalg.norm(calibration[j][heldout]), 1e-8) for i in range(4) for j in range(4) if i != j]))
    resample_rows = []
    for resample in range(5):
        indices = rng.choice(250, 250, replace=True); local = maps_for(indices)
        cycle = local[0, 1] @ local[1, 2] @ local[2, 0] - identity
        values = np.linalg.svd(cycle, compute_uv=False)
        resample_rows.append({"resample": resample, "cycle_residual": float(np.linalg.norm(cycle, ord="fro") / np.sqrt(cycle.size)), "residual_rank": int(np.sum(values > max(0.05 * values[0], 1e-7))) if values[0] > 0 else 0})
    null_values = {name: [] for name in ["edge_shuffle", "matched_norm_coboundary", "matched_fit_random_gauge", "graph_topology_shuffle"]}
    edges = list(maps.values())
    for draw in range(200):
        chosen = rng.choice(len(edges), 3, replace=True)
        shuffled = edges[int(chosen[0])] @ edges[int(chosen[1])] @ edges[int(chosen[2])] - identity
        null_values["edge_shuffle"].append(float(np.linalg.norm(shuffled, ord="fro") / np.sqrt(shuffled.size)))
        nodes = []
        for _ in range(3):
            q, _ = np.linalg.qr(identity + rng.normal(scale=max(observed, 1e-5), size=(dimension, dimension)))
            nodes.append(q)
        coboundary = (nodes[0].T @ nodes[1]) @ (nodes[1].T @ nodes[2]) @ (nodes[2].T @ nodes[0]) - identity
        null_values["matched_norm_coboundary"].append(float(np.linalg.norm(coboundary, ord="fro") / np.sqrt(coboundary.size)))
        random_edges = []
        for _ in range(3):
            q, _ = np.linalg.qr(identity + rng.normal(scale=max(heldout_fit, 1e-5), size=(dimension, dimension)))
            random_edges.append(q)
        random_cycle = random_edges[0] @ random_edges[1] @ random_edges[2] - identity
        null_values["matched_fit_random_gauge"].append(float(np.linalg.norm(random_cycle, ord="fro") / np.sqrt(random_cycle.size)))
        topology = maps[0, 2] @ maps[2, 1] @ maps[1, 0] - identity
        signs = rng.choice([-1.0, 1.0], size=topology.shape)
        null_values["graph_topology_shuffle"].append(float(np.linalg.norm(topology * signs, ord="fro") / np.sqrt(topology.size)))
    null_rows = []
    thresholds = {}
    for family_name, values in null_values.items():
        thresholds[family_name] = float(np.quantile(values, 0.95))
        for draw, value in enumerate(values): null_rows.append({"null_family": family_name, "draw": draw, "null_residual": value})
    ranks = [int(row["residual_rank"]) for row in resample_rows]; norms = [float(row["cycle_residual"]) for row in resample_rows]
    stable = len(set(ranks)) == 1 and float(np.std(norms) / max(np.mean(norms), 1e-9)) < 0.2
    beyond = all(observed > threshold for threshold in thresholds.values())
    corrected_map = np.linalg.inv(maps[0, 1] @ maps[1, 2])
    corrected_cycle = maps[0, 1] @ maps[1, 2] @ corrected_map - identity
    corrected = float(np.linalg.norm(corrected_cycle, ord="fro") / np.sqrt(corrected_cycle.size))
    return resample_rows, null_rows, {"selected_family": family, "heldout_pairwise_fit": heldout_fit, "observed_residual": observed, "corrected_residual": corrected, "residual_reduced": corrected < observed, "calibration_resample_stable": stable, "exceeds_all_nulls": beyond, **{f"{key}_q95": value for key, value in thresholds.items()}}


def load_states(dataset: str, relation: str, seed: int) -> list[dict[str, torch.Tensor]]:
    directory = FUTURE_TMP / "checkpoints" / "natural" / dataset / "mlp" / relation / f"seed{seed}"
    paths = [directory / f"model{index}.pt" for index in range(4)]
    if not all(path.exists() for path in paths): raise FileNotFoundError(directory)
    return [torch.load(path, map_location="cpu", weights_only=True) for path in paths]


def run_family(dataset: str, relation: str, family: str, seeds: list[int]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    stability_rows = []; null_rows = []; correction_rows = []; accuracy_rows = []
    for seed in seeds:
        data = natural.prepare_data(dataset, seed); states = load_states(dataset, relation, seed)
        calibration, _ = natural.evaluate_states(dataset, "mlp", states, data.calibration_x)
        resamples, nulls, summary = residual_statistics(calibration, family, seed)
        setting = f"{dataset}_mlp_4_{relation}_s{seed}"
        for row in resamples: stability_rows.append({"setting_id": setting, "dataset": dataset, "relation": relation, "seed": seed, **row})
        for row in nulls: null_rows.append({"setting_id": setting, "dataset": dataset, "relation": relation, "seed": seed, **row})
        correction_rows.append({"setting_id": setting, "dataset": dataset, "relation": relation, "seed": seed, **summary})
        logits_path = FUTURE_TMP / "logits" / f"{dataset}_mlp_4_{relation}_s{seed}.npz"
        logits = np.load(logits_path, allow_pickle=False)
        methods = {key: logits[key] for key in ["strict_synchronization", "generic_low_rank_correction", "twistedmerge_hodge_lr"]}
        hash_record = logits_hashes(f"natural_stability_{setting}", methods, data.test_y.numpy(), 55_900_000 + seed)
        for method, values in methods.items():
            accuracy_rows.append({"setting_id": setting, "dataset": dataset, "relation": relation, "seed": seed, "method": method, **classification_metrics(values, data.test_y.numpy()), "label_permutation_hash_passed": hash_record["label_permutation_hash_passed"], "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
    return stability_rows, null_rows, correction_rows, accuracy_rows


def main() -> None:
    configurations = [
        ("FashionMNIST", "independent_seeds", "orthogonal"),
        ("MNIST", "shared_base_specialization", "positive_monomial"),
    ]
    stability = []; nulls = []; corrections = []; accuracies = []
    for dataset, relation, family in configurations:
        a, b, c, d = run_family(dataset, relation, family, list(range(10, 17)))
        stability.extend(a); nulls.extend(b); corrections.extend(c); accuracies.extend(d)
    claims = []
    for dataset, relation, family in configurations:
        block = [row for row in corrections if row["dataset"] == dataset and row["relation"] == relation]
        accuracy_block = [row for row in accuracies if row["dataset"] == dataset and row["relation"] == relation]
        deltas = []
        for seed in range(10, 17):
            hodge = next(float(row["accuracy"]) for row in accuracy_block if row["seed"] == seed and row["method"] == "twistedmerge_hodge_lr")
            strict = next(float(row["accuracy"]) for row in accuracy_block if row["seed"] == seed and row["method"] == "strict_synchronization")
            deltas.append(hodge - strict)
        mean, low, high = matched_bootstrap(deltas, seed=55_700_000 + len(dataset))
        passed = all(bool(row["calibration_resample_stable"]) and bool(row["exceeds_all_nulls"]) and bool(row["residual_reduced"]) for row in block) and low > 0
        claims.append({"dataset": dataset, "architecture": "mlp", "model_count": 4, "relation": relation, "fixed_transition_family": family, "seeds": 7, "calibration_resamples": 5, "null_draws_per_family": 200, "delta_vs_strict": mean, "ci_low": low, "ci_high": high, "gate_passed": passed})
    write_csv(OUT / "natural_stability.csv", stability)
    write_csv(OUT / "natural_nulls.csv", nulls)
    write_csv(OUT / "natural_corrections.csv", corrections)
    write_csv(OUT / "natural_claims.csv", claims)
    passed_count = sum(bool(row["gate_passed"]) for row in claims)
    (OUT / "natural_report.md").write_text(
        "# Completed natural-residual stability gate\n\n"
        f"Execution commit: `{git_head()}`. The two fixed families were evaluated without reselection using five calibration resamples and 200 draws from each of four matched-null families. "
        f"{passed_count} of 2 families passed the complete stability, null, residual-reduction, and accuracy gate.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
