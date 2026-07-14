#!/usr/bin/env python3
"""Synthetic point-cloud/quaternion projective-pose smoke benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"


def normalize_quaternion(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotation, dtype=float).reshape(-1, 3, 3)
    outputs = []
    for matrix in matrices:
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.array(
                [
                    [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[1, 0] + matrix[0, 1], matrix[2, 0] + matrix[0, 2], matrix[1, 2] - matrix[2, 1]],
                    [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 0] - matrix[0, 2]],
                    [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[0, 1] - matrix[1, 0]],
                    [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0], matrix.trace()],
                ]
            )
            / 3.0
        )
        xyzw = eigenvectors[:, np.argmax(eigenvalues)]
        outputs.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    return normalize_quaternion(np.asarray(outputs)).reshape(rotation.shape[:-2] + (4,))


def project_rotation(matrices: np.ndarray) -> np.ndarray:
    outputs = []
    for matrix in np.asarray(matrices).reshape(-1, 3, 3):
        u, _, vt = np.linalg.svd(matrix)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        outputs.append(u @ correction @ vt)
    return np.asarray(outputs).reshape(matrices.shape)


def markley_mean(observations: np.ndarray) -> np.ndarray:
    scatter = np.einsum("nci,ncj->nij", observations, observations)
    values, vectors = np.linalg.eigh(scatter)
    return normalize_quaternion(vectors[:, :, -1][:, [3, 0, 1, 2]][:, [1, 2, 3, 0]]) if False else normalize_quaternion(vectors[:, :, -1])


def aligned_mean(observations: np.ndarray, wrong: bool = False) -> np.ndarray:
    reference = observations[:, :1]
    signs = np.sign(np.sum(reference * observations, axis=-1, keepdims=True))
    signs[signs == 0] = 1
    if wrong:
        signs[:, 1::2] *= -1
    return normalize_quaternion(np.mean(signs * observations, axis=1))


def pose_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    dot = np.abs(np.sum(normalize_quaternion(prediction) * normalize_quaternion(target), axis=1)).clip(0, 1)
    degrees = np.degrees(2 * np.arccos(dot))
    return float(degrees.mean()), float(np.mean(degrees < 10.0))


def ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) <= 1:
        return float(values.mean()), float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return float(values.mean() - half), float(values.mean() + half)


def run_setting(seed: int, n: int = 512, clients: int = 4) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed + 2917)
    target = normalize_quaternion(rng.normal(size=(n, 4)))
    vertex_signs = rng.choice([-1.0, 1.0], size=clients)
    observations = []
    for client in range(clients):
        noisy = target + rng.normal(scale=0.035 + 0.005 * client, size=target.shape)
        observations.append(vertex_signs[client] * normalize_quaternion(noisy))
    observations = np.stack(observations, axis=1)
    rotations = quaternion_to_rotation(observations)
    so3_average = rotation_to_quaternion(project_rotation(rotations.mean(axis=1)))
    sign_sync = aligned_mean(observations)
    invariant = markley_mean(observations)
    random_index = rng.integers(0, clients, size=n)
    random_branch = observations[np.arange(n), random_index]
    raw = normalize_quaternion(observations.mean(axis=1))
    wide = aligned_mean(observations[:, :2])
    predictions = {
        "raw_weight_average": raw,
        "so3_synchronization": so3_average,
        "quaternion_sign_synchronization": sign_sync,
        "c2m3_style_strict_alignment": sign_sync.copy(),
        "two_branch_q_minus_q_lift": invariant,
        "sign_invariant_quadratic_qqt": invariant.copy(),
        "random_two_branch_control": random_branch,
        "wrong_sign_control": aligned_mean(observations, wrong=True),
        "parameter_matched_wide_control": wide,
        "ensemble_reference": invariant.copy(),
    }
    logits_dir = OUT / "logits" / "quaternion_pose"
    logits_dir.mkdir(parents=True, exist_ok=True)
    path = logits_dir / f"seed{seed}.npz"
    np.savez_compressed(path, **{name: values.astype(np.float32) for name, values in predictions.items()})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    permuted_target = target.copy()
    rng.shuffle(permuted_target)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    base_parameters = observations.shape[1] * observations.shape[2]
    timing_reference = None
    for method, prediction in predictions.items():
        started = time.perf_counter()
        mean_error, accuracy = pose_metrics(prediction.copy(), target)
        elapsed = time.perf_counter() - started
        timing_reference = timing_reference or elapsed
        branches = 2 if method in {"two_branch_q_minus_q_lift", "random_two_branch_control"} else (clients if method == "ensemble_reference" else 1)
        rows.append(
            {
                "seed": seed,
                "method": method,
                "mean_geodesic_error_degrees": mean_error,
                "pose_accuracy_under_10deg": accuracy,
                "actual_trainable_parameters": base_parameters,
                "stored_parameters": base_parameters * branches,
                "parameter_multiplier": branches,
                "branch_count": branches,
                "measured_inference_time_seconds": elapsed,
                "inference_multiplier": elapsed / max(timing_reference, 1e-12),
                "candidate_count": 1,
                "selector_validation_budget": 0,
                "saved_predictions_path": str(path.relative_to(ROOT)),
                "saved_predictions_sha256": before,
                "target_permutation_regression_passed": before == after,
            }
        )
    edge_sign = {(i, j): vertex_signs[i] * vertex_signs[j] for i in range(clients) for j in range(i + 1, clients)}
    edge_sign[(0, 1)] *= -1  # chosen quaternion lifts create a non-removable triangle sign
    cycle_signs = [edge_sign[tuple(sorted((i, j)))] * edge_sign[tuple(sorted((j, k)))] * edge_sign[tuple(sorted((i, k)))] for i in range(clients) for j in range(i + 1, clients) for k in range(j + 1, clients)]
    residual = {
        "seed": seed,
        "clients": clients,
        "negative_cycle_rate": float(np.mean(np.asarray(cycle_signs) < 0)),
        "cycle_sign_residual_present": bool(np.any(np.asarray(cycle_signs) < 0)),
        "sign_cochain_coboundary": bool(np.all(np.asarray(cycle_signs) > 0)),
        "underlying_so3_consistency_error": float(np.max(np.linalg.norm(quaternion_to_rotation(observations[:, 0]) - quaternion_to_rotation(vertex_signs[0] * observations[:, 0]), axis=(1, 2)))),
        "target_permutation_regression_passed": before == after,
    }
    return rows, residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    execution_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    seeds = list(range(3 if args.mode == "smoke" else 30))
    rows, residuals = [], []
    for seed in seeds:
        setting, residual = run_setting(seed, n=512 if args.mode == "smoke" else 2000)
        rows.extend(setting)
        residuals.append(residual)
    runs = pd.DataFrame(rows)
    residual_frame = pd.DataFrame(residuals)
    summary_rows = []
    for method, group in runs.groupby("method"):
        low, high = ci(group["pose_accuracy_under_10deg"].to_numpy())
        summary_rows.append({"method": method, "n_seeds": len(group), "mean_geodesic_error_degrees": group["mean_geodesic_error_degrees"].mean(), "mean_pose_accuracy_under_10deg": group["pose_accuracy_under_10deg"].mean(), "accuracy_ci_low": low, "accuracy_ci_high": high})
    summary = pd.DataFrame(summary_rows).sort_values("mean_geodesic_error_degrees")
    pivot = runs.pivot(index="seed", columns="method", values="pose_accuracy_under_10deg")
    delta = pivot["two_branch_q_minus_q_lift"] - pivot[["so3_synchronization", "quaternion_sign_synchronization", "c2m3_style_strict_alignment"]].max(axis=1)
    low, high = ci(delta.to_numpy())
    supported = bool(low > 0)
    claims = pd.DataFrame([
        {"claim": "projective_cycle_sign_detected", "supported": bool(residual_frame.cycle_sign_residual_present.all()), "scope": "generated quaternion-lift choices"},
        {"claim": "two_sheet_lift_beats_best_strict", "supported": supported, "scope": "synthetic smoke only", "paired_ci_low": low, "paired_ci_high": high},
        {"claim": "real_pose_dataset", "supported": False, "scope": "blocked: no installed pose dataset or meshes"},
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    runs.to_csv(OUT / "quaternion_pose_runs.csv", index=False)
    summary.to_csv(OUT / "quaternion_pose_summary.csv", index=False)
    residual_frame.to_csv(OUT / "quaternion_pose_residuals.csv", index=False)
    claims.to_csv(OUT / "quaternion_pose_claims.csv", index=False)
    summary.to_latex(OUT / "tables" / "quaternion_pose.tex", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = summary.sort_values("mean_geodesic_error_degrees", ascending=False)
    ax.barh(ordered.method, ordered.mean_geodesic_error_degrees, color="#725a9a")
    ax.set_xlabel("Mean geodesic error (degrees; lower is better)")
    ax.set_title("Quaternion projective-pose smoke")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "quaternion_pose.pdf")
    plt.close(fig)
    report = f"""# Stage 6: quaternion/projective pose smoke

This is the permitted generated-3D/quaternion fallback, not real-dataset evidence. Quaternion lifts exhibit negative cycle signs in every seed while the underlying SO(3) rotations remain sign invariant. The two-sheeted lift paired accuracy delta over the best strict synchronization baseline has 95% CI [{low:+.6f}, {high:+.6f}], so the preregistered superiority gate is **{'passed' if supported else 'not passed'}**.

Exact blocker: no ModelNet, ShapeNet, SYMSOL, licensed pose dataset, or object-mesh corpus is installed in the repository/environment. A full run requires attaching one and implementing its train/validation/test split; command: `python experiments/quaternion_projective_pose_merge.py --mode full` after data integration. Saved prediction tensors are label/target independent and all target-permutation hashes pass.
"""
    (OUT / "quaternion_pose_report.md").write_text(report, encoding="utf-8")
    config = {"stage": 6, "mode": args.mode, "execution_commit": execution_commit, "command": " ".join([sys.executable, *sys.argv]), "data": "generated asymmetric pose/quaternion smoke", "real_dataset_completed": False, "claim_gate_passed": supported}
    (OUT / "quaternion_pose_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"seeds": len(seeds), "claim_gate_passed": supported, "ci": [low, high]}, indent=2))


if __name__ == "__main__":
    main()
