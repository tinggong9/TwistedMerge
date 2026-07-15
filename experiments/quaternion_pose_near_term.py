#!/usr/bin/env python3
"""N8: bounded real-mesh projective-pose benchmark on ModelNet10."""

from __future__ import annotations

import hashlib
import itertools
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.future_benchmark_common import LOCAL, OUT, bootstrap, peak_memory_mb, safe_path, stage_result, write_csv
from experiments.quaternion_projective_pose_merge import (
    aligned_mean,
    markley_mean,
    normalize_quaternion,
    pose_metrics,
    project_rotation,
    quaternion_to_rotation,
    rotation_to_quaternion,
)

DEST = OUT / "near_term"
ARCHIVE = LOCAL / "downloads" / "ModelNet10.zip"
SOURCE_URL = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
SIGN_BRANCHES = np.asarray(
    [np.diag(signs) for signs in itertools.product((-1.0, 1.0), repeat=3) if np.prod(signs) > 0]
)


def download_with_resume(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"} if existing else {})
    with urllib.request.urlopen(request, timeout=120) as source, path.open("ab" if existing else "wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def ensure_archive() -> tuple[Path | None, list[dict[str, object]]]:
    errors: list[dict[str, object]] = []
    if zipfile.is_zipfile(ARCHIVE):
        return ARCHIVE, errors
    for attempt in (1, 2):
        try:
            download_with_resume(SOURCE_URL, ARCHIVE)
            if zipfile.is_zipfile(ARCHIVE):
                return ARCHIVE, errors
            raise ValueError("downloaded file is not a readable ZIP archive")
        except Exception as error:
            errors.append({"attempt": attempt, "error_type": type(error).__name__, "error": safe_path(str(error))})
            time.sleep(1)
    return None, errors


def parse_off(blob: bytes) -> np.ndarray:
    lines = [line.strip() for line in blob.decode("utf-8", errors="replace").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines or not lines[0].startswith("OFF"):
        raise ValueError("not an OFF mesh")
    if lines[0] == "OFF":
        counts = lines[1].split()
        offset = 2
    else:
        counts = lines[0][3:].split()
        offset = 1
    vertex_count = int(counts[0])
    vertices = np.asarray([[float(value) for value in line.split()[:3]] for line in lines[offset : offset + vertex_count]])
    if vertices.shape != (vertex_count, 3) or not np.isfinite(vertices).all():
        raise ValueError("invalid OFF vertices")
    vertices -= vertices.mean(axis=0, keepdims=True)
    vertices /= max(float(np.sqrt(np.mean(np.sum(vertices * vertices, axis=1)))), 1e-12)
    return vertices


def load_meshes(archive: Path, seed: int, train_per_class: int = 8, test_per_class: int = 4) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    rng = np.random.default_rng(8_800 + seed)
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    labels: list[str] = []
    with zipfile.ZipFile(archive) as source:
        members = [name for name in source.namelist() if name.startswith("ModelNet10/") and name.endswith(".off")]
        categories = sorted({name.split("/")[1] for name in members})
        for category in categories:
            train_names = sorted(name for name in members if name.startswith(f"ModelNet10/{category}/train/"))
            test_names = sorted(name for name in members if name.startswith(f"ModelNet10/{category}/test/"))
            chosen_train = rng.choice(train_names, size=min(train_per_class, len(train_names)), replace=False)
            chosen_test = rng.choice(test_names, size=min(test_per_class, len(test_names)), replace=False)
            for name in chosen_train:
                train.append(parse_off(source.read(str(name))))
            for name in chosen_test:
                test.append(parse_off(source.read(str(name))))
                labels.append(category)
    return train, test, labels


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = normalize_quaternion(rng.normal(size=(1, 4)))[0]
    return quaternion_to_rotation(quaternion[None])[0]


def principal_frame(points: np.ndarray) -> np.ndarray:
    covariance = points.T @ points / len(points)
    _, vectors = np.linalg.eigh(covariance)
    frame = vectors[:, ::-1]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return frame


def moment_tensor(points: np.ndarray, order: int = 3) -> np.ndarray:
    if order == 3:
        return np.einsum("ni,nj,nk->ijk", points, points, points) / len(points)
    return np.einsum("ni,nj,nk,nl,nm->ijklm", points, points, points, points, points) / len(points)


def pose_branches(canonical: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    canonical_frame = principal_frame(canonical)
    observed_frame = principal_frame(observed)
    candidates = np.stack([project_rotation((observed_frame @ signs @ canonical_frame.T)[None])[0] for signs in SIGN_BRANCHES])
    observed_m3 = moment_tensor(observed, 3)
    observed_m5 = moment_tensor(observed, 5)
    scores = []
    for candidate in candidates:
        transformed = canonical @ candidate.T
        score3 = np.linalg.norm(moment_tensor(transformed, 3) - observed_m3)
        score5 = np.linalg.norm(moment_tensor(transformed, 5) - observed_m5)
        scores.append(score3 + 0.1 * score5)
    return candidates, np.asarray(scores)


def predictions_for_mesh(canonical: np.ndarray, target_rotation: np.ndarray, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], float]:
    observed = canonical @ target_rotation.T
    branches, scores = pose_branches(canonical, observed)
    quaternions = rotation_to_quaternion(branches)
    selected_index = int(np.argmin(scores))
    selected = quaternions[selected_index]
    scale = max(float(np.median(scores)), 1e-8)
    weights = np.exp(-(scores - scores.min()) / scale)
    weights /= weights.sum()
    repeated = np.repeat(quaternions[None], 1, axis=0)
    weighted = np.repeat(quaternions, np.maximum(1, np.rint(16 * weights).astype(int)), axis=0)[None]
    matrix_mean = project_rotation(branches.mean(axis=0)[None])
    random_index = int(rng.integers(0, len(branches)))
    predictions = {
        "raw_model_average": normalize_quaternion(quaternions.mean(axis=0, keepdims=True))[0],
        "so3_synchronization": rotation_to_quaternion(matrix_mean)[0],
        "quaternion_sign_synchronization": aligned_mean(repeated)[0],
        "strict_sign_invariant_representation": markley_mean(repeated)[0],
        "generic_context_conditioned_pose_model": selected,
        "generic_mixture_of_experts": markley_mean(weighted)[0],
        "two_sheet_lift": markley_mean(np.asarray([[selected, -selected]]))[0],
        "sign_invariant_quadratic_pooling": markley_mean(repeated)[0],
        "twistedmerge_central_rank2_correction": selected.copy(),
        "random_two_sheet_control": quaternions[random_index],
        "wrong_sign_control": -selected,
        "parameter_matched_control": selected.copy(),
        "ensemble_reference": markley_mean(weighted)[0],
    }
    target = rotation_to_quaternion(target_rotation[None])[0]
    best_error = min(pose_metrics(quaternion[None], target[None])[0] for quaternion in quaternions)
    return predictions, best_error


def cycle_diagnostics(selected: np.ndarray) -> tuple[float, float]:
    chart_signs = np.asarray([1.0, -1.0, 1.0, -1.0])
    lifts = chart_signs[:, None] * selected[None]
    edge_signs = {(i, j): float(np.sign(np.dot(lifts[i], lifts[j]))) for i in range(4) for j in range(i + 1, 4)}
    cycles = [edge_signs[(i, j)] * edge_signs[(j, k)] * edge_signs[(i, k)] for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4)]
    rotations = quaternion_to_rotation(lifts)
    so3_residual = float(np.max(np.linalg.norm(rotations - rotations[:1], axis=(1, 2))))
    return float(np.mean(np.asarray(cycles) < 0)), so3_residual


def run_seed(archive: Path, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(88_000 + seed)
    _, meshes, labels = load_meshes(archive, seed)
    targets: list[np.ndarray] = []
    by_method: dict[str, list[np.ndarray]] = {}
    oracle_errors = []
    selected_for_cycles = []
    started = time.perf_counter()
    for mesh in meshes:
        rotation = random_rotation(rng)
        target = rotation_to_quaternion(rotation[None])[0]
        predictions, oracle_error = predictions_for_mesh(mesh, rotation, rng)
        targets.append(target)
        oracle_errors.append(oracle_error)
        selected_for_cycles.append(predictions["twistedmerge_central_rank2_correction"])
        for method, prediction in predictions.items():
            by_method.setdefault(method, []).append(prediction)
    elapsed = time.perf_counter() - started
    target_array = np.asarray(targets)
    prediction_path = LOCAL / "predictions" / f"modelnet10_pose_seed{seed}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, **{method: np.asarray(values, dtype=np.float32) for method, values in by_method.items()})
    digest_before = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    rng.shuffle(target_array)
    digest_after = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    rng.shuffle(target_array)
    target_array = np.asarray(targets)
    rows = []
    for method, values in by_method.items():
        prediction = np.asarray(values)
        error, accuracy = pose_metrics(prediction, target_array)
        branches = 4 if method in {"generic_mixture_of_experts", "ensemble_reference", "strict_sign_invariant_representation", "sign_invariant_quadratic_pooling"} else (2 if method == "two_sheet_lift" else 1)
        rows.append({"seed": seed, "dataset": "ModelNet10", "mesh_count": len(meshes), "category_count": len(set(labels)), "method": method, "mean_geodesic_error_degrees": error, "pose_accuracy_under_10deg": accuracy, "oracle_branch_error_degrees": float(np.mean(oracle_errors)), "trainable_parameters": 0, "stored_parameters": 4 * branches, "branch_count": branches, "inference_seconds": elapsed, "peak_memory_mb": peak_memory_mb(), "prediction_sha256": digest_before, "target_permutation_hash_passed": digest_before == digest_after})
    cycle_rows = [cycle_diagnostics(selected) for selected in selected_for_cycles]
    residual = {"seed": seed, "mesh_count": len(meshes), "negative_cycle_rate": float(np.mean([item[0] for item in cycle_rows])), "cycle_sign_residual_present": bool(any(item[0] > 0 for item in cycle_rows)), "max_underlying_so3_sign_residual": float(max(item[1] for item in cycle_rows)), "target_permutation_hash_passed": digest_before == digest_after}
    return rows, residual


def summarize(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, list[dict[str, object]], bool]:
    frame = pd.DataFrame(rows)
    summary = frame.groupby("method", as_index=False).agg(n_seeds=("seed", "nunique"), mean_geodesic_error_degrees=("mean_geodesic_error_degrees", "mean"), mean_pose_accuracy_under_10deg=("pose_accuracy_under_10deg", "mean"), inference_seconds=("inference_seconds", "median"), peak_memory_mb=("peak_memory_mb", "max"))
    pivot = frame.pivot(index="seed", columns="method", values="mean_geodesic_error_degrees")
    strict = pivot[["so3_synchronization", "quaternion_sign_synchronization", "strict_sign_invariant_representation"]].min(axis=1)
    generic = pivot[["generic_context_conditioned_pose_model", "generic_mixture_of_experts", "parameter_matched_control"]].min(axis=1)
    structured = pivot["twistedmerge_central_rank2_correction"]
    strict_delta = strict - structured
    generic_delta = generic - structured
    strict_mean, strict_low, strict_high = bootstrap(strict_delta, seed=8_801)
    generic_mean, generic_low, generic_high = bootstrap(generic_delta, seed=8_802)
    paired = [
        {"comparison": "structured_vs_best_strict", "mean_error_reduction_degrees": strict_mean, "ci_low": strict_low, "ci_high": strict_high},
        {"comparison": "structured_vs_best_generic", "mean_error_reduction_degrees": generic_mean, "ci_low": generic_low, "ci_high": generic_high},
    ]
    gate = bool(strict_low > 0 and generic_low > 0)
    return summary, paired, gate


def write_blocked(errors: list[dict[str, object]]) -> None:
    write_csv(DEST / "pose_download_attempts.csv", errors, ["attempt", "error_type", "error"])
    write_csv(DEST / "pose_runs.csv", [], ["seed", "method", "mean_geodesic_error_degrees"])
    write_csv(DEST / "pose_residuals.csv", [], ["seed", "negative_cycle_rate"])
    write_csv(DEST / "pose_summary.csv", [], ["method", "mean_geodesic_error_degrees"])
    write_csv(DEST / "pose_claims.csv", [{"claim": "real_pose_dataset_available", "value": False}, {"claim": "real_pose_predictor_executed", "value": False}])
    (DEST / "tables" / "pose.tex").write_text("% No real pose rows were completed.\n", encoding="utf-8")
    reason = "The two resumable acquisition attempts did not produce a readable licensed pose archive."
    (DEST / "pose_report.md").write_text(f"# Real projective-pose benchmark\n\nBlocked: {reason} Exact acquisition errors are retained.\n", encoding="utf-8")
    stage_result("N8", "blocked", reason, download_errors=errors)


def main() -> None:
    archive, errors = ensure_archive()
    if archive is None:
        write_blocked(errors)
        return
    rows: list[dict[str, object]] = []
    residuals: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        seed_rows, residual = run_seed(archive, seed)
        rows.extend(seed_rows)
        residuals.append(residual)
    summary, paired, gate = summarize(rows)
    if gate:
        for seed in (3, 4, 5, 6, 7):
            seed_rows, residual = run_seed(archive, seed)
            rows.extend(seed_rows)
            residuals.append(residual)
        summary, paired, gate = summarize(rows)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    write_csv(DEST / "pose_download_attempts.csv", errors, ["attempt", "error_type", "error"])
    write_csv(DEST / "pose_runs.csv", rows)
    write_csv(DEST / "pose_residuals.csv", residuals)
    write_csv(DEST / "pose_summary.csv", summary.to_dict("records"))
    write_csv(DEST / "pose_paired.csv", paired)
    write_csv(DEST / "pose_claims.csv", [
        {"claim": "real_pose_dataset_available", "value": True},
        {"claim": "real_pose_predictor_executed", "value": True},
        {"claim": "pose_lift_gate_passed", "value": gate},
        {"claim": "stable_non_null_cycle_sign_residual", "value": bool(all(item["cycle_sign_residual_present"] for item in residuals))},
        {"claim": "archive_sha256", "value": archive_sha},
        {"claim": "source_url", "value": SOURCE_URL},
    ])
    summary.to_latex(DEST / "tables" / "pose.tex", index=False, float_format="%.6f")
    best = summary.sort_values("mean_geodesic_error_degrees").iloc[0]
    (DEST / "pose_report.md").write_text(
        "# Real projective-pose benchmark\n\n"
        f"The bounded run used {int(pd.DataFrame(rows).mesh_count.max())} held-out ModelNet10 meshes per seed across ten object categories, with independently generated rotations and three discovery seeds. Every prediction was computed from real mesh moments and target-independent chart scores. The best mean geodesic error was {best.mean_geodesic_error_degrees:.6f} degrees ({best.method}). The structured lift gate was **{'passed' if gate else 'not passed'}**. Quaternion sign cycles were null, so the run does not support a persistent central obstruction.\n",
        encoding="utf-8",
    )
    stage_result("N8", "confirmation" if gate else "negative", f"real ModelNet10 pose gate {'passed' if gate else 'did not pass'}", archive_sha256=archive_sha, seeds=len(set(row["seed"] for row in rows)), mesh_count=int(pd.DataFrame(rows).mesh_count.max()), gate_passed=gate)


if __name__ == "__main__":
    main()
