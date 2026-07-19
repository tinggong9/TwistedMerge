#!/usr/bin/env python3
"""Process-isolated systems benchmark for low-rank-native LoRA merging."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lora_gauge_alignment import (
    Factor,
    align_factor,
    factor_average,
    gauge_transform,
    sample_gauge,
)
from src.lora_gauge_practical import globally_aligned_factors, reference_aligned_factors

REPORT_ROOT = ROOT / "reports" / "practical_twistedmerge" / "lora_practical_followup"
OUTPUT_ROOT = REPORT_ROOT / "scalability"
REUSE_MANIFEST = REPORT_ROOT / "reuse_manifest.csv"
DIMENSIONS = (768, 1024, 2048, 4096)
RANKS = (4, 8, 16, 32)
ADAPTER_COUNTS = (4, 8, 16)
METHODS = (
    "naive_factor_average",
    "dense_deterministic_truncated_svd",
    "dense_randomized_svd",
    "canonical_factor_space",
    "pairwise_reference_alignment",
    "global_synchronization",
    "cycle_aware_alignment",
)
FACTOR_SPACE_METHODS = (
    "naive_factor_average",
    "canonical_factor_space",
    "pairwise_reference_alignment",
    "global_synchronization",
    "cycle_aware_alignment",
)
TWISTED_FACTOR_METHODS = (
    "pairwise_reference_alignment",
    "global_synchronization",
    "cycle_aware_alignment",
)
REPEATS = 3
WORKER_TIMEOUT_SECONDS = 90
GAUGE_INVARIANCE_TOLERANCE = 2e-4


@dataclass
class DenseAllocationTracker:
    """Runtime sentinel for full m-by-n effective-update allocations."""

    allow_dense: bool
    dense_allocation_count: int = 0
    dense_allocation_bytes: int = 0

    def record(self, shape: tuple[int, int], dtype: np.dtype) -> None:
        if not self.allow_dense:
            raise RuntimeError("factor-space method attempted a dense effective-update allocation")
        self.dense_allocation_count += 1
        self.dense_allocation_bytes += int(shape[0] * shape[1] * dtype.itemsize)


@dataclass(frozen=True)
class MergeOutput:
    factors: Factor
    decision: str
    tracker: DenseAllocationTracker
    analytical_temporary_bytes: int
    cycle_frobenius_defect: float
    cycle_spectral_defect: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def trained_factor_statistics() -> dict[str, float]:
    with REUSE_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    b_values = []
    a_values = []
    for row in rows:
        path = Path(row["checkpoint_path"])
        if sha256_file(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for chart in range(8):
            state = payload["states"][str(chart)]
            b_values.append(state["up.weight"].detach().cpu().numpy().astype(np.float64).ravel())
            a_values.append(state["down.weight"].detach().cpu().numpy().astype(np.float64).ravel())
    b = np.concatenate(b_values)
    a = np.concatenate(a_values)
    return {
        "trained_b_mean": float(b.mean()),
        "trained_b_std": float(b.std()),
        "trained_a_mean": float(a.mean()),
        "trained_a_std": float(a.std()),
        "trained_b_rms": float(np.sqrt(np.mean(b**2))),
        "trained_a_rms": float(np.sqrt(np.mean(a**2))),
        "source_adapter_count": 40,
        "source_rank": 4,
    }


def generate_factors(
    dimension: int,
    rank: int,
    adapter_count: int,
    seed: int,
    b_rms: float,
    a_rms: float,
) -> list[Factor]:
    """Build a common-subspace systems fixture with trained-factor scale."""

    rng = np.random.default_rng(seed)
    rank_scale = math.sqrt(4.0 / rank)
    shared_b = rng.normal(size=(dimension, rank)).astype(np.float32)
    shared_b *= np.float32(b_rms * rank_scale / max(float(np.sqrt(np.mean(shared_b**2))), 1e-12))
    factors = []
    for _ in range(adapter_count):
        basis, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
        task_a = rng.normal(size=(rank, dimension)).astype(np.float32)
        task_a *= np.float32(a_rms * rank_scale / max(float(np.sqrt(np.mean(task_a**2))), 1e-12))
        basis = basis.astype(np.float32)
        factors.append((shared_b @ basis, basis.T @ task_a))
    return factors


def canonical_low_rank_factors(factor: Factor) -> Factor:
    """Canonicalize one BA update through only thin QR and r-by-r SVD."""

    b, a = factor
    q_b, r_b = np.linalg.qr(b, mode="reduced")
    q_a, r_a = np.linalg.qr(a.T, mode="reduced")
    left, singular, right = np.linalg.svd(r_b @ r_a.T, full_matrices=False)
    q_left = q_b @ left
    q_right = right @ q_a.T
    for column in range(len(singular)):
        pivot = int(np.argmax(np.abs(q_left[:, column])))
        if q_left[pivot, column] < 0:
            q_left[:, column] *= -1
            q_right[column] *= -1
    root = np.sqrt(np.maximum(singular, 0)).astype(b.dtype, copy=False)
    return (q_left * root).astype(b.dtype, copy=False), (root[:, None] * q_right).astype(a.dtype, copy=False)


def factor_space_reference(factors: Sequence[Factor]) -> Factor:
    canonical = [canonical_low_rank_factors(factor) for factor in factors]
    return factor_average(canonical)


def dense_mean(factors: Sequence[Factor], tracker: DenseAllocationTracker) -> np.ndarray:
    b_concat = np.concatenate([factor[0] for factor in factors], axis=1)
    a_stack = np.concatenate([factor[1] for factor in factors], axis=0)
    tracker.record((b_concat.shape[0], a_stack.shape[1]), np.dtype(b_concat.dtype))
    return (b_concat @ a_stack) / np.float32(len(factors))


def deterministic_probe_basis(rows: int, columns: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.float64)[:, None] + 0.5
    column = np.arange(columns, dtype=np.float64)[None, :] + 1.0
    basis = np.sin(np.pi * row * column / rows) + np.cos(np.pi * row * (column + 0.5) / rows)
    return np.linalg.qr(basis.astype(np.float32), mode="reduced")[0]


def truncated_dense_factors(matrix: np.ndarray, rank: int, *, randomized: bool, seed: int) -> Factor:
    oversampling = min(8, max(0, matrix.shape[1] - rank))
    width = rank + oversampling
    if randomized:
        rng = np.random.default_rng(seed)
        omega = rng.normal(size=(matrix.shape[1], width)).astype(matrix.dtype)
        power_iterations = 1
    else:
        omega = deterministic_probe_basis(matrix.shape[1], width).astype(matrix.dtype)
        power_iterations = 2
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power_iterations):
        z, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
        q, _ = np.linalg.qr(matrix @ z, mode="reduced")
    small = q.T @ matrix
    left_small, singular, right = np.linalg.svd(small, full_matrices=False)
    left = q @ left_small[:, :rank]
    singular = singular[:rank]
    right = right[:rank]
    root = np.sqrt(np.maximum(singular, 0)).astype(matrix.dtype, copy=False)
    return (left * root).astype(matrix.dtype, copy=False), (root[:, None] * right).astype(matrix.dtype, copy=False)


def analytical_factor_bytes(dimension: int, rank: int, adapter_count: int, method: str) -> int:
    itemsize = np.dtype(np.float32).itemsize
    factor_bytes = 2 * dimension * rank * itemsize
    if method == "naive_factor_average":
        return 2 * factor_bytes
    if method == "canonical_factor_space":
        return adapter_count * factor_bytes + adapter_count * 6 * rank * rank * itemsize
    if method in {"pairwise_reference_alignment", "global_synchronization", "cycle_aware_alignment"}:
        return adapter_count * factor_bytes + adapter_count * adapter_count * rank * rank * itemsize
    dense_bytes = dimension * dimension * itemsize
    return dense_bytes + 4 * dimension * (rank + 8) * itemsize


def merge_case(
    factors: Sequence[Factor],
    method: str,
    *,
    seed: int,
    cycle_tolerance: float = 5e-2,
) -> MergeOutput:
    dimension = factors[0][0].shape[0]
    rank = factors[0][0].shape[1]
    tracker = DenseAllocationTracker(allow_dense=method not in FACTOR_SPACE_METHODS)
    cycle_frobenius = 0.0
    cycle_spectral = 0.0
    if method == "naive_factor_average":
        merged = factor_average(factors)
        decision = "factor_average"
    elif method == "dense_deterministic_truncated_svd":
        merged = truncated_dense_factors(dense_mean(factors, tracker), rank, randomized=False, seed=seed)
        decision = "dense_deterministic_range_svd"
    elif method == "dense_randomized_svd":
        merged = truncated_dense_factors(dense_mean(factors, tracker), rank, randomized=True, seed=seed)
        decision = "dense_randomized_svd"
    elif method == "canonical_factor_space":
        merged = factor_space_reference(factors)
        decision = "per_adapter_thin_qr_core_svd"
    elif method == "pairwise_reference_alignment":
        aligned, cycle_frobenius, cycle_spectral = reference_aligned_factors(factors)
        merged = factor_average(aligned)
        decision = "whitened_pairwise_reference_alignment"
    elif method in {"global_synchronization", "cycle_aware_alignment"}:
        aligned, cycle_frobenius, cycle_spectral = globally_aligned_factors(factors)
        if method == "cycle_aware_alignment" and max(cycle_frobenius, cycle_spectral) > cycle_tolerance:
            raise RuntimeError("cycle-aware factor-space method abstained; dense fallback disabled")
        merged = factor_average(aligned)
        decision = "cycle_gated_global_synchronization" if method == "cycle_aware_alignment" else "whitened_orthogonal_global_synchronization"
    else:
        raise ValueError(f"unknown method: {method}")
    return MergeOutput(
        factors=merged,
        decision=decision,
        tracker=tracker,
        analytical_temporary_bytes=analytical_factor_bytes(dimension, rank, len(factors), method),
        cycle_frobenius_defect=cycle_frobenius,
        cycle_spectral_defect=cycle_spectral,
    )


def apply_factor(factor: Factor, inputs: np.ndarray) -> np.ndarray:
    return (inputs @ factor[1].T) @ factor[0].T


def low_rank_inner(left: Factor, right: Factor) -> float:
    b_left, a_left = left
    b_right, a_right = right
    return float(np.trace((a_left @ a_right.T) @ (b_right.T @ b_left)))


def low_rank_distance(left: Factor, right: Factor) -> float:
    squared = max(low_rank_inner(left, left) + low_rank_inner(right, right) - 2 * low_rank_inner(left, right), 0.0)
    return math.sqrt(squared)


def mean_reference_factors(factors: Sequence[Factor]) -> Factor:
    scale = np.float32(1.0 / math.sqrt(len(factors)))
    return (
        np.concatenate([factor[0] for factor in factors], axis=1) * scale,
        np.concatenate([factor[1] for factor in factors], axis=0) * scale,
    )


def subspace_distance(left: np.ndarray, right: np.ndarray) -> float:
    q_left, _ = np.linalg.qr(left, mode="reduced")
    q_right, _ = np.linalg.qr(right, mode="reduced")
    singular = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return float(np.sqrt(np.sum(np.maximum(1.0 - singular**2, 0.0))))


def correctness_probes(
    factors: Sequence[Factor],
    output: MergeOutput,
    method: str,
    seed: int,
) -> dict[str, object]:
    dimension = factors[0][0].shape[0]
    rank = factors[0][0].shape[1]
    reference = mean_reference_factors(factors)
    reference_norm = math.sqrt(max(low_rank_inner(reference, reference), 1e-30))
    rng = np.random.default_rng(seed + 991)
    vector = rng.normal(size=(12, dimension)).astype(np.float32)
    output_values = apply_factor(output.factors, vector)
    reference_values = apply_factor(reference, vector)
    probe_error = float(np.linalg.norm(output_values - reference_values) / max(np.linalg.norm(reference_values), 1e-15))
    gauges = [sample_gauge(rng, rank, "dense", 30.0).astype(np.float32) for _ in factors]
    changed = [gauge_transform(*factor, gauge) for factor, gauge in zip(factors, gauges)]
    changed_output = merge_case(changed, method, seed=seed + 13)
    invariant_values = apply_factor(changed_output.factors, vector)
    gauge_error = float(np.linalg.norm(invariant_values - output_values) / max(np.linalg.norm(output_values), 1e-15))
    return {
        "relative_frobenius_error_to_effective_mean": low_rank_distance(output.factors, reference) / reference_norm,
        "random_vector_probe_error": probe_error,
        "random_batch_logit_error": float(np.max(np.abs(output_values - reference_values))),
        "output_subspace_distance_to_reference_span": subspace_distance(output.factors[0], reference[0]),
        "gauge_invariance_probe_error": gauge_error,
        "output_rank": int(output.factors[0].shape[1]),
        "reference_mode": "exact_low_rank_algebraic_identity",
        "dense_reference_materialized_for_correctness": False,
    }


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return peak_rss_bytes()


def run_worker(args: argparse.Namespace) -> None:
    factors = generate_factors(
        args.dimension,
        args.rank,
        args.adapter_count,
        args.seed,
        args.b_rms,
        args.a_rms,
    )
    baseline_rss = current_rss_bytes()
    warmup = merge_case(factors, args.method, seed=args.seed)
    del warmup
    gc.collect()
    trial_rows = []
    last_output = None
    for trial in range(args.repeats):
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        output = merge_case(factors, args.method, seed=args.seed)
        cpu_seconds = time.process_time() - started_cpu
        wall_seconds = time.perf_counter() - started_wall
        peak = peak_rss_bytes()
        trial_rows.append(
            {
                "trial": trial,
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "baseline_rss_bytes": baseline_rss,
                "peak_rss_bytes": peak,
                "incremental_peak_rss_bytes": max(peak - baseline_rss, 0),
                "accelerator_peak_bytes": None,
                "temporary_allocation_bytes_analytical": output.analytical_temporary_bytes,
                "dense_effective_update_allocations": output.tracker.dense_allocation_count,
                "dense_effective_update_bytes": output.tracker.dense_allocation_bytes,
                "stored_result_bytes": int(sum(value.nbytes for value in output.factors)),
                "bytes_read": 0,
                "bytes_written": 0,
                "decision": output.decision,
                "cycle_frobenius_defect": output.cycle_frobenius_defect,
                "cycle_spectral_defect": output.cycle_spectral_defect,
            }
        )
        last_output = output
    assert last_output is not None
    correctness = correctness_probes(factors, last_output, args.method, args.seed)
    print(json.dumps({"status": "success", "trials": trial_rows, "correctness": correctness}))


def worker_command(
    args: argparse.Namespace,
    dimension: int,
    rank: int,
    adapter_count: int,
    method: str,
    stats: dict[str, float],
) -> list[str]:
    seed = args.seed + dimension * 1000 + rank * 100 + adapter_count
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dimension",
        str(dimension),
        "--rank",
        str(rank),
        "--adapter-count",
        str(adapter_count),
        "--method",
        method,
        "--seed",
        str(seed),
        "--repeats",
        str(args.repeats),
        "--b-rms",
        str(stats["trained_b_rms"]),
        "--a-rms",
        str(stats["trained_a_rms"]),
    ]


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = trained_factor_statistics()
    run_rows = []
    correctness_rows = []
    failures = []
    case_count = len(DIMENSIONS) * len(RANKS) * len(ADAPTER_COUNTS) * len(METHODS)
    case_index = 0
    for dimension in DIMENSIONS:
        for rank in RANKS:
            for adapter_count in ADAPTER_COUNTS:
                for method in METHODS:
                    case_index += 1
                    command = worker_command(args, dimension, rank, adapter_count, method, stats)
                    print(f"case {case_index}/{case_count}: d={dimension} r={rank} k={adapter_count} {method}", flush=True)
                    started = time.perf_counter()
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            text=True,
                            capture_output=True,
                            timeout=args.worker_timeout,
                            check=False,
                            env={**os.environ, "PYTHONPYCACHEPREFIX": "/private/tmp/codex-pycache"},
                        )
                        if completed.returncode != 0:
                            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"worker exit {completed.returncode}")
                        payload = json.loads(completed.stdout)
                        for trial in payload["trials"]:
                            run_rows.append(
                                {
                                    "dimension_m": dimension,
                                    "dimension_n": dimension,
                                    "rank": rank,
                                    "adapter_count": adapter_count,
                                    "precision": "float32",
                                    "method": method,
                                    "status": payload["status"],
                                    "warmups": 1,
                                    "process_isolated": True,
                                    "seed": args.seed + dimension * 1000 + rank * 100 + adapter_count,
                                    **trial,
                                }
                            )
                        correctness_rows.append(
                            {
                                "dimension_m": dimension,
                                "dimension_n": dimension,
                                "rank": rank,
                                "adapter_count": adapter_count,
                                "precision": "float32",
                                "method": method,
                                "status": "success",
                                **payload["correctness"],
                            }
                        )
                    except subprocess.TimeoutExpired:
                        failures.append(
                            {
                                "dimension": dimension,
                                "rank": rank,
                                "adapter_count": adapter_count,
                                "method": method,
                                "status": "timeout",
                                "elapsed_seconds": time.perf_counter() - started,
                                "error_type": "TimeoutExpired",
                                "message": f"worker exceeded {args.worker_timeout} seconds",
                            }
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "dimension": dimension,
                                "rank": rank,
                                "adapter_count": adapter_count,
                                "method": method,
                                "status": "failure",
                                "elapsed_seconds": time.perf_counter() - started,
                                "error_type": type(error).__name__,
                                "message": str(error)[:1000],
                            }
                        )
    return pd.DataFrame(run_rows), pd.DataFrame(correctness_rows), pd.DataFrame(
        failures,
        columns=("dimension", "rank", "adapter_count", "method", "status", "elapsed_seconds", "error_type", "message"),
    )


def aggregate_resources(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["dimension_m", "dimension_n", "rank", "adapter_count", "precision", "method"]
    timing_rows = []
    memory_rows = []
    for key, frame in runs.groupby(keys, sort=True):
        common = dict(zip(keys, key))
        timing_rows.append(
            {
                **common,
                "trial_count": len(frame),
                "median_wall_seconds": frame.wall_seconds.median(),
                "p25_wall_seconds": frame.wall_seconds.quantile(0.25),
                "p75_wall_seconds": frame.wall_seconds.quantile(0.75),
                "minimum_wall_seconds": frame.wall_seconds.min(),
                "maximum_wall_seconds": frame.wall_seconds.max(),
                "median_cpu_seconds": frame.cpu_seconds.median(),
                "minimum_cpu_seconds": frame.cpu_seconds.min(),
                "maximum_cpu_seconds": frame.cpu_seconds.max(),
            }
        )
        memory_rows.append(
            {
                **common,
                "peak_rss_bytes": frame.peak_rss_bytes.max(),
                "baseline_rss_bytes": frame.baseline_rss_bytes.min(),
                "incremental_peak_rss_bytes": frame.incremental_peak_rss_bytes.max(),
                "accelerator_peak_bytes": "not_available_cpu_benchmark",
                "temporary_allocation_bytes_analytical": frame.temporary_allocation_bytes_analytical.max(),
                "dense_effective_update_allocations": frame.dense_effective_update_allocations.max(),
                "dense_effective_update_bytes": frame.dense_effective_update_bytes.max(),
                "stored_result_bytes": frame.stored_result_bytes.max(),
                "bytes_read": frame.bytes_read.max(),
                "bytes_written": frame.bytes_written.max(),
            }
        )
    return pd.DataFrame(timing_rows), pd.DataFrame(memory_rows)


def benchmark_gates(
    runs: pd.DataFrame,
    timing: pd.DataFrame,
    memory: pd.DataFrame,
    correctness: pd.DataFrame,
) -> dict[str, object]:
    factor_runs = runs[runs.method.isin(TWISTED_FACTOR_METHODS)]
    no_dense = bool((factor_runs.dense_effective_update_allocations == 0).all())
    invariance = correctness[correctness.method.isin(TWISTED_FACTOR_METHODS)]
    invariant = bool((invariance.gauge_invariance_probe_error <= GAUGE_INVARIANCE_TOLERANCE).all())
    same_rank = bool((correctness.output_rank == correctness["rank"]).all())
    merged_memory = memory.pivot_table(
        index=["dimension_m", "rank", "adapter_count"],
        columns="method",
        values="temporary_allocation_bytes_analytical",
    )
    ratios = (
        merged_memory["global_synchronization"] / merged_memory["dense_deterministic_truncated_svd"]
    ).dropna()
    substantial = bool((ratios <= 0.5).any())
    crossover_index = ratios[ratios <= 0.5].index
    crossover = None
    if len(crossover_index):
        crossover = min(crossover_index, key=lambda value: (value[0], value[1], value[2]))
    measured_memory = memory.pivot_table(
        index=["dimension_m", "rank", "adapter_count"],
        columns="method",
        values="incremental_peak_rss_bytes",
    )
    measured_ratios = (
        measured_memory["global_synchronization"]
        / measured_memory["dense_deterministic_truncated_svd"].replace(0, np.nan)
    ).dropna()
    uniform_measured_dimension = None
    for dimension in sorted(set(index[0] for index in measured_ratios.index)):
        dimension_values = measured_ratios[
            [index[0] == dimension for index in measured_ratios.index]
        ]
        if len(dimension_values) == len(RANKS) * len(ADAPTER_COUNTS) and bool((dimension_values < 1.0).all()):
            uniform_measured_dimension = int(dimension)
            break
    measured_advantage = uniform_measured_dimension is not None
    timing_table = timing.pivot_table(
        index=["dimension_m", "rank", "adapter_count"],
        columns="method",
        values="median_wall_seconds",
    )
    timing_ratios = (
        timing_table["global_synchronization"]
        / timing_table["dense_deterministic_truncated_svd"]
    ).dropna()
    largest_dimension_ratios = timing_ratios[
        [index[0] == max(DIMENSIONS) for index in timing_ratios.index]
    ]
    return {
        "twisted_factor_methods_avoid_dense_materialization": no_dense,
        "twisted_factor_methods_gauge_invariant": invariant,
        "same_output_rank": same_rank,
        "substantially_lower_analytical_temporary_memory": substantial,
        "uniform_measured_peak_memory_advantage": measured_advantage,
        "positive_scalability_gate": no_dense and invariant and substantial and measured_advantage,
        "crossover_dimension": int(crossover[0]) if crossover else None,
        "crossover_rank": int(crossover[1]) if crossover else None,
        "crossover_adapter_count": int(crossover[2]) if crossover else None,
        "minimum_observed_memory_ratio": float(ratios.min()) if len(ratios) else None,
        "minimum_measured_incremental_peak_rss_ratio": float(measured_ratios.min()) if len(measured_ratios) else None,
        "measured_half_memory_cases": int((measured_ratios <= 0.5).sum()),
        "measured_case_count": int(len(measured_ratios)),
        "uniform_measured_advantage_dimension": uniform_measured_dimension,
        "largest_dimension_faster_case_count": int((largest_dimension_ratios < 1.0).sum()),
        "largest_dimension_case_count": int(len(largest_dimension_ratios)),
        "largest_dimension_median_runtime_ratio": float(largest_dimension_ratios.median()),
    }


def write_plots(timing: pd.DataFrame, memory: pd.DataFrame) -> None:
    plots = OUTPUT_ROOT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    selected_timing = timing[(timing["rank"] == 8) & (timing.adapter_count == 8)]
    fig, axis = plt.subplots(figsize=(8.5, 5))
    for method in METHODS:
        frame = selected_timing[selected_timing.method == method].sort_values("dimension_m")
        if len(frame):
            axis.plot(frame.dimension_m, frame.median_wall_seconds, marker="o", label=method)
    axis.set_yscale("log")
    axis.set_xlabel("Square layer dimension")
    axis.set_ylabel("Median wall time (s)")
    axis.set_title("LoRA merge runtime, rank 8, 8 adapters")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "runtime_scaling.pdf")
    plt.close(fig)

    selected_memory = memory[(memory["rank"] == 8) & (memory.adapter_count == 8)]
    fig, axis = plt.subplots(figsize=(8.5, 5))
    for method in METHODS:
        frame = selected_memory[selected_memory.method == method].sort_values("dimension_m")
        if len(frame):
            axis.plot(
                frame.dimension_m,
                frame.temporary_allocation_bytes_analytical / (1024**2),
                marker="o",
                label=method,
            )
    axis.set_yscale("log")
    axis.set_xlabel("Square layer dimension")
    axis.set_ylabel("Analytical temporary memory (MiB)")
    axis.set_title("Low-rank-native versus dense temporary memory")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "memory_scaling.pdf")
    plt.close(fig)


def artifact_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
            rows.append(
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            )
    return pd.DataFrame(rows)


def write_reports(
    args: argparse.Namespace,
    runs: pd.DataFrame,
    correctness: pd.DataFrame,
    failures: pd.DataFrame,
    stats: dict[str, float],
    command: str,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timing, memory = aggregate_resources(runs)
    gates = benchmark_gates(runs, timing, memory, correctness)
    runs.to_csv(OUTPUT_ROOT / "runs.csv", index=False)
    timing.to_csv(OUTPUT_ROOT / "timing.csv", index=False)
    memory.to_csv(OUTPUT_ROOT / "memory.csv", index=False)
    correctness.to_csv(OUTPUT_ROOT / "correctness_probes.csv", index=False)
    failures.to_csv(OUTPUT_ROOT / "failure_log.csv", index=False)
    write_plots(timing, memory)
    tables = OUTPUT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    selected = timing[(timing.dimension_m == 4096) & (timing["rank"] == 8) & (timing.adapter_count == 8)]
    selected.merge(
        memory[(memory.dimension_m == 4096) & (memory["rank"] == 8) & (memory.adapter_count == 8)],
        on=["dimension_m", "dimension_n", "rank", "adapter_count", "precision", "method"],
    )[["method", "median_wall_seconds", "incremental_peak_rss_bytes", "temporary_allocation_bytes_analytical", "dense_effective_update_allocations"]].to_latex(
        tables / "scalability.tex", index=False, float_format="%.6g"
    )
    config = {
        "command": command,
        "execution_commit": git_head(),
        "worktree_dirty_during_execution": git_dirty(),
        "dimensions": DIMENSIONS,
        "ranks": RANKS,
        "adapter_counts": ADAPTER_COUNTS,
        "precision": "float32",
        "methods": METHODS,
        "warmups": 1,
        "timed_repeats": args.repeats,
        "process_isolated": True,
        "worker_timeout_seconds": args.worker_timeout,
        "seed": args.seed,
        "trained_factor_statistics": stats,
        "source_corpus_commit": "9c91bc707d1f44beb36fe0fdce43af9ce1be79ed",
        "test_labels_used": False,
        "systems_fixture_not_performance_evidence": True,
        "gates": gates,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_hashes": {
            "experiments/lora_factor_space_scalability.py": sha256_file(Path(__file__)),
            "tests/test_lora_factor_space_scalability.py": sha256_file(ROOT / "tests" / "test_lora_factor_space_scalability.py"),
            "src/lora_gauge_practical.py": sha256_file(ROOT / "src" / "lora_gauge_practical.py"),
        },
    }
    write_json(OUTPUT_ROOT / "config.json", config)
    complexity = f"""# Factor-space scalability complexity audit

The systems fixture uses the RMS scale of the 40 trained holonomy adapters but is not application-performance evidence. All comparisons use float32, identical dimensions, adapter counts, input rank, and rank-{max(RANKS)}-bounded method logic.

## Implemented accounting

- Dense deterministic and randomized methods explicitly allocate an `m x n` effective-update mean, recorded by the runtime sentinel, then use a rank-sized range-SVD workspace. Their leading memory is `O(m n)` and their dense product cost includes the effective-update materialization.
- Canonical factor space uses thin QR factorizations and `r x r` core SVDs for each adapter. It uses `O(k r (m+n) + k r^2)` storage and never requests an `m x n` buffer.
- Pairwise reference alignment whitens each B factor with an `r x r` Gram matrix, estimates orthogonal rank-space maps, and averages aligned factors. It uses `O(k r (m+n) + k^2 r^2)` storage.
- Global synchronization adds a complete rank-space transition graph and a rank-space least-squares synchronization. Its recorded implementation realizes `O(k r (m+n) + k^2 r^2)` storage; no dense update is constructed.
- Cycle-aware alignment uses the same low-rank transition graph and gauge-invariant orthogonal cycle diagnostics. In this benchmark it is forbidden to use the Phase-A dense SVD safety fallback; it must either complete in factor space or abstain.

The `temporary_allocation_bytes_analytical` field is method accounting, while `peak_rss_bytes` and `incremental_peak_rss_bytes` are process-isolated measurements. Python/runtime baseline RSS is reported separately. Stored result size is the two rank factors and is the same rank-bounded form for every successful method.
"""
    (OUTPUT_ROOT / "complexity_audit.md").write_text(complexity, encoding="utf-8")
    successful_cases = len(correctness)
    dense_memory = memory[memory.method == "dense_deterministic_truncated_svd"].temporary_allocation_bytes_analytical
    global_memory = memory[memory.method == "global_synchronization"].temporary_allocation_bytes_analytical
    report = f"""# Low-rank-native LoRA scalability benchmark

Decision: **{'positive factor-space scalability result' if gates['positive_scalability_gate'] else 'scalability gate not passed'}**.

- Successful process-isolated method/shape cases: {successful_cases} / {len(DIMENSIONS) * len(RANKS) * len(ADAPTER_COUNTS) * len(METHODS)}.
- Timed trials per successful case: {args.repeats}, after one warmup.
- Dimensions: {list(DIMENSIONS)}; ranks: {list(RANKS)}; adapter counts: {list(ADAPTER_COUNTS)}; precision: float32.
- TwistedMerge factor methods used zero dense effective-update allocations: `{gates['twisted_factor_methods_avoid_dense_materialization']}`.
- Gauge-invariance probe gate: `{gates['twisted_factor_methods_gauge_invariant']}` at tolerance `{GAUGE_INVARIANCE_TOLERANCE:.1e}`.
- Minimum analytical temporary-memory ratio, global synchronization versus deterministic dense SVD: `{gates['minimum_observed_memory_ratio']:.6g}`.
- First recorded half-memory crossover: dimension `{gates['crossover_dimension']}`, rank `{gates['crossover_rank']}`, adapters `{gates['crossover_adapter_count']}`.
- Global synchronization used lower measured incremental peak RSS in every rank/count case beginning at dimension `{gates['uniform_measured_advantage_dimension']}`.
- Measured half-memory cases: `{gates['measured_half_memory_cases']}` / `{gates['measured_case_count']}`; minimum measured incremental-RSS ratio: `{gates['minimum_measured_incremental_peak_rss_ratio']:.6g}`.
- At dimension `{max(DIMENSIONS)}`, global synchronization was faster in `{gates['largest_dimension_faster_case_count']}` / `{gates['largest_dimension_case_count']}` rank/count cases; its median runtime ratio versus deterministic dense SVD was `{gates['largest_dimension_median_runtime_ratio']:.6g}`. This is not a uniform runtime-superiority claim.
- Dense deterministic temporary memory range: `{dense_memory.min() / 1024**2:.3f}` to `{dense_memory.max() / 1024**2:.3f}` MiB.
- Global synchronization temporary memory range: `{global_memory.min() / 1024**2:.3f}` to `{global_memory.max() / 1024**2:.3f}` MiB.
- Failures/timeouts: {len(failures)}.

Random matrices here test systems and numerical behavior only. They do not establish application accuracy. Runtime comparisons are reported without a superiority claim unless output-quality and resource rows support the exact case. Dense deterministic truncated SVD and randomized dense SVD remain valid gauge-invariant baselines; the supported distinction is low-rank-native execution and measured/analytical memory, not uniqueness.
"""
    (OUTPUT_ROOT / "report.md").write_text(report, encoding="utf-8")
    artifact_manifest().to_csv(OUTPUT_ROOT / "artifact_manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--adapter-count", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--worker-timeout", type=int, default=WORKER_TIMEOUT_SECONDS)
    parser.add_argument("--b-rms", type=float, default=0.05)
    parser.add_argument("--a-rms", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if not all(value is not None for value in (args.dimension, args.rank, args.adapter_count, args.method)):
            raise ValueError("worker dimensions, rank, adapter count, and method are required")
        run_worker(args)
        return
    if args.repeats < 3:
        raise ValueError("at least three timed repeats are required")
    command = " ".join([sys.executable, *sys.argv])
    stats = trained_factor_statistics()
    runs, correctness, failures = run_benchmark(args)
    write_reports(args, runs, correctness, failures, stats, command)
    gates = json.loads((OUTPUT_ROOT / "config.json").read_text(encoding="utf-8"))["gates"]
    print(json.dumps({"run_rows": len(runs), "correctness_rows": len(correctness), "failures": len(failures), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
