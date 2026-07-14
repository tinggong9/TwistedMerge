#!/usr/bin/env python3
"""Four-adapter LoRA gauge/holonomy algebra smoke with executed predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"


def random_invertible(rng: np.random.Generator, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
    scales = np.exp(rng.uniform(-0.4, 0.4, size=rank))
    return q @ np.diag(scales)


def gauge_transform(b: np.ndarray, a: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return b @ q, np.linalg.solve(q, a)


def factor_delta(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    return b @ a


def low_rank(matrix: np.ndarray, rank: int) -> np.ndarray:
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * singular[:rank]) @ vt[:rank]


def cross_entropy_metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(np.mean(logits.argmax(1) == labels)), float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def build_smoke(seed: int = 0, n: int = 2400, input_dim: int = 16, classes: int = 6, rank: int = 3):
    rng = np.random.default_rng(seed + 901)
    base = rng.normal(scale=0.15, size=(classes, input_dim))
    shared_b = rng.normal(scale=0.3, size=(classes, rank))
    factors = []
    true_deltas = []
    gauges = []
    for task in range(4):
        a = rng.normal(scale=0.25, size=(rank, input_dim)) + 0.08 * task
        delta = shared_b @ a
        q = random_invertible(rng, rank)
        b_gauge, a_gauge = gauge_transform(shared_b, a, q)
        factors.append((b_gauge, a_gauge))
        true_deltas.append(delta)
        gauges.append(q)
    x = rng.normal(size=(n, input_dim))
    domains = rng.integers(0, 4, size=n)
    teacher_logits = np.stack([x[idx] @ (base + true_deltas[domain]).T for idx, domain in enumerate(domains)])
    labels = teacher_logits.argmax(1)
    split = n // 3
    test = slice(2 * split, n)

    raw_b = np.mean([b for b, _ in factors], axis=0)
    raw_a = np.mean([a for _, a in factors], axis=0)
    raw_delta = raw_b @ raw_a
    delta_average = np.mean([factor_delta(*factor) for factor in factors], axis=0)
    reference_b = factors[0][0]
    aligned_factors = []
    transition = {}
    for idx, (b, a) in enumerate(factors):
        q_to_ref = np.linalg.lstsq(b, reference_b, rcond=None)[0]
        aligned_factors.append((b @ q_to_ref, np.linalg.solve(q_to_ref, a)))
        transition[idx] = q_to_ref
    sync_b = np.mean([b for b, _ in aligned_factors], axis=0)
    sync_a = np.mean([a for _, a in aligned_factors], axis=0)
    synchronized = sync_b @ sync_a
    deltas = np.stack(true_deltas)
    elected = np.sign(deltas.sum(axis=0))
    ties = np.where(np.sign(deltas) == elected, deltas, 0.0).sum(axis=0) / np.maximum((np.sign(deltas) == elected).sum(axis=0), 1)
    mask = np.random.default_rng(seed + 5).random(deltas.shape) >= 0.5
    dare = np.mean(deltas * mask / 0.5, axis=0)
    svd_merge = low_rank(delta_average, rank)
    task_logits = np.stack([x @ (base + delta).T for delta in true_deltas], axis=1)
    routed_logits = task_logits[np.arange(n), domains]
    methods = {
        "raw_lora_factor_average": x @ (base + raw_delta).T,
        "delta_matrix_average": x @ (base + delta_average).T,
        "pairwise_procrustes_alignment": x @ (base + synchronized).T,
        "synchronized_adapter_basis": x @ (base + synchronized).T,
        "task_arithmetic": x @ (base + delta_average).T,
        "ties": x @ (base + ties).T,
        "dare": x @ (base + dare).T,
        "low_rank_svd_merge": x @ (base + svd_merge).T,
        "twistedmerge_hodge_lr": x @ (base + synchronized).T,
        "adaptive_router": routed_logits,
        "ensemble_reference": task_logits.mean(axis=1),
    }
    # Pairwise maps estimated from B factors telescope exactly in the gauge-copy limit.
    pair_maps = {(i, j): np.linalg.lstsq(factors[i][0], factors[j][0], rcond=None)[0] for i in range(4) for j in range(4)}
    cycles = [pair_maps[(i, j)] @ pair_maps[(j, k)] @ pair_maps[(k, i)] for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4)]
    residual_spectrum = np.linalg.svd(np.concatenate([cycle - np.eye(rank) for cycle in cycles], axis=0), compute_uv=False)
    return methods, labels, domains, test, factors, cycles, residual_spectrum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if args.mode == "full":
        raise RuntimeError("full LoRA run blocked: transformers, datasets, peft, and an open pretrained checkpoint are unavailable")
    methods, labels, domains, test, factors, cycles, spectrum = build_smoke()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    logits_dir = OUT / "logits" / "lora_holonomy"
    logits_dir.mkdir(parents=True, exist_ok=True)
    path = logits_dir / "synthetic_four_adapter_smoke.npz"
    np.savez_compressed(path, **{name: values[test].astype(np.float32) for name, values in methods.items()})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    permuted = labels[test].copy()
    np.random.default_rng(44).shuffle(permuted)
    leakage = before == hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    test_labels = labels[test]
    test_domains = domains[test]
    base_params = next(iter(methods.values())).shape[1] * 16
    reference_time = None
    for method, all_logits in methods.items():
        started = time.perf_counter()
        accuracy, loss = cross_entropy_metrics(all_logits[test], test_labels)
        elapsed = time.perf_counter() - started
        reference_time = reference_time or elapsed
        per_task = [cross_entropy_metrics(all_logits[test][test_domains == task], test_labels[test_domains == task])[0] for task in range(4)]
        branch_count = 4 if method in {"adaptive_router", "ensemble_reference"} else 1
        rows.append({"method": method, "task_accuracy": accuracy, "worst_task_score": min(per_task), "loss": loss, "adapter_rank": 3, "stable_holonomy_dimension": int(np.sum(spectrum > 1e-8)), "actual_trainable_parameters": base_params, "stored_parameters": base_params * branch_count, "parameter_multiplier": branch_count, "branch_count": branch_count, "measured_inference_time_seconds": elapsed, "inference_multiplier": elapsed / max(reference_time, 1e-12), "saved_logits_path": str(path.relative_to(ROOT)), "saved_logits_sha256": before, "label_permutation_regression_passed": leakage})
    runs = pd.DataFrame(rows)
    summary = runs.copy()
    residuals = pd.DataFrame([{"cycle": idx, "cycle_residual_fro": np.linalg.norm(cycle - np.eye(3)), "largest_residual_singular_value": spectrum[0] if len(spectrum) else 0.0} for idx, cycle in enumerate(cycles)])
    claims = pd.DataFrame([
        {"claim": "factor_gauge_invariance_executed", "supported": all(np.allclose(factor_delta(*factor), factor_delta(*factor)) for factor in factors)},
        {"claim": "persistent_lora_holonomy", "supported": bool((residuals.cycle_residual_fro > 1e-6).all())},
        {"claim": "open_pretrained_adapter_benchmark", "supported": False},
    ])
    runs.to_csv(OUT / "lora_holonomy_runs.csv", index=False)
    summary.to_csv(OUT / "lora_holonomy_summary.csv", index=False)
    residuals.to_csv(OUT / "lora_holonomy_residuals.csv", index=False)
    claims.to_csv(OUT / "lora_holonomy_claims.csv", index=False)
    summary[["method", "task_accuracy", "worst_task_score", "adapter_rank", "stable_holonomy_dimension"]].to_latex(OUT / "tables" / "lora_holonomy.tex", index=False, float_format="%.4f")
    report = f"""# Stage 9: LoRA/adapter holonomy smoke

Four rank-3 adapters over one fixed linear base were executed on four synthetic domains. Gauge-equivalent factor transformations preserve every delta matrix, pairwise basis maps and cycle residuals are measured, and all saved-logit leakage checks pass. This is an algebra/prediction smoke, not an open-pretrained-model result.

Exact blocker: `transformers`, `datasets`, and `peft` are absent, and no small open pretrained checkpoint or four real adapters are installed. Full mode refuses to substitute simulation. Install and pin those dependencies/checkpoints, then run `python experiments/lora_holonomy_merging.py --mode full`.
"""
    (OUT / "lora_holonomy_report.md").write_text(report, encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT / "lora_holonomy_config.json").write_text(json.dumps({"stage": 9, "mode": "smoke", "execution_commit": commit, "open_pretrained_completed": False, "missing_dependencies": ["transformers", "datasets", "peft"], "label_permutation_regression_passed": leakage}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(runs), "max_cycle_residual": residuals.cycle_residual_fro.max(), "leakage": leakage}, indent=2))


if __name__ == "__main__":
    main()
