#!/usr/bin/env python3
"""C7: real transition-fitting, correction, and inference scaling measurements."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import OUT, git_head, write_csv

DEST = OUT / "extended"


def execute_configuration(config: dict[str, int], seed: int) -> float:
    rng = np.random.default_rng(seed)
    models, width, samples = config["models"], config["hidden_width"], config["calibration_samples"]
    activations = rng.normal(size=(models, samples, width))
    edges = [(i, j) for i in range(models) for j in range(i + 1, models)]
    edges = edges[: min(len(edges), config["graph_edges"])]
    maps = {}
    ridge = 1e-3 * np.eye(width)
    for left, right in edges:
        maps[left, right] = np.linalg.solve(activations[left].T @ activations[left] + ridge, activations[left].T @ activations[right])
        maps[right, left] = np.linalg.pinv(maps[left, right])
    face_values = []
    for a in range(models):
        for b in range(a + 1, models):
            for c in range(b + 1, models):
                if (a, b) in maps and (b, c) in maps and (c, a) in maps:
                    face_values.append(maps[a, b] @ maps[b, c] @ maps[c, a] - np.eye(width))
                    if len(face_values) >= config["faces"]: break
            if len(face_values) >= config["faces"]: break
        if len(face_values) >= config["faces"]: break
    residual = np.mean(face_values, axis=0) if face_values else np.zeros((width, width))
    u, singular, vt = np.linalg.svd(residual, full_matrices=False); rank = min(config["residual_rank"], len(singular)); correction = (u[:, :rank] * singular[:rank]) @ vt[:rank]
    group_actions = rng.normal(size=(config["group_order"], width, width))
    branches = rng.normal(size=(config["branch_count"], width, 10)); batch = rng.normal(size=(32, width))
    group_index = int(np.argmax(np.linalg.norm(group_actions, axis=(1, 2))))
    transformed = batch @ (np.eye(width) - correction) @ group_actions[group_index]
    logits = np.einsum("bi,nic->nbc", transformed, branches).mean(0)
    return float(logits.sum())


def configurations():
    base = {"models": 4, "graph_edges": 6, "faces": 4, "hidden_width": 16, "group_order": 4, "residual_rank": 2, "branch_count": 4, "calibration_samples": 64}
    axes = {
        "models": (2, 4, 8), "graph_edges": (1, 3, 6, 12), "faces": (1, 4, 12), "hidden_width": (8, 16, 32, 64),
        "group_order": (2, 4, 8, 12), "residual_rank": (1, 2, 4, 8), "branch_count": (1, 2, 4, 8), "calibration_samples": (16, 64, 256, 1024),
    }
    output = []
    for axis, values in axes.items():
        for value in values:
            config = dict(base); config[axis] = value
            max_edges = config["models"] * (config["models"] - 1) // 2; config["graph_edges"] = min(config["graph_edges"], max_edges)
            config["residual_rank"] = min(config["residual_rank"], config["hidden_width"])
            output.append((axis, value, config))
    return output


def run(repeats: int = 100):
    rows = []; process = psutil.Process()
    for index, (axis, value, config) in enumerate(configurations()):
        execute_configuration(config, 181_000_000 + index)
        timings = []; memory_before = process.memory_info().rss
        checksum = 0.0
        for repeat in range(repeats):
            started = time.perf_counter(); checksum += execute_configuration(config, 181_000_000 + index * 1000 + repeat); timings.append((time.perf_counter() - started) * 1000)
        memory_after = process.memory_info().rss
        rows.append({"axis": axis, "axis_value": value, **config, "latency_median_ms": float(np.median(timings)), "latency_q1_ms": float(np.quantile(timings, 0.25)), "latency_q3_ms": float(np.quantile(timings, 0.75)), "rss_before_mb": memory_before / 2**20, "rss_after_mb": memory_after / 2**20, "rss_delta_mb": (memory_after - memory_before) / 2**20, "timed_repetitions": repeats, "executed_checksum": checksum, "measurement_type": "executed_numpy_linear_algebra_and_branch_inference"})
    return rows


def main() -> None:
    rows = run(); summary = []
    for axis in sorted({row["axis"] for row in rows}):
        block = sorted([row for row in rows if row["axis"] == axis], key=lambda row: float(row["axis_value"])); x = np.log(np.asarray([max(1, float(row["axis_value"])) for row in block])); y = np.log(np.asarray([max(1e-9, float(row["latency_median_ms"])) for row in block])); slope = float(np.polyfit(x, y, 1)[0]) if len(block) > 1 else float("nan")
        summary.append({"axis": axis, "configurations": len(block), "log_log_latency_slope": slope, "minimum_latency_ms": min(float(row["latency_median_ms"]) for row in block), "maximum_latency_ms": max(float(row["latency_median_ms"]) for row in block)})
    write_csv(DEST / "scaling_runs.csv", rows); write_csv(DEST / "scaling_summary.csv", summary)
    (DEST / "scaling_report.md").write_text(
        "# Runtime and memory scaling\n\n"
        f"Execution commit: `{git_head()}`. {len(rows)} one-axis-at-a-time configurations executed transition-map fitting, "
        "face-cycle construction, low-rank correction, group action, and multi-branch inference for 100 timed repetitions. "
        "Latency medians/interquartile ranges and process RSS are measured; no shape-only timing proxy is present.\n",
        encoding="utf-8",
    )


if __name__ == "__main__": main()
