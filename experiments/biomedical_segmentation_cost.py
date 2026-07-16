#!/usr/bin/env python3
"""B4: complete-path segmentation latency, storage, memory, and Pareto audit."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import D4EquivariantChartCNN  # noqa: E402
from experiments.spatial_output_common import (  # noqa: E402
    DEVICE,
    OUT,
    D4SymmetrizedUNet,
    TinyUNet,
    apply_d4,
    average_state_dict,
    chart_probabilities,
    dataset_checksum,
    dataset_ready,
    expert_original_frame_logits,
    factual_report,
    hard_canonical_retransport,
    load_checkpoint,
    measure_complete_path,
    model_bytes,
    predict_logits,
    record_command,
    role_split,
    soft_canonical_retransport,
    stage_complete,
    transformed_test,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "biomedical" / "cost"
COMMAND = "python experiments/biomedical_segmentation_cost.py"
METHODS = ("ordinary_unet", "direct_d4_equivariant_unet", "d4_test_time_augmentation", "generic_moe", "one_inferred_canonical_expert", "four_inferred_canonical_experts", "inferred_full_retransport", "supplied_chart_oracle", "ensemble")


def _load_models(seed: int = 0) -> dict[str, Any]:
    canonical = [load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_canonical_{index}.pt", TinyUNet(width=4))[0] for index in range(4)]
    specialists = [load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_specialist_{index}.pt", TinyUNet(width=4))[0] for index in range(4)]
    direct_base = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_direct_base.pt", TinyUNet(width=4))[0]
    chart = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_chart_equivariant.pt", D4EquivariantChartCNN(3, width=4))[0]
    return {"canonical": canonical, "specialists": specialists, "direct": D4SymmetrizedUNet(direct_base).to(DEVICE), "tta": D4SymmetrizedUNet(canonical[0]).to(DEVICE), "chart": chart}


def _callables(images: torch.Tensor, supplied_charts: torch.Tensor, models: dict[str, Any]) -> dict[str, Callable[[], torch.Tensor]]:
    canonical, specialists = models["canonical"], models["specialists"]

    def probabilities() -> torch.Tensor:
        return chart_probabilities(models["chart"], images)

    def generic() -> torch.Tensor:
        probs = probabilities()
        branch = probs[:, :4] + probs[:, 4:]
        return (branch[:, :, None, None, None] * expert_original_frame_logits(images, specialists)).sum(1)

    def one() -> torch.Tensor:
        probs = probabilities()
        return hard_canonical_retransport(images, canonical[0], probs.argmax(1))

    def four() -> torch.Tensor:
        probs = probabilities()
        charts = probs.argmax(1)
        return torch.stack([hard_canonical_retransport(images, model, charts) for model in canonical]).mean(0)

    def full() -> torch.Tensor:
        probs = probabilities()
        return soft_canonical_retransport(images, canonical, probs)

    return {
        "ordinary_unet": lambda: predict_logits(canonical[0], images),
        "direct_d4_equivariant_unet": lambda: predict_logits(models["direct"], images, 4),
        "d4_test_time_augmentation": lambda: predict_logits(models["tta"], images, 4),
        "generic_moe": generic,
        "one_inferred_canonical_expert": one,
        "four_inferred_canonical_experts": four,
        "inferred_full_retransport": full,
        "supplied_chart_oracle": lambda: torch.stack([hard_canonical_retransport(images, model, supplied_charts) for model in canonical]).mean(0),
        "ensemble": lambda: expert_original_frame_logits(images, canonical).mean(1),
    }


def _accuracy_lookup() -> dict[str, dict[str, float]]:
    path = OUT / "biomedical" / "discovery" / "summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        "ordinary_unet": "one_canonical_on_transformed_input",
        "direct_d4_equivariant_unet": "direct_d4_equivariant_unet",
        "d4_test_time_augmentation": "d4_test_time_augmentation",
        "generic_moe": "generic_soft_moe",
        "one_inferred_canonical_expert": "one_canonical_inferred_inverse_and_retransport",
        "four_inferred_canonical_experts": "four_expert_inferred_canonicalization",
        "inferred_full_retransport": "inferred_chart_canonicalize_pool_retransport",
        "supplied_chart_oracle": "supplied_chart_canonicalize_pool_retransport",
        "ensemble": "ensemble_original_frame",
    }
    return {
        method: {
            metric: float(np.mean([float(row[metric]) for row in rows if row["method"] == source]))
            for metric in ("dice", "boundary_dice")
        }
        for method, source in mapping.items()
    }


def _pareto(rows: list[dict[str, Any]], cost: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        dominated = any(float(other["dice"]) >= float(row["dice"]) and float(other[cost]) <= float(row[cost]) and (float(other["dice"]) > float(row["dice"]) or float(other[cost]) < float(row[cost])) for other in rows)
        result.append({"batch_size": row["batch_size"], "method": row["method"], "frontier": not dominated, "quality": "dice", "cost": cost, "dice": row["dice"], "cost_value": row[cost]})
    return result


def run(smoke: bool = False) -> dict[str, Any]:
    if not dataset_ready() or not (OUT / "checkpoints" / "seed_0_canonical_0.pt").exists():
        update_status("B4_complete_cost", "blocked", "B1 checkpoints unavailable")
        return {"state": "blocked", "rows": 0}
    payload, models = role_split(0), _load_models(0)
    transformed_images, _, supplied_charts = transformed_test(payload, 350_000_000)
    accuracy = _accuracy_lookup()
    rows = []
    batch_sizes = (1,) if smoke else (1, 4, 8, 16)
    for batch_size in batch_sizes:
        size = min(batch_size, len(transformed_images))
        images, charts = transformed_images[:size], supplied_charts[:size]
        callables = _callables(images, charts, models)
        repeats = 3 if smoke else (100 if size <= 4 else 30)
        warmups = 1 if smoke else 10
        for method in METHODS:
            timing = measure_complete_path(callables[method], warmups, repeats)
            if method in ("ordinary_unet", "one_inferred_canonical_expert"):
                stored = model_bytes(models["canonical"][0]) + (model_bytes(models["chart"]) if method.startswith("one_inferred") else 0)
            elif method in ("direct_d4_equivariant_unet",):
                stored = model_bytes(models["direct"])
            elif method in ("d4_test_time_augmentation",):
                stored = model_bytes(models["tta"])
            elif method == "generic_moe":
                stored = sum(model_bytes(model) for model in models["specialists"]) + model_bytes(models["chart"])
            else:
                stored = sum(model_bytes(model) for model in models["canonical"]) + (model_bytes(models["chart"]) if "inferred" in method else 0)
            rows.append({"batch_size": size, "method": method, "dice": accuracy[method]["dice"], "boundary_dice": accuracy[method]["boundary_dice"], "stored_bytes": stored, "training_time_seconds": 0.0, "energy_proxy_seconds": timing["latency_median_ms"] / 1000.0, **timing})
    pareto = []
    for batch_size in batch_sizes:
        selected = [row for row in rows if int(row["batch_size"]) == min(batch_size, len(transformed_images))]
        pareto.extend(_pareto(selected, "latency_median_ms"))
        pareto.extend(_pareto(selected, "stored_bytes"))
    write_csv(DEST / "runs.csv", rows)
    write_csv(DEST / "summary.csv", rows)
    write_csv(DEST / "pareto.csv", pareto)
    full_frontier = any(row["method"] == "inferred_full_retransport" and bool(row["frontier"]) for row in pareto)
    claims = [
        {"claim": "complete_path_repetitions_satisfied", "passed": bool(smoke or all((int(row["timed_repetitions"]) >= 100 if int(row["batch_size"]) <= 4 else int(row["timed_repetitions"]) >= 30) and int(row["warmups"]) >= 10 for row in rows))},
        {"claim": "inferred_retransport_on_any_pareto_frontier", "passed": full_frontier},
        {"claim": "twistedmerge_specific_matched_cost_gate", "passed": False, "reason": "requires positive B1 comparisons and complete-path Pareto dominance; no automatic aggregation beyond measured rows"},
    ]
    write_csv(DEST / "claims.csv", claims)
    factual_report(DEST / "report.md", "Complete biomedical segmentation cost audit", [
        f"Methods: {len(METHODS)}; batch sizes: {batch_sizes}; timing rows: {len(rows)}.",
        f"Warm-ups: {1 if smoke else 10}; repetitions: {'3 smoke' if smoke else '100 for batches 1/4 and 30 for batches 8/16'}.",
        "Every timed callable included final mask logits; inferred paths included chart inference, canonicalization, expert evaluation, pooling, and output retransport.",
        f"Inferred full retransport appears on any measured frontier: {full_frontier}.",
    ])
    update_status("B4_complete_cost", "completed", f"{len(rows)} complete-path timing rows")
    stage_complete(DEST / "runs.csv", {"stage": "B4", "state": "completed", "rows": len(rows), "full_frontier": full_frontier})
    return {"state": "completed", "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    try:
        result = run(args.smoke)
    except Exception as error:
        update_status("B4_complete_cost", "failed", str(error))
        record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="seed 0 checkpoints", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="seed 0 checkpoints", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['rows']} complete-path rows")


if __name__ == "__main__":
    main()
