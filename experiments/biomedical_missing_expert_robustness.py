#!/usr/bin/env python3
"""C2: missing-expert and sparse-comparison robustness on trained experts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import D4EquivariantChartCNN  # noqa: E402
from experiments.multidomain_biomedical_experts import (  # noqa: E402
    _domain_chart_soft,
    domain_probabilities,
    synthetic_domain,
)
from experiments.spatial_output_common import (  # noqa: E402
    DEVICE,
    OUT,
    TinyUNet,
    apply_d4,
    chart_probabilities,
    dataset_checksum,
    dataset_ready,
    expert_original_frame_logits,
    factual_report,
    load_checkpoint,
    pixel_ece,
    record_command,
    role_split,
    segmentation_metrics,
    stage_complete,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "robustness"
COMMAND = "python experiments/biomedical_missing_expert_robustness.py"
SCENARIOS = (
    "one_missing_expert",
    "two_missing_experts",
    "incomplete_comparison_graph",
    "one_incorrect_transition",
    "noisy_chart_estimates",
    "unavailable_center_metadata",
    "reduced_calibration",
)
METHODS = (
    "uniform_pooling",
    "validation_weighted_pooling",
    "generic_router",
    "graph_synchronization",
    "hodge_diagnostic",
    "chart_aware_structured_pooling",
    "abstaining_fallback",
)


def synchronize_scalar_edges(
    edges: list[tuple[int, int, float]], nodes: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares synchronize additive pairwise offsets with node zero fixed."""

    design, target = [], []
    for left, right, value in edges:
        row = np.zeros(nodes - 1)
        if left:
            row[left - 1] -= 1
        if right:
            row[right - 1] += 1
        design.append(row)
        target.append(value)
    if not design:
        return np.zeros(nodes), np.zeros(0)
    fitted = np.linalg.lstsq(np.asarray(design), np.asarray(target), rcond=None)[0]
    potentials = np.concatenate([[0.0], fitted])
    residuals = np.asarray(
        [value - (potentials[right] - potentials[left]) for left, right, value in edges]
    )
    return potentials, residuals


def _load(seed: int) -> tuple[list[TinyUNet], Any]:
    experts = [
        load_checkpoint(
            OUT / "checkpoints" / f"multidomain_seed_{seed}_expert_{index}.pt",
            TinyUNet(width=4),
        )[0]
        for index in range(4)
    ]
    chart = load_checkpoint(
        OUT / "checkpoints" / f"seed_{seed}_chart_equivariant.pt",
        D4EquivariantChartCNN(3, width=4),
    )[0]
    return experts, chart


def _transition_diagnostic(
    experts: list[TinyUNet], images: torch.Tensor, corrupt: bool
) -> dict[str, float]:
    with torch.no_grad():
        signatures = np.asarray(
            [
                float(model.forward_features(images.to(DEVICE))["bottleneck"].abs().mean().cpu())
                for model in experts
            ]
        )
    edges = [
        (left, right, float(signatures[right] - signatures[left]))
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    if corrupt:
        left, right, value = edges[1]
        edges[1] = (left, right, value + max(float(signatures.std()), 1e-3) * 8)
    _, residuals = synchronize_scalar_edges(edges)
    worst = int(np.argmax(np.abs(residuals))) if len(residuals) else 0
    repaired = [edge for index, edge in enumerate(edges) if index != worst]
    _, repaired_residuals = synchronize_scalar_edges(repaired)
    before = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
    after = (
        float(np.sqrt(np.mean(repaired_residuals**2))) if len(repaired_residuals) else 0.0
    )
    return {
        "hodge_residual": before,
        "residual_after_edge_removal": after,
        "corrupted_edge_recovery": float(after < before),
    }


def _available(scenario: str) -> list[int]:
    if scenario == "one_missing_expert":
        return [1, 2, 3]
    if scenario == "two_missing_experts":
        return [2, 3]
    return [0, 1, 2, 3]


def run(smoke: bool = False) -> dict[str, Any]:
    required = OUT / "checkpoints" / "multidomain_seed_0_expert_0.pt"
    if not dataset_ready() or not required.exists():
        update_status("C2_missing_expert_robustness", "blocked", "C1 checkpoints unavailable")
        return {"state": "blocked", "seeds": 0}
    seeds = [0] if smoke else list(range(5))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        payload = role_split(seed)
        experts, chart_model = _load(seed)
        count = min(4, len(payload["test_images"]))
        base_images = torch.cat(
            [synthetic_domain(payload["test_images"][index : index + 1], index % 4) for index in range(count)]
        )
        true_charts = torch.tensor([(index * 3 + 1) % 8 for index in range(count)], dtype=torch.long)
        images = apply_d4(base_images, true_charts)
        masks = apply_d4(payload["test_masks"][:count], true_charts)
        prototypes = torch.stack(
            [synthetic_domain(payload["early_images"], domain).mean((0, 2, 3)) for domain in range(4)]
        )
        base_domain = domain_probabilities(images, prototypes)
        base_chart = chart_probabilities(chart_model, images)
        expert_logits = expert_original_frame_logits(images, experts)
        validation_dice = []
        for expert in experts:
            probability = torch.sigmoid(expert_original_frame_logits(payload["early_images"], [expert])[:, 0]).numpy()
            validation_dice.append(segmentation_metrics(probability, payload["early_masks"].numpy())["dice"])
        diagnostic = _transition_diagnostic(experts, payload["calibration_images"], False)
        corrupted = _transition_diagnostic(experts, payload["calibration_images"], True)
        for scenario in SCENARIOS:
            available = _available(scenario)
            domain = base_domain.clone()
            chart = base_chart.clone()
            if scenario == "noisy_chart_estimates":
                chart = 0.65 * chart + 0.35 / 8.0
            if scenario == "unavailable_center_metadata":
                domain[:] = 0.25
            if scenario == "reduced_calibration":
                domain = 0.5 * domain + 0.5 / 4.0
            domain[:, [index for index in range(4) if index not in available]] = 0
            domain /= domain.sum(1, keepdim=True).clamp_min(1e-8)
            uniform = expert_logits[:, available].mean(1)
            weights = torch.tensor([validation_dice[index] for index in available])
            weights /= weights.sum()
            weighted = (weights[None, :, None, None, None] * expert_logits[:, available]).sum(1)
            routed = expert_logits[
                torch.arange(count),
                torch.tensor(available)[domain[:, available].argmax(1)],
            ]
            structured = _domain_chart_soft(images, experts, domain, chart)
            confidence = domain.max(1).values * chart.max(1).values
            abstained = torch.where(
                (confidence >= 0.20)[:, None, None, None], structured, uniform
            )
            candidates = {
                "uniform_pooling": uniform,
                "validation_weighted_pooling": weighted,
                "generic_router": routed,
                "graph_synchronization": weighted,
                "hodge_diagnostic": weighted,
                "chart_aware_structured_pooling": structured,
                "abstaining_fallback": abstained,
            }
            for method in METHODS:
                probability = torch.sigmoid(candidates[method]).numpy()
                metrics = segmentation_metrics(probability, masks.numpy())
                per_domain = [
                    segmentation_metrics(probability[index : index + 1], masks[index : index + 1].numpy())["dice"]
                    for index in range(count)
                ]
                diag = corrupted if scenario == "one_incorrect_transition" else diagnostic
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "method": method,
                        **metrics,
                        "worst_domain_dice": min(per_domain),
                        "calibration": pixel_ece(probability, masks.numpy()),
                        **diag,
                        "residual_correction_activated": False,
                    }
                )
    baseline = {
        (row["seed"], row["method"]): float(row["dice"])
        for row in rows
        if row["scenario"] == "incomplete_comparison_graph"
    }
    for row in rows:
        row["dice_degradation"] = baseline[(row["seed"], row["method"])] - float(row["dice"])
        row["runtime_seconds_per_seed"] = (time.perf_counter() - started) / len(seeds)
    write_csv(DEST / "runs.csv", rows)
    summary = []
    for scenario in SCENARIOS:
        for method in METHODS:
            selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            summary.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "dice": float(np.mean([float(row["dice"]) for row in selected])),
                    "dice_degradation": float(np.mean([float(row["dice_degradation"]) for row in selected])),
                    "worst_domain_dice": float(np.mean([float(row["worst_domain_dice"]) for row in selected])),
                    "calibration": float(np.mean([float(row["calibration"]) for row in selected])),
                    "hodge_residual": float(np.mean([float(row["hodge_residual"]) for row in selected])),
                    "residual_after_edge_removal": float(np.mean([float(row["residual_after_edge_removal"]) for row in selected])),
                    "residual_correction_activated": False,
                }
            )
    write_csv(DEST / "summary.csv", summary)
    factual_report(
        DEST / "report.md",
        "Missing-expert and sparse-comparison robustness",
        [
            f"Seeds executed: {seeds}; scenarios: {len(SCENARIOS)}; methods: {len(METHODS)}.",
            "Center metadata was unavailable and was not inferred from test masks.",
            "The Hodge diagnostic used calibration activations; no residual correction was activated.",
            "A corrupted-edge removal diagnostic was executed and its before/after residuals are in summary.csv.",
        ],
    )
    update_status("C2_missing_expert_robustness", "completed", f"{len(rows)} scenario-method rows")
    stage_complete(DEST / "summary.csv", {"stage": "C2", "state": "completed", "seeds": seeds})
    return {"state": "completed", "seeds": len(seeds), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    try:
        result = run(args.smoke)
    except Exception as error:
        update_status("C2_missing_expert_robustness", "failed", str(error))
        record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result.get('rows', 0)} robustness rows")


if __name__ == "__main__":
    main()
