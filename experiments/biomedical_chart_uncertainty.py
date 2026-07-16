#!/usr/bin/env python3
"""B3: chart uncertainty, perturbations, coverage, and abstention."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import D4EquivariantChartCNN  # noqa: E402
from experiments.spatial_output_common import (  # noqa: E402
    DEVICE,
    OUT,
    D4SymmetrizedUNet,
    TinyUNet,
    apply_d4,
    calibrate_temperature,
    chart_probabilities,
    dataset_checksum,
    dataset_ready,
    factual_report,
    hard_canonical_retransport,
    load_checkpoint,
    predict_logits,
    record_command,
    role_split,
    save_predictions_before_metrics,
    segmentation_metrics,
    soft_canonical_retransport,
    stage_complete,
    transformed_test,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "biomedical" / "uncertainty"
COMMAND = "python experiments/biomedical_chart_uncertainty.py"
PERTURBATIONS = ("none", "approximately_symmetric", "blur", "noise", "partial_crop", "color_shift", "missing_region")
METHODS = ("hard_chart_choice", "soft_group_marginalization", "uncertainty_weighted_retransport", "abstention_to_tta", "abstention_to_direct_equivariant", "supplied_chart_oracle")


def _load_models(seed: int) -> dict[str, Any]:
    canonical, _ = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_canonical_0.pt", TinyUNet(width=4))
    direct_base, _ = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_direct_base.pt", TinyUNet(width=4))
    chart, _ = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_chart_equivariant.pt", D4EquivariantChartCNN(3, width=4))
    return {"canonical": canonical, "direct": D4SymmetrizedUNet(direct_base).to(DEVICE), "tta": D4SymmetrizedUNet(canonical).to(DEVICE), "chart": chart}


def _perturb(images: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    if name == "none":
        return images
    if name == "approximately_symmetric":
        return torch.stack([apply_d4(images, chart) for chart in range(8)]).mean(0)
    if name == "blur":
        return F.avg_pool2d(images, 7, stride=1, padding=3)
    if name == "noise":
        generator = torch.Generator().manual_seed(seed)
        return (images + 0.18 * torch.randn(images.shape, generator=generator)).clamp(0, 1)
    if name == "partial_crop":
        values = images.clone()
        values[..., : values.shape[-2] // 4, :] = 0
        return values
    if name == "color_shift":
        scale = torch.tensor([1.2, 0.8, 1.05])[None, :, None, None]
        return (images * scale).clamp(0, 1)
    if name == "missing_region":
        values = images.clone()
        h, w = values.shape[-2:]
        values[..., h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 0
        return values
    raise ValueError(name)


def _temperature_and_threshold(seed: int, payload: dict[str, Any], models: dict[str, Any]) -> tuple[float, float]:
    calibration_images = torch.cat([apply_d4(payload["calibration_images"], chart) for chart in range(8)])
    calibration_charts = torch.cat([torch.full((len(payload["calibration_images"]),), chart, dtype=torch.long) for chart in range(8)])
    temperature = calibrate_temperature(predict_logits(models["chart"], calibration_images), calibration_charts)
    threshold_images, threshold_masks, _ = transformed_test({**payload, "test_images": payload["threshold_images"], "test_masks": payload["threshold_masks"]}, 340_000_000 + seed)
    probabilities = chart_probabilities(models["chart"], threshold_images, temperature)
    full = soft_canonical_retransport(threshold_images, [models["canonical"]], probabilities)
    fallback = predict_logits(models["tta"], threshold_images, 4)
    confidence = probabilities.max(1).values
    best_threshold, best_dice = 0.0, -1.0
    for threshold in np.linspace(0, 1, 21):
        candidate = torch.where((confidence >= threshold)[:, None, None, None], full, fallback)
        value = segmentation_metrics(torch.sigmoid(candidate).numpy(), threshold_masks.numpy())["dice"]
        if value > best_dice:
            best_threshold, best_dice = float(threshold), value
    return temperature, best_threshold


def run(smoke: bool = False) -> dict[str, Any]:
    discovery = OUT / "biomedical" / "discovery" / "summary.csv"
    if not dataset_ready() or not discovery.exists():
        update_status("B3_chart_uncertainty", "blocked", "B1 trained checkpoints unavailable")
        return {"state": "blocked", "seeds": 0}
    seeds = [0] if smoke else list(range(5))
    rows, coverage_rows = [], []
    plot_payload = None
    for seed in seeds:
        payload, models = role_split(seed), _load_models(seed)
        temperature, threshold = _temperature_and_threshold(seed, payload, models)
        base_images, target_masks, supplied_charts = transformed_test(payload, 340_010_000 + seed)
        saved = {}
        for perturbation in PERTURBATIONS:
            images = _perturb(base_images, perturbation, 340_020_000 + seed)
            probabilities = chart_probabilities(models["chart"], images, temperature)
            confidence = probabilities.max(1).values
            hard = hard_canonical_retransport(images, models["canonical"], probabilities.argmax(1))
            soft = soft_canonical_retransport(images, [models["canonical"]], probabilities)
            tta = predict_logits(models["tta"], images, 4)
            direct = predict_logits(models["direct"], images, 4)
            uncertainty = confidence[:, None, None, None] * soft + (1 - confidence[:, None, None, None]) * tta
            abstain_tta = torch.where((confidence >= threshold)[:, None, None, None], soft, tta)
            abstain_direct = torch.where((confidence >= threshold)[:, None, None, None], soft, direct)
            candidates = {
                "hard_chart_choice": hard,
                "soft_group_marginalization": soft,
                "uncertainty_weighted_retransport": uncertainty,
                "abstention_to_tta": abstain_tta,
                "abstention_to_direct_equivariant": abstain_direct,
                "supplied_chart_oracle": hard_canonical_retransport(images, models["canonical"], supplied_charts),
            }
            for method, logits in candidates.items():
                prediction = torch.sigmoid(logits).numpy()
                saved[f"{perturbation}__{method}"] = prediction
                rows.append({"seed": seed, "perturbation": perturbation, "method": method, **segmentation_metrics(prediction, target_masks.numpy()), "mean_chart_confidence": float(confidence.mean()), "chart_accuracy": float((probabilities.argmax(1) == supplied_charts).float().mean()), "threshold": threshold})
            soft_prediction = torch.sigmoid(soft).numpy()
            for cutoff in np.linspace(0, 1, 21):
                selected = (confidence >= cutoff).numpy()
                if selected.any():
                    metrics = segmentation_metrics(soft_prediction[selected], target_masks.numpy()[selected])
                    coverage_rows.append({"seed": seed, "perturbation": perturbation, "confidence_threshold": float(cutoff), "coverage": float(selected.mean()), "dice": metrics["dice"], "boundary_dice": metrics["boundary_dice"], "selective_risk": 1 - metrics["dice"], "chart_accuracy": float((probabilities.argmax(1)[selected] == supplied_charts[selected]).float().mean())})
            if seed == 0 and perturbation == "approximately_symmetric":
                plot_payload = (images, target_masks, confidence, candidates)
        audit = save_predictions_before_metrics(DEST / f"seed_{seed}_predictions.npz", saved, target_masks.numpy(), 340_030_000 + seed)
        for row in rows:
            if int(row["seed"]) == seed:
                row["prediction_sha256"] = audit["sha256"]
                row["prediction_hashes_unchanged"] = audit["candidate_hashes_unchanged"]
    write_csv(DEST / "runs.csv", rows)
    write_csv(DEST / "coverage.csv", coverage_rows)
    if plot_payload is not None:
        images, masks, confidence, candidates = plot_payload
        indices = torch.argsort(confidence)[: min(3, len(confidence))]
        figure, axes = plt.subplots(len(indices), 4, figsize=(8, 2 * len(indices)))
        for row, index in enumerate(indices):
            values = (images[index].permute(1,2,0).numpy(), masks[index,0].numpy(), torch.sigmoid(candidates["soft_group_marginalization"])[index,0].numpy(), torch.sigmoid(candidates["abstention_to_tta"])[index,0].numpy())
            for column, (value, title) in enumerate(zip(values, ("ambiguous input", "target", "soft", "abstain to TTA"), strict=True)):
                axes[row, column].imshow(value, cmap=None if value.ndim == 3 else "gray", vmin=0, vmax=1)
                axes[row, column].set_title(f"{title}; p={float(confidence[index]):.2f}" if column == 0 else title, fontsize=7)
                axes[row, column].axis("off")
        figure.tight_layout()
        figure.savefig(DEST / "plots" / "coverage_dice.pdf")
        plt.close(figure)
    factual_report(DEST / "report.md", "Biomedical chart uncertainty and abstention", [
        f"Seeds executed: {seeds}; perturbations: {PERTURBATIONS}.",
        f"Method-condition rows: {len(rows)}; coverage rows: {len(coverage_rows)}.",
        "Abstention thresholds were selected on the threshold role, not the final test set.",
        "Coverage-Dice, coverage-boundary-Dice, selective risk, chart accuracy, and mask calibration were recorded.",
    ])
    update_status("B3_chart_uncertainty", "completed", f"{len(seeds)} seeds and {len(PERTURBATIONS)} perturbations executed")
    stage_complete(DEST / "runs.csv", {"stage": "B3", "state": "completed", "seeds": seeds, "rows": len(rows)})
    return {"state": "completed", "seeds": len(seeds)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    try:
        result = run(args.smoke)
    except Exception as error:
        update_status("B3_chart_uncertainty", "failed", str(error))
        record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['seeds']} seeds")


if __name__ == "__main__":
    main()
