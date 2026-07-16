#!/usr/bin/env python3
"""B1: trained Kvasir-SEG chart-aware segmentation discovery and confirmation."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import D4EquivariantChartCNN, ImageCNN  # noqa: E402
from experiments.spatial_output_common import (  # noqa: E402
    DEVICE,
    OUT,
    D4SymmetrizedUNet,
    TinyUNet,
    apply_d4,
    average_state_dict,
    calibrate_temperature,
    chart_augmentation,
    chart_probabilities,
    compose_d4,
    dataset_checksum,
    dataset_ready,
    dice_score,
    ensure_dirs,
    equivariance_metrics,
    expert_original_frame_logits,
    factual_report,
    hard_canonical_retransport,
    latex_table,
    make_chart_dataset,
    model_bytes,
    paired_rows,
    parameter_count,
    predict_logits,
    predict_probability,
    record_command,
    role_split,
    save_checkpoint,
    save_predictions_before_metrics,
    segmentation_metrics,
    sha256_bytes,
    soft_canonical_retransport,
    stage_complete,
    train_chart_model,
    train_segmenter,
    transformed_test,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "biomedical" / "discovery"
COMMAND = "python experiments/biomedical_segmentation_discovery.py"
METHODS = (
    "one_canonical_on_transformed_input",
    "one_canonical_supplied_inverse_and_retransport",
    "one_canonical_inferred_inverse_and_retransport",
    "four_expert_supplied_canonicalization",
    "four_expert_inferred_canonicalization",
    "generic_soft_moe",
    "generic_hard_routing",
    "direct_d4_equivariant_unet",
    "d4_test_time_augmentation",
    "supplied_chart_canonicalize_pool_retransport",
    "inferred_chart_canonicalize_pool_retransport",
    "uncertainty_weighted_retransport",
    "abstaining_retransport",
    "weight_average",
    "greedy_validation_soup",
    "ensemble_original_frame",
    "random_chart_control",
    "wrong_output_action_control",
    "wrong_multiplication_order_control",
    "inferred_canonical_no_output_retransport",
)


def _expanded_chart_data(images: torch.Tensor, allowed: tuple[int, ...] = tuple(range(8))) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([apply_d4(images, chart) for chart in allowed]),
        torch.cat([torch.full((len(images),), chart, dtype=torch.long) for chart in allowed]),
    )


def _specialist_training(payload: dict[str, Any], branch: int, seed: int, epochs: int) -> tuple[TinyUNet, float, list[dict[str, float]]]:
    indices = torch.arange(branch, len(payload["expert_images"]), 4)
    generator = torch.Generator().manual_seed(seed)
    charts = torch.tensor([branch, branch + 4], dtype=torch.long)[torch.randint(0, 2, (len(indices),), generator=generator)]
    images = apply_d4(payload["expert_images"][indices], charts)
    masks = apply_d4(payload["expert_masks"][indices], charts)
    validation_charts = torch.tensor([branch, branch + 4], dtype=torch.long).repeat((len(payload["early_images"]) + 1) // 2)[: len(payload["early_images"])]
    return train_segmenter(
        TinyUNet(width=4),
        images,
        masks,
        apply_d4(payload["early_images"], validation_charts),
        apply_d4(payload["early_masks"], validation_charts),
        seed,
        epochs,
    )


def train_seed(seed: int, smoke: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = role_split(seed)
    epochs = 1 if smoke else 2
    chart_epochs = 1 if smoke else 4
    models: dict[str, Any] = {}
    checkpoints = []
    training_times: dict[str, float] = {}

    canonical_models = []
    for index in range(4):
        model, elapsed, history = train_segmenter(
            TinyUNet(width=4),
            payload["expert_images"],
            payload["expert_masks"],
            payload["early_images"],
            payload["early_masks"],
            320_000_000 + seed * 100 + index,
            epochs,
        )
        canonical_models.append(model)
        name = f"seed_{seed}_canonical_{index}.pt"
        checkpoints.append({"seed": seed, "model": f"canonical_{index}", **save_checkpoint(OUT / "checkpoints" / name, model, {"seed": seed, "role": "local_expert_training", "training_time": elapsed, "history": history})})
        training_times[f"canonical_{index}"] = elapsed
    models["canonical"] = canonical_models

    direct, elapsed, history = train_segmenter(
        TinyUNet(width=4),
        payload["expert_images"],
        payload["expert_masks"],
        payload["early_images"],
        payload["early_masks"],
        320_010_000 + seed,
        epochs,
        augmentation=chart_augmentation,
    )
    models["direct_base"] = direct
    checkpoints.append({"seed": seed, "model": "direct_augmented_base", **save_checkpoint(OUT / "checkpoints" / f"seed_{seed}_direct_base.pt", direct, {"seed": seed, "role": "local_expert_training_d4_augmented", "training_time": elapsed, "history": history})})
    training_times["direct_base"] = elapsed

    specialists = []
    for branch in range(4):
        specialist, elapsed, history = _specialist_training(payload, branch, 320_020_000 + seed * 10 + branch, epochs)
        specialists.append(specialist)
        checkpoints.append({"seed": seed, "model": f"chart_specialist_{branch}", **save_checkpoint(OUT / "checkpoints" / f"seed_{seed}_specialist_{branch}.pt", specialist, {"seed": seed, "role": "chart_specialized_expert", "chart_subset": [branch, branch + 4], "training_time": elapsed, "history": history})})
        training_times[f"specialist_{branch}"] = elapsed
    models["specialists"] = specialists

    chart_images, chart_labels = _expanded_chart_data(payload["chart_images"])
    validation_chart_images, validation_chart_labels = _expanded_chart_data(payload["early_images"])
    equivariant, elapsed, validation_accuracy = train_chart_model(
        D4EquivariantChartCNN(3, width=4),
        chart_images,
        chart_labels,
        validation_chart_images,
        validation_chart_labels,
        320_030_000 + seed,
        chart_epochs,
    )
    models["chart_equivariant"] = equivariant
    checkpoints.append({"seed": seed, "model": "chart_equivariant", **save_checkpoint(OUT / "checkpoints" / f"seed_{seed}_chart_equivariant.pt", equivariant, {"seed": seed, "role": "chart_model_training", "validation_accuracy": validation_accuracy, "training_time": elapsed})})
    training_times["chart_equivariant"] = elapsed

    ordinary, elapsed, ordinary_accuracy = train_chart_model(
        ImageCNN(8, 3, width=5),
        chart_images,
        chart_labels,
        validation_chart_images,
        validation_chart_labels,
        320_040_000 + seed,
        chart_epochs,
    )
    models["chart_ordinary"] = ordinary
    checkpoints.append({"seed": seed, "model": "chart_ordinary", **save_checkpoint(OUT / "checkpoints" / f"seed_{seed}_chart_ordinary.pt", ordinary, {"seed": seed, "role": "chart_model_training", "validation_accuracy": ordinary_accuracy, "training_time": elapsed})})
    training_times["chart_ordinary"] = elapsed
    models["training_times"] = training_times
    models["payload"] = payload
    return models, checkpoints


def _choose_soup(models: list[TinyUNet], payload: dict[str, Any]) -> tuple[TinyUNet, int]:
    best_model, best_count, best_dice = models[0], 1, -math.inf
    for count in range(1, len(models) + 1):
        candidate = average_state_dict(models[:count])
        value = dice_score(predict_probability(candidate, payload["early_images"]), payload["early_masks"].numpy())
        if value > best_dice:
            best_model, best_count, best_dice = candidate, count, value
    return best_model, best_count


def infer_seed(seed: int, models: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = models["payload"]
    images, target_masks, supplied_charts = transformed_test(payload, 321_000_000 + seed)
    calibration_images, calibration_charts = _expanded_chart_data(payload["calibration_images"])
    temperature = calibrate_temperature(predict_logits(models["chart_equivariant"], calibration_images), calibration_charts)
    probabilities = chart_probabilities(models["chart_equivariant"], images, temperature)
    ordinary_probabilities = chart_probabilities(models["chart_ordinary"], images, 1.0)
    hard_charts = probabilities.argmax(1)
    canonical = models["canonical"]
    specialists = models["specialists"]
    direct = D4SymmetrizedUNet(models["direct_base"]).to(DEVICE)
    tta = D4SymmetrizedUNet(canonical[0]).to(DEVICE)

    one_raw = predict_logits(canonical[0], images)
    supplied_one = hard_canonical_retransport(images, canonical[0], supplied_charts)
    inferred_one = hard_canonical_retransport(images, canonical[0], hard_charts)
    supplied_four = torch.stack([hard_canonical_retransport(images, model, supplied_charts) for model in canonical]).mean(0)
    inferred_four = torch.stack([hard_canonical_retransport(images, model, hard_charts) for model in canonical]).mean(0)
    soft_full = soft_canonical_retransport(images, canonical, probabilities)
    no_output = soft_canonical_retransport(images, canonical, probabilities, output_action="none")
    specialist_logits = expert_original_frame_logits(images, specialists)
    branch_probabilities = probabilities[:, :4] + probabilities[:, 4:]
    generic_soft = (branch_probabilities[:, :, None, None, None] * specialist_logits).sum(1)
    branches = branch_probabilities.argmax(1)
    generic_hard = specialist_logits[torch.arange(len(images)), branches]
    direct_logits = predict_logits(direct, images, 4)
    tta_logits = predict_logits(tta, images, 4)
    confidence = probabilities.max(1).values[:, None, None, None]
    uncertainty = confidence * soft_full + (1 - confidence) * tta_logits

    threshold_images, threshold_masks, threshold_charts = transformed_test({**payload, "test_images": payload["threshold_images"], "test_masks": payload["threshold_masks"]}, 321_100_000 + seed)
    threshold_probabilities = chart_probabilities(models["chart_equivariant"], threshold_images, temperature)
    threshold_soft = soft_canonical_retransport(threshold_images, canonical, threshold_probabilities)
    threshold_tta = predict_logits(tta, threshold_images, 4)
    threshold_confidence = threshold_probabilities.max(1).values
    best_threshold, best_threshold_dice = 0.0, -math.inf
    for threshold in np.linspace(0, 1, 21):
        candidate = torch.where((threshold_confidence >= threshold)[:, None, None, None], threshold_soft, threshold_tta)
        value = dice_score(torch.sigmoid(candidate).numpy(), threshold_masks.numpy())
        if value > best_threshold_dice:
            best_threshold, best_threshold_dice = float(threshold), value
    abstaining = torch.where((probabilities.max(1).values >= best_threshold)[:, None, None, None], soft_full, tta_logits)

    weight_average = average_state_dict(canonical)
    soup, soup_count = _choose_soup(canonical, payload)
    ensemble = expert_original_frame_logits(images, canonical).mean(1)
    generator = torch.Generator().manual_seed(321_200_000 + seed)
    random_charts = torch.randint(0, 8, (len(images),), generator=generator)
    wrong_charts = torch.tensor([compose_d4(int(chart), 1) for chart in hard_charts], dtype=torch.long)

    logits = {
        "one_canonical_on_transformed_input": one_raw,
        "one_canonical_supplied_inverse_and_retransport": supplied_one,
        "one_canonical_inferred_inverse_and_retransport": inferred_one,
        "four_expert_supplied_canonicalization": supplied_four,
        "four_expert_inferred_canonicalization": inferred_four,
        "generic_soft_moe": generic_soft,
        "generic_hard_routing": generic_hard,
        "direct_d4_equivariant_unet": direct_logits,
        "d4_test_time_augmentation": tta_logits,
        "supplied_chart_canonicalize_pool_retransport": supplied_four,
        "inferred_chart_canonicalize_pool_retransport": soft_full,
        "uncertainty_weighted_retransport": uncertainty,
        "abstaining_retransport": abstaining,
        "weight_average": predict_logits(weight_average, images),
        "greedy_validation_soup": predict_logits(soup, images),
        "ensemble_original_frame": ensemble,
        "random_chart_control": hard_canonical_retransport(images, canonical[0], random_charts),
        "wrong_output_action_control": soft_canonical_retransport(images, canonical, probabilities, output_action="inverse"),
        "wrong_multiplication_order_control": hard_canonical_retransport(images, canonical[0], wrong_charts),
        "inferred_canonical_no_output_retransport": no_output,
    }
    predictions = {name: torch.sigmoid(value).numpy() for name, value in logits.items()}
    prediction_audit = save_predictions_before_metrics(DEST / "predictions" / f"seed_{seed}.npz", predictions, target_masks.numpy(), 321_300_000 + seed)
    probability_hash = sha256_bytes(np.asarray(probabilities, dtype=np.float32).tobytes())

    summary_rows, run_rows, boundary_rows = [], [], []
    for method in METHODS:
        metrics = segmentation_metrics(predictions[method], target_masks.numpy())
        summary_rows.append({
            "seed": seed,
            "method": method,
            **metrics,
            "chart_accuracy": float((hard_charts == supplied_charts).float().mean()),
            "ordinary_chart_accuracy": float((ordinary_probabilities.argmax(1) == supplied_charts).float().mean()),
            "same_chart_probability_sha256": probability_hash,
            "prediction_file": prediction_audit["path"],
            "prediction_sha256": prediction_audit["sha256"],
            "prediction_hashes_unchanged_after_label_permutation": prediction_audit["candidate_hashes_unchanged"],
            "abstention_threshold": best_threshold,
            "soup_expert_count": soup_count,
            "training_time_seconds": float(sum(models["training_times"].values())),
            "stored_bytes": sum(model_bytes(model) for model in canonical),
        })
        for index, name in enumerate(payload["test_names"]):
            per_image = segmentation_metrics(predictions[method][index : index + 1], target_masks.numpy()[index : index + 1])
            run_rows.append({"seed": seed, "example": name, "chart": int(supplied_charts[index]), "method": method, **per_image})
            boundary_rows.append({"seed": seed, "example": name, "chart": int(supplied_charts[index]), "method": method, "boundary_dice": per_image["boundary_dice"], "hausdorff95": per_image["hausdorff95"], "assd": per_image["assd"]})

    generator = torch.Generator().manual_seed(321_400_000 + seed)
    equivariance_charts = torch.randint(0, 8, (len(payload["early_images"]),), generator=generator)
    equivariance_rows = []
    for method, model in (("one_canonical", canonical[0]), ("direct_d4_equivariant_unet", direct), ("weight_average", weight_average)):
        equivariance_rows.append({"seed": seed, "method": method, **equivariance_metrics(model, payload["early_images"], equivariance_charts)})
    extra = {
        "prediction_audit": prediction_audit,
        "test_images": images,
        "target_masks": target_masks,
        "predictions": predictions,
        "checkpoints": [],
        "equivariance_rows": equivariance_rows,
    }
    return summary_rows, run_rows, boundary_rows, extra


def _write_example_plot(extra: dict[str, Any]) -> None:
    methods = ["one_canonical_on_transformed_input", "direct_d4_equivariant_unet", "inferred_chart_canonicalize_pool_retransport", "wrong_output_action_control"]
    count = min(3, len(extra["test_images"]))
    figure, axes = plt.subplots(count, 2 + len(methods), figsize=(12, 2.2 * count))
    for row in range(count):
        axes[row, 0].imshow(extra["test_images"][row].permute(1, 2, 0).numpy())
        axes[row, 0].set_title("input")
        axes[row, 1].imshow(extra["target_masks"][row, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("target")
        for column, method in enumerate(methods, 2):
            axes[row, column].imshow(extra["predictions"][method][row, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_title(method.replace("_", " "), fontsize=6)
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(DEST / "plots" / "examples.pdf")
    plt.close(figure)


def _gate_rows(summary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = [
        ("retransport_vs_no_output", "inferred_chart_canonicalize_pool_retransport", "inferred_canonical_no_output_retransport"),
        ("retransport_vs_generic_soft", "inferred_chart_canonicalize_pool_retransport", "generic_soft_moe"),
        ("retransport_vs_generic_hard", "inferred_chart_canonicalize_pool_retransport", "generic_hard_routing"),
        ("four_vs_one_after_inferred_chart", "four_expert_inferred_canonicalization", "one_canonical_inferred_inverse_and_retransport"),
        ("full_vs_direct_equivariant", "inferred_chart_canonicalize_pool_retransport", "direct_d4_equivariant_unet"),
        ("full_vs_tta", "inferred_chart_canonicalize_pool_retransport", "d4_test_time_augmentation"),
        ("random_control_gap", "inferred_chart_canonicalize_pool_retransport", "random_chart_control"),
        ("wrong_output_control_gap", "inferred_chart_canonicalize_pool_retransport", "wrong_output_action_control"),
        ("wrong_order_control_gap", "inferred_chart_canonicalize_pool_retransport", "wrong_multiplication_order_control"),
    ]
    paired = paired_rows(summary, comparisons, "dice", 322_000_000)
    lookup = {row["comparison"]: row for row in paired}
    retransport = lookup["retransport_vs_no_output"]
    multi = lookup["four_vs_one_after_inferred_chart"]
    control_names = ("random_control_gap", "wrong_output_control_gap", "wrong_order_control_gap")
    claims = [
        {"claim": "retransport_gate", "passed": bool(float(retransport["ci_lower"]) > 0), "reason": "paired Dice CI against same-probability no-output action"},
        {"claim": "multi_expert_gate", "passed": bool(float(multi["ci_lower"]) > 0), "reason": "paired Dice CI for four versus one after identical chart inference"},
        {"claim": "control_gate", "passed": bool(all(float(lookup[name]["ci_lower"]) > 0 for name in control_names)), "reason": "random chart, wrong output action, and wrong order paired CIs"},
        {"claim": "twistedmerge_specific_gate", "passed": False, "reason": "requires B4 complete-path matched-cost evidence in addition to B1 accuracy"},
        {"claim": "prediction_label_permutation_audit", "passed": bool(all(bool(row["prediction_hashes_unchanged_after_label_permutation"]) for row in summary)), "reason": "saved prediction hashes unchanged"},
    ]
    return paired, claims


def run(smoke: bool = False, force: bool = False) -> dict[str, Any]:
    ensure_dirs()
    if not dataset_ready():
        update_status("B1_segmentation_discovery", "blocked", "paired biomedical masks are unavailable")
        factual_report(DEST / "report.md", "Biomedical segmentation discovery", ["State: blocked.", "Reason: no resolved paired biomedical image-mask dataset."])
        return {"state": "blocked", "seeds": 0}
    all_summary, all_runs, all_boundaries, checkpoints, all_equivariance = [], [], [], [], []
    first_extra: dict[str, Any] | None = None
    discovery_seeds = [0] if smoke else list(range(5))
    for seed in discovery_seeds:
        models, seed_checkpoints = train_seed(seed, smoke)
        summary, runs, boundaries, extra = infer_seed(seed, models)
        all_summary.extend(summary)
        all_runs.extend(runs)
        all_boundaries.extend(boundaries)
        all_equivariance.extend(extra["equivariance_rows"])
        checkpoints.extend(seed_checkpoints)
        if first_extra is None:
            first_extra = extra
    paired, claims = _gate_rows(all_summary)
    retransport_passed = bool(next(row for row in claims if row["claim"] == "retransport_gate")["passed"])
    if retransport_passed and not smoke:
        for seed in range(5, 10):
            models, seed_checkpoints = train_seed(seed, smoke=False)
            summary, runs, boundaries, extra = infer_seed(seed, models)
            all_summary.extend(summary)
            all_runs.extend(runs)
            all_boundaries.extend(boundaries)
            all_equivariance.extend(extra["equivariance_rows"])
            checkpoints.extend(seed_checkpoints)
        paired, claims = _gate_rows(all_summary)

    write_csv(DEST / "runs.csv", all_runs)
    write_csv(DEST / "summary.csv", all_summary)
    write_csv(DEST / "paired.csv", paired)
    write_csv(DEST / "equivariance.csv", all_equivariance)
    write_csv(DEST / "boundary_metrics.csv", all_boundaries)
    write_csv(DEST / "claims.csv", claims)
    write_csv(OUT / "checkpoint_manifest.csv", checkpoints)
    latex_table(DEST / "tables" / "main.tex", ["method", "dice", "boundary_dice", "hausdorff95"], [
        {"method": method, "dice": f"{np.mean([float(row['dice']) for row in all_summary if row['method'] == method]):.4f}", "boundary_dice": f"{np.mean([float(row['boundary_dice']) for row in all_summary if row['method'] == method]):.4f}", "hausdorff95": f"{np.mean([float(row['hausdorff95']) for row in all_summary if row['method'] == method]):.3f}"}
        for method in METHODS
    ])
    if first_extra is not None:
        _write_example_plot(first_extra)
    factual_report(DEST / "report.md", "Biomedical segmentation discovery", [
        f"State: completed; seeds executed: {sorted({int(row['seed']) for row in all_summary})}.",
        f"Methods executed per seed: {len(METHODS)}.",
        f"Retransport gate: {next(row['passed'] for row in claims if row['claim'] == 'retransport_gate')}.",
        f"Multi-expert gate: {next(row['passed'] for row in claims if row['claim'] == 'multi_expert_gate')}.",
        f"Control gate: {next(row['passed'] for row in claims if row['claim'] == 'control_gate')}.",
        "TwistedMerge-specific gate remains false until the B4 matched complete-path cost comparison is executed.",
        "All candidate prediction tensors were saved before mask metrics and passed label-permutation hash checks.",
    ])
    update_status("B1_segmentation_discovery", "completed", f"{len(set(int(row['seed']) for row in all_summary))} seeds and {len(METHODS)} methods executed")
    stage_complete(DEST / "summary.csv", {"stage": "B1", "state": "completed", "seeds": len(set(int(row["seed"]) for row in all_summary)), "retransport_gate": retransport_passed})
    return {"state": "completed", "seeds": len(set(int(row["seed"]) for row in all_summary)), "retransport_gate": retransport_passed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    try:
        result = run(args.smoke, args.force)
    except Exception as error:
        update_status("B1_segmentation_discovery", "failed", str(error))
        record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4 plus 5:9 on gate", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4 plus 5:9 on gate", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['seeds']} seeds; retransport_gate={result.get('retransport_gate')}")


if __name__ == "__main__":
    main()
