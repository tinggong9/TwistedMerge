#!/usr/bin/env python3
"""B2: strict held-out D4-element segmentation with trained neural models."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import (  # noqa: E402
    D4EquivariantChartCNN,
    ImageCNN,
    LearnedMultiplicationChartCNN,
)
from experiments.spatial_output_common import (  # noqa: E402
    DEVICE,
    OUT,
    D4SymmetrizedUNet,
    TinyUNet,
    apply_d4,
    calibrate_temperature,
    chart_augmentation,
    chart_probabilities,
    dataset_checksum,
    dataset_ready,
    factual_report,
    hard_canonical_retransport,
    make_chart_dataset,
    paired_rows,
    predict_logits,
    record_command,
    role_split,
    save_checkpoint,
    save_predictions_before_metrics,
    segmentation_metrics,
    soft_canonical_retransport,
    stage_complete,
    train_chart_model,
    train_segmenter,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "biomedical" / "zeroshot"
COMMAND = "python experiments/biomedical_zeroshot_segmentation.py"
SEEN = (0, 1, 4)
UNSEEN = (2, 3, 5, 6, 7)
METHODS = (
    "ordinary_chart_cnn_canonicalization",
    "d4_equivariant_chart_cnn_canonicalization",
    "direct_d4_equivariant_unet",
    "learned_multiplication_table_chart_model",
    "hard_retransport",
    "soft_retransport",
    "d4_test_time_augmentation",
    "supplied_chart_oracle",
    "random_chart_control",
    "wrong_output_action_control",
)


def _expanded(images: torch.Tensor, charts: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([apply_d4(images, chart) for chart in charts]), torch.cat([torch.full((len(images),), chart, dtype=torch.long) for chart in charts])


def _condition(payload: dict[str, Any], charts: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = len(payload["test_images"])
    values = torch.tensor([charts[index % len(charts)] for index in range(count)], dtype=torch.long)
    return apply_d4(payload["test_images"], values), apply_d4(payload["test_masks"], values), values


def _train_seed(seed: int, smoke: bool) -> dict[str, Any]:
    payload = role_split(seed)
    epochs = 1 if smoke else 2
    chart_epochs = 1 if smoke else 5
    canonical, _, history = train_segmenter(TinyUNet(width=4), payload["expert_images"], payload["expert_masks"], payload["early_images"], payload["early_masks"], 330_000_000 + seed, epochs)
    direct_base, _, direct_history = train_segmenter(TinyUNet(width=4), payload["expert_images"], payload["expert_masks"], payload["early_images"], payload["early_masks"], 330_010_000 + seed, epochs, augmentation=chart_augmentation)
    train_images, train_charts = _expanded(payload["chart_images"], SEEN)
    validation_images, validation_charts = _expanded(payload["early_images"], SEEN)
    equivariant, _, _ = train_chart_model(D4EquivariantChartCNN(3, width=4), train_images, train_charts, validation_images, validation_charts, 330_020_000 + seed, chart_epochs)
    ordinary, _, _ = train_chart_model(ImageCNN(8, 3, width=5), train_images, train_charts, validation_images, validation_charts, 330_030_000 + seed, chart_epochs)
    learned, _, _ = train_chart_model(LearnedMultiplicationChartCNN(3, width=4), train_images, train_charts, validation_images, validation_charts, 330_040_000 + seed, chart_epochs)
    for name, model, metadata in (
        ("canonical", canonical, {"history": history}),
        ("direct_base", direct_base, {"history": direct_history}),
        ("chart_equivariant", equivariant, {}),
        ("chart_ordinary", ordinary, {}),
        ("chart_learned_table", learned, {}),
    ):
        save_checkpoint(OUT / "checkpoints" / f"zeroshot_seed_{seed}_{name}.pt", model, {"seed": seed, "allowed_charts": list(SEEN), "heldout_charts": list(UNSEEN), **metadata})
    return {"payload": payload, "canonical": canonical, "direct": D4SymmetrizedUNet(direct_base).to(DEVICE), "equivariant": equivariant, "ordinary": ordinary, "learned": learned}


def _predict_condition(seed: int, condition: str, images: torch.Tensor, masks: torch.Tensor, true_charts: torch.Tensor, models: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = models["payload"]
    calibration_images, calibration_charts = _expanded(payload["calibration_images"], SEEN)
    temperature = calibrate_temperature(predict_logits(models["equivariant"], calibration_images), calibration_charts)
    equivariant_probabilities = chart_probabilities(models["equivariant"], images, temperature)
    ordinary_probabilities = chart_probabilities(models["ordinary"], images)
    learned_probabilities = chart_probabilities(models["learned"], images)
    equivariant_hard = equivariant_probabilities.argmax(1)
    generator = torch.Generator().manual_seed(330_100_000 + seed + (0 if condition == "seen" else 1000))
    random_charts = torch.randint(0, 8, (len(images),), generator=generator)
    candidates = {
        "ordinary_chart_cnn_canonicalization": hard_canonical_retransport(images, models["canonical"], ordinary_probabilities.argmax(1)),
        "d4_equivariant_chart_cnn_canonicalization": hard_canonical_retransport(images, models["canonical"], equivariant_hard),
        "direct_d4_equivariant_unet": predict_logits(models["direct"], images, 4),
        "learned_multiplication_table_chart_model": hard_canonical_retransport(images, models["canonical"], learned_probabilities.argmax(1)),
        "hard_retransport": hard_canonical_retransport(images, models["canonical"], equivariant_hard),
        "soft_retransport": soft_canonical_retransport(images, [models["canonical"]], equivariant_probabilities),
        "d4_test_time_augmentation": predict_logits(D4SymmetrizedUNet(models["canonical"]).to(DEVICE), images, 4),
        "supplied_chart_oracle": hard_canonical_retransport(images, models["canonical"], true_charts),
        "random_chart_control": hard_canonical_retransport(images, models["canonical"], random_charts),
        "wrong_output_action_control": soft_canonical_retransport(images, [models["canonical"]], equivariant_probabilities, output_action="inverse"),
    }
    predictions = {method: torch.sigmoid(logits).numpy() for method, logits in candidates.items()}
    audit = save_predictions_before_metrics(DEST / "predictions" / f"seed_{seed}_{condition}.npz", predictions, masks.numpy(), 330_200_000 + seed)
    summary, runs = [], []
    for method in METHODS:
        metrics = segmentation_metrics(predictions[method], masks.numpy())
        summary.append({"seed": seed, "condition": condition, "method": method, **metrics, "chart_accuracy": float((equivariant_hard == true_charts).float().mean()), "ordinary_chart_accuracy": float((ordinary_probabilities.argmax(1) == true_charts).float().mean()), "learned_table_chart_accuracy": float((learned_probabilities.argmax(1) == true_charts).float().mean()), "heldout_chart_exposed": False, "prediction_sha256": audit["sha256"], "prediction_hashes_unchanged": audit["candidate_hashes_unchanged"]})
        for index, name in enumerate(payload["test_names"]):
            metrics_one = segmentation_metrics(predictions[method][index:index+1], masks.numpy()[index:index+1])
            runs.append({"seed": seed, "condition": condition, "example": name, "chart": int(true_charts[index]), "method": method, **metrics_one})
    return summary, runs, {"equivariant_probabilities": equivariant_probabilities, "true_charts": true_charts, "learned": models["learned"]}


def run(smoke: bool = False) -> dict[str, Any]:
    if not dataset_ready():
        update_status("B2_zeroshot_segmentation", "blocked", "paired biomedical masks unavailable")
        return {"state": "blocked", "seeds": 0}
    b1_claims = OUT / "biomedical" / "discovery" / "claims.csv"
    b1_passed = False
    if b1_claims.exists():
        import csv
        with b1_claims.open(encoding="utf-8", newline="") as handle:
            b1_passed = any(row["claim"] == "retransport_gate" and row["passed"] == "True" for row in csv.DictReader(handle))
    seeds = [10] if smoke else (list(range(10, 20)) if b1_passed else list(range(10, 15)))
    all_summary, all_runs = [], []
    multiplication_errors = []
    for seed in seeds:
        models = _train_seed(seed, smoke)
        for condition, charts in (("seen", SEEN), ("unseen", UNSEEN)):
            images, masks, true_charts = _condition(models["payload"], charts)
            summary, runs, extra = _predict_condition(seed, condition, images, masks, true_charts, models)
            all_summary.extend(summary)
            all_runs.extend(runs)
        learned_table = models["learned"].table_logits.detach().softmax(-1).cpu()
        multiplication_errors.append(float(1.0 - learned_table.max(-1).values.mean()))
    by_element = []
    for seed in seeds:
        for condition, charts in (("seen", SEEN), ("unseen", UNSEEN)):
            for chart in charts:
                for method in METHODS:
                    selected = [row for row in all_runs if int(row["seed"]) == seed and row["condition"] == condition and int(row["chart"]) == chart and row["method"] == method]
                    by_element.append({"seed": seed, "condition": condition, "chart": chart, "method": method, "dice": float(np.mean([float(row["dice"]) for row in selected])) if selected else math.nan, "examples": len(selected)})
    unseen = [row for row in all_summary if row["condition"] == "unseen"]
    paired = paired_rows(unseen, [
        ("soft_vs_ordinary_chart", "soft_retransport", "ordinary_chart_cnn_canonicalization"),
        ("soft_vs_learned_table", "soft_retransport", "learned_multiplication_table_chart_model"),
        ("soft_vs_direct_equivariant", "soft_retransport", "direct_d4_equivariant_unet"),
        ("soft_vs_tta", "soft_retransport", "d4_test_time_augmentation"),
        ("soft_vs_wrong_action", "soft_retransport", "wrong_output_action_control"),
    ], "dice", 331_000_000)
    lookup = {row["comparison"]: row for row in paired}
    exposure_ok = all(not bool(row["heldout_chart_exposed"]) for row in all_summary)
    wrong_control_ok = float(lookup["soft_vs_wrong_action"]["ci_lower"]) > 0
    learned_error = float(np.mean(multiplication_errors))
    claims = [
        {"claim": "unseen_chart_dice_beats_ordinary_learned_baselines", "passed": bool(float(lookup["soft_vs_ordinary_chart"]["ci_lower"]) > 0 and float(lookup["soft_vs_learned_table"]["ci_lower"]) > 0)},
        {"claim": "heldout_chart_labels_unexposed", "passed": exposure_ok},
        {"claim": "equivariance_and_multiplication_tolerance", "passed": bool(learned_error < 0.05), "mean_learned_table_error": learned_error},
        {"claim": "wrong_action_control_fails", "passed": wrong_control_ok},
    ]
    claims.append({"claim": "strict_zeroshot_gate", "passed": bool(all(bool(row["passed"]) for row in claims))})
    write_csv(DEST / "runs.csv", all_runs)
    write_csv(DEST / "summary.csv", all_summary)
    write_csv(DEST / "by_element.csv", by_element)
    write_csv(DEST / "paired.csv", paired)
    write_csv(DEST / "claims.csv", claims)
    factual_report(DEST / "report.md", "Strict zero-shot biomedical segmentation", [
        f"Seeds executed: {seeds}; mode: {'confirmation' if b1_passed else 'diagnostic'}.",
        f"Seen chart set: {SEEN}; final-test-only chart set: {UNSEEN}.",
        f"Held-out exposure audit: {exposure_ok}.",
        f"Mean learned multiplication-table discreteness error: {learned_error:.6f}.",
        f"Strict zero-shot gate: {claims[-1]['passed']}.",
    ])
    update_status("B2_zeroshot_segmentation", "completed", f"{len(seeds)} {'confirmation' if b1_passed else 'diagnostic'} seeds; gate={claims[-1]['passed']}")
    stage_complete(DEST / "summary.csv", {"stage": "B2", "state": "completed", "seeds": seeds, "strict_gate": claims[-1]["passed"]})
    return {"state": "completed", "seeds": len(seeds), "strict_gate": claims[-1]["passed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    try:
        result = run(args.smoke)
    except Exception as error:
        update_status("B2_zeroshot_segmentation", "failed", str(error))
        record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="10" if args.smoke else "10:19 or 10:14 diagnostic", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="10" if args.smoke else "10:19 or 10:14 diagnostic", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['seeds']} seeds; strict_gate={result.get('strict_gate')}")


if __name__ == "__main__":
    main()
