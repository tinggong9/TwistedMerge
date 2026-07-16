#!/usr/bin/env python3
"""S1: exact mask canonicalization and output retransport checks."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import (
    OUT,
    apply_d4,
    binary_boundary,
    compose_d4,
    dice_score,
    ensure_dirs,
    inverse_chart,
    inverse_d4,
    iou_score,
    record_command,
    stage_complete,
    surface_distances,
    update_status,
    utc_now,
    write_csv,
    wrong_inverse_d4,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "sanity"
COMMAND = "python experiments/exact_mask_retransport.py"


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (64, 64), 0)
    return image, ImageDraw.Draw(image)


def generated_masks() -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}

    image, draw = _canvas()
    draw.polygon([(7, 9), (49, 5), (56, 25), (32, 52), (11, 43)], fill=255)
    draw.rectangle((12, 13, 18, 19), fill=0)
    masks["asymmetric_polygon"] = image

    image, draw = _canvas()
    draw.ellipse((9, 11, 51, 53), fill=255)
    draw.ellipse((16, 17, 23, 24), fill=0)
    draw.rectangle((43, 38, 56, 44), fill=255)
    masks["off_center_circle_marker"] = image

    image, draw = _canvas()
    draw.polygon([(7, 28), (39, 28), (39, 18), (58, 35), (39, 52), (39, 42), (7, 42)], fill=255)
    masks["arrow"] = image

    image, draw = _canvas()
    font = ImageFont.load_default()
    draw.text((13, 8), "R7", font=font, fill=255, stroke_width=1)
    draw.rectangle((45, 43, 54, 57), fill=255)
    masks["symbol"] = image

    image, draw = _canvas()
    draw.rectangle((6, 7, 21, 28), fill=255)
    draw.ellipse((39, 13, 57, 31), fill=255)
    draw.polygon([(27, 40), (48, 55), (18, 58)], fill=255)
    masks["disconnected_components"] = image

    image, draw = _canvas()
    draw.line([(5, 57), (19, 12), (43, 44), (58, 7)], fill=255, width=1)
    draw.rectangle((10, 49, 13, 53), fill=255)
    masks["thin_boundary"] = image

    return {
        name: torch.from_numpy((np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32))[None, None]
        for name, mask in masks.items()
    }


def exact_predictor(canonical_image: torch.Tensor) -> torch.Tensor:
    """Executed predictor: foreground is encoded only in input channel zero."""

    return (canonical_image[:, :1] >= 0.5).float()


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = prediction.numpy()[0, 0]
    truth = target.numpy()[0, 0]
    hd95, _ = surface_distances(predicted, truth)
    boundary_equal = float(np.array_equal(binary_boundary(predicted), binary_boundary(truth)))
    return {
        "exact_equal": float(torch.equal(prediction, target)),
        "iou": iou_score(predicted, truth),
        "dice": dice_score(predicted, truth),
        "boundary_dice": boundary_equal,
        "hausdorff": 0.0 if torch.equal(prediction, target) else hd95,
    }


def run() -> dict[str, object]:
    ensure_dirs()
    rows = []
    example_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for mask_index, (name, mask) in enumerate(generated_masks().items()):
        image = torch.cat([mask, torch.zeros_like(mask), 0.25 * torch.ones_like(mask)], dim=1)
        for chart in range(8):
            transformed_image = apply_d4(image, chart)
            transformed_target = apply_d4(mask, chart)
            canonical = inverse_d4(transformed_image, chart)
            canonical_prediction = exact_predictor(canonical)
            candidates = {
                "correct_retransport": apply_d4(canonical_prediction, chart),
                "omit_output_retransport": canonical_prediction,
                "inverse_output_action": apply_d4(canonical_prediction, inverse_chart(chart)),
                "wrong_reflection_rotation_order": wrong_inverse_d4(canonical_prediction, chart),
                "random_chart": apply_d4(canonical_prediction, (chart + 3) % 8),
                "wrong_multiplication_table": apply_d4(canonical_prediction, compose_d4(chart, 1)),
            }
            for method, prediction in candidates.items():
                rows.append({"mask": name, "chart": chart, "method": method, **_metrics(prediction, transformed_target)})
            if chart == (mask_index + 1) % 8:
                example_rows.append((name, transformed_image[0].permute(1, 2, 0).numpy(), transformed_target[0, 0].numpy(), candidates["correct_retransport"][0, 0].numpy()))

    write_csv(DEST / "mask_runs.csv", rows)
    summary = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "cases": len(selected),
            "exact_rate": float(np.mean([row["exact_equal"] for row in selected])),
            "mean_iou": float(np.mean([row["iou"] for row in selected])),
            "mean_dice": float(np.mean([row["dice"] for row in selected])),
            "mean_boundary_dice": float(np.mean([row["boundary_dice"] for row in selected])),
        })
    write_csv(DEST / "mask_summary.csv", summary)
    correct = next(row for row in summary if row["method"] == "correct_retransport")
    controls = [row for row in summary if row["method"] != "correct_retransport"]
    claims = [
        {"claim": "nearest_neighbor_mask_action_exact", "passed": bool(correct["exact_rate"] == 1.0)},
        {"claim": "iou_dice_boundary_all_one", "passed": bool(correct["mean_iou"] == correct["mean_dice"] == correct["mean_boundary_dice"] == 1.0)},
        {"claim": "every_negative_control_detected", "passed": bool(all(row["exact_rate"] < 1.0 for row in controls))},
    ]
    write_csv(DEST / "mask_claims.csv", claims)

    figure, axes = plt.subplots(len(example_rows), 3, figsize=(7.2, 2.2 * len(example_rows)))
    for index, (name, image, target, prediction) in enumerate(example_rows):
        for column, (value, title) in enumerate(((image, "transformed input"), (target, "target"), (prediction, "retransported"))):
            axes[index, column].imshow(value, cmap=None if value.ndim == 3 else "gray", vmin=0, vmax=1)
            axes[index, column].set_title(f"{name}: {title}" if column == 0 else title, fontsize=7)
            axes[index, column].axis("off")
    figure.tight_layout()
    figure.savefig(DEST / "plots" / "mask_examples.pdf")
    plt.close(figure)

    report_lines = [
        "# Exact mask retransport",
        "",
        f"- Correct pipeline exact cases: {int(correct['exact_rate'] * correct['cases'])}/{correct['cases']}.",
        f"- Correct mean IoU, Dice, and exact-boundary Dice: {correct['mean_iou']:.6f}, {correct['mean_dice']:.6f}, {correct['mean_boundary_dice']:.6f}.",
        f"- Negative controls detected: {sum(row['exact_rate'] < 1.0 for row in controls)}/{len(controls)} aggregate controls.",
    ]
    (DEST / "mask_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    state = "completed" if all(bool(row["passed"]) for row in claims) else "failed"
    update_status("S1_exact_mask_retransport", state, "exact output action and asymmetric negative controls executed")
    stage_complete(DEST / "mask_runs.csv", {"stage": "S1", "state": state, "cases": len(rows)})
    return {"state": state, "cases": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    started_at = utc_now()
    started = time.perf_counter()
    try:
        result = run()
    except Exception as error:
        update_status("S1_exact_mask_retransport", "failed", str(error))
        record_command(command=COMMAND, source=SCRIPT, seed_scope="exact", dataset_revision="procedural_exact_masks_v1", started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND, source=SCRIPT, seed_scope="exact", dataset_revision="procedural_exact_masks_v1", started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['cases']} mask-action cases")


if __name__ == "__main__":
    main()
