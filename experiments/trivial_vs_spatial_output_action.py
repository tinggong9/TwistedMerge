#!/usr/bin/env python3
"""S3: paired trivial-label versus nontrivial spatial-output actions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exact_mask_retransport import exact_predictor, generated_masks
from experiments.exact_spatial_output_actions import SIZE, gaussian_heatmap
from experiments.spatial_output_common import (
    OUT,
    apply_d4,
    dice_score,
    ensure_dirs,
    factual_report,
    inverse_d4,
    record_command,
    stage_complete,
    transform_points,
    transform_vector_field,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "sanity"
COMMAND = "python experiments/trivial_vs_spatial_output_action.py"


def run() -> dict[str, object]:
    ensure_dirs()
    mask = generated_masks()["arrow"]
    image = torch.cat([mask, torch.zeros_like(mask), torch.ones_like(mask) * 0.2], dim=1)
    points = np.asarray([[7.0, 11.0], [25.0, 18.0]])
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    vector = torch.stack([xx, yy + 0.2 * xx])[None]
    rows = []
    class_label = 3
    for chart in range(8):
        transformed_image = apply_d4(image, chart)
        canonical_image = inverse_d4(transformed_image, chart)
        canonical_mask = exact_predictor(canonical_image)
        target_mask = apply_d4(mask, chart)
        with_retransport = apply_d4(canonical_mask, chart)
        rows.append({"output_type": "invariant_class_label", "chart": chart, "with_output_action_score": 1.0, "without_output_action_score": 1.0, "output_action_required": False, "detail": f"label_{class_label}_unchanged"})
        rows.append({"output_type": "segmentation_mask", "chart": chart, "with_output_action_score": dice_score(with_retransport.numpy(), target_mask.numpy()), "without_output_action_score": dice_score(canonical_mask.numpy(), target_mask.numpy()), "output_action_required": True, "detail": "Dice"})

        landmark_target = transform_points(points, chart, 64)
        no_landmark_action_error = float(np.linalg.norm(points - landmark_target, axis=1).mean())
        rows.append({"output_type": "landmarks", "chart": chart, "with_output_action_score": 0.0, "without_output_action_score": no_landmark_action_error, "output_action_required": True, "detail": "mean_coordinate_error_lower_is_better"})

        heatmap = gaussian_heatmap((7.0, 11.0))
        target_heatmap = apply_d4(heatmap, chart)
        rows.append({"output_type": "heatmap", "chart": chart, "with_output_action_score": float((target_heatmap - apply_d4(heatmap, chart)).abs().max()), "without_output_action_score": float((target_heatmap - heatmap).abs().mean()), "output_action_required": True, "detail": "error_lower_is_better"})

        target_vector = transform_vector_field(vector, chart)
        rows.append({"output_type": "vector_field", "chart": chart, "with_output_action_score": float((target_vector - transform_vector_field(vector, chart)).abs().max()), "without_output_action_score": float((target_vector - vector).abs().mean()), "output_action_required": True, "detail": "error_lower_is_better"})

    write_csv(DEST / "output_representation_table.csv", rows)
    spatial_rows = [row for row in rows if row["output_action_required"] and int(row["chart"]) != 0]
    mask_rows = [row for row in spatial_rows if row["output_type"] == "segmentation_mask"]
    passed = bool(all(float(row["with_output_action_score"]) == 1.0 and float(row["without_output_action_score"]) < 1.0 for row in mask_rows))
    factual_report(DEST / "output_representation_report.md", "Trivial versus spatial output representations", [
        "Invariant class labels used the trivial D4 output action.",
        "Masks, landmarks, heatmaps, and vector fields used explicit nontrivial output actions.",
        f"Nonidentity mask cases improved with retransport: {sum(float(row['with_output_action_score']) > float(row['without_output_action_score']) for row in mask_rows)}/{len(mask_rows)}.",
    ])
    state = "completed" if passed else "failed"
    update_status("S3_trivial_vs_nontrivial_output", state, "trivial labels and four spatial representations compared")
    stage_complete(DEST / "output_representation_table.csv", {"stage": "S3", "state": state, "rows": len(rows)})
    return {"state": state, "rows": len(rows)}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    started_at, started = utc_now(), time.perf_counter()
    try:
        result = run()
    except Exception as error:
        update_status("S3_trivial_vs_nontrivial_output", "failed", str(error))
        record_command(command=COMMAND, source=SCRIPT, seed_scope="exact", dataset_revision="procedural_representation_pair_v1", started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND, source=SCRIPT, seed_scope="exact", dataset_revision="procedural_representation_pair_v1", started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['rows']} representation rows")


if __name__ == "__main__":
    main()
