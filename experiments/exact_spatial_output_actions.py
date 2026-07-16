#!/usr/bin/env python3
"""S2: exact heatmap, landmark, point-set, and vector-field actions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import (
    OUT,
    apply_d4,
    compose_d4,
    d4_matrix,
    ensure_dirs,
    factual_report,
    inverse_chart,
    record_command,
    stage_complete,
    transform_points,
    transform_vector_field,
    update_status,
    utc_now,
    write_csv,
    wrong_vector_field_coordinates_only,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "sanity"
COMMAND = "python experiments/exact_spatial_output_actions.py"
SIZE = 33


def gaussian_heatmap(point: tuple[float, float], sigma: float = 2.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    value = torch.exp(-((xx - point[0]) ** 2 + (yy - point[1]) ** 2) / (2 * sigma**2))
    return value[None, None]


def run() -> dict[str, object]:
    ensure_dirs()
    rows: list[dict[str, object]] = []
    point = np.asarray([[7.0, 11.0]])
    heatmap = gaussian_heatmap((7.0, 11.0))
    ordered = np.asarray([[4.0, 5.0], [27.0, 6.0], [25.0, 28.0], [8.0, 24.0]])
    point_set = np.asarray([[6.0, 8.0], [17.0, 4.0], [26.0, 21.0]])
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIZE), torch.linspace(-1, 1, SIZE), indexing="ij")
    field = torch.stack([xx + 0.3 * yy, -0.2 * xx + yy])[None]
    orientation = torch.stack([torch.cos(torch.pi * xx), torch.sin(torch.pi * yy)])[None]

    for chart in range(8):
        inverse = inverse_chart(chart)
        transformed_point = transform_points(point, chart, SIZE)
        heatmap_expected = gaussian_heatmap(tuple(transformed_point[0]))
        heatmap_actual = apply_d4(heatmap, chart)
        rows.append({"output_type": "gaussian_heatmap", "chart": chart, "check": "coordinate_transform", "error": float((heatmap_actual - heatmap_expected).abs().max()), "passed": bool(torch.allclose(heatmap_actual, heatmap_expected, atol=2e-4))})

        ordered_transformed = transform_points(ordered, chart, SIZE)
        recovered_ordered = transform_points(ordered_transformed, inverse, SIZE)
        rows.append({"output_type": "ordered_keypoints", "chart": chart, "check": "inverse_law", "error": float(np.abs(recovered_ordered - ordered).max()), "passed": bool(np.allclose(recovered_ordered, ordered))})

        point_set_transformed = transform_points(point_set, chart, SIZE)
        sorted_original = np.asarray(sorted(map(tuple, np.round(point_set, 8))))
        sorted_recovered = np.asarray(sorted(map(tuple, np.round(transform_points(point_set_transformed, inverse, SIZE), 8))))
        rows.append({"output_type": "unoriented_point_set", "chart": chart, "check": "inverse_law", "error": float(np.abs(sorted_recovered - sorted_original).max()), "passed": bool(np.allclose(sorted_recovered, sorted_original))})

        for name, value in (("displacement_field", field), ("orientation_field", orientation)):
            transformed = transform_vector_field(value, chart)
            recovered = transform_vector_field(transformed, inverse)
            rows.append({"output_type": name, "chart": chart, "check": "coordinate_and_component_inverse", "error": float((recovered - value).abs().max()), "passed": bool(torch.allclose(recovered, value, atol=1e-6))})
            wrong = wrong_vector_field_coordinates_only(value, chart)
            rows.append({"output_type": name, "chart": chart, "check": "negative_coordinates_only", "error": float((wrong - transformed).abs().max()), "passed": bool(chart == 0 or not torch.allclose(wrong, transformed, atol=1e-6))})

    for left in range(8):
        for right in range(8):
            product = compose_d4(left, right)
            matrix_error = float(np.abs(d4_matrix(left) @ d4_matrix(right) - d4_matrix(product)).max())
            field_error = float((transform_vector_field(transform_vector_field(field, right), left) - transform_vector_field(field, product)).abs().max())
            rows.append({"output_type": "vector_field", "chart": f"{left}*{right}", "check": "composition_law", "error": field_error, "passed": bool(matrix_error == 0 and field_error < 1e-6)})

    wrong_sign_detected = bool(not torch.allclose(transform_vector_field(field, 4), apply_d4(field, 4)))
    wrong_permutation_detected = bool(not np.allclose(transform_points(ordered, 1, SIZE), transform_points(ordered[[1, 0, 2, 3]], 1, SIZE)))
    noncommuting_left, noncommuting_right = 4, 1
    correct_product = compose_d4(noncommuting_left, noncommuting_right)
    reversed_product = compose_d4(noncommuting_right, noncommuting_left)
    order_probe = torch.zeros_like(field)
    order_probe[0, 0, 3, 7] = 1.0
    order_probe[0, 1, 3, 7] = 0.25
    correct_order = transform_vector_field(order_probe, correct_product)
    reversed_order = transform_vector_field(order_probe, reversed_product)
    wrong_order_detected = bool(not torch.allclose(correct_order, reversed_order, atol=1e-6))
    rows.extend([
        {"output_type": "vector_field", "chart": 4, "check": "negative_wrong_reflection_sign_detected", "error": float(wrong_sign_detected), "passed": wrong_sign_detected},
        {"output_type": "ordered_keypoints", "chart": 1, "check": "negative_wrong_landmark_permutation_detected", "error": float(wrong_permutation_detected), "passed": wrong_permutation_detected},
        {"output_type": "vector_field", "chart": f"{noncommuting_left}*{noncommuting_right}", "check": "negative_reversed_multiplication_order_detected", "error": float((correct_order - reversed_order).abs().max()), "passed": wrong_order_detected},
    ])
    write_csv(DEST / "output_action_runs.csv", rows)
    passed = all(bool(row["passed"]) for row in rows)
    factual_report(DEST / "output_action_report.md", "Exact spatial-output actions", [
        f"Executed checks: {len(rows)}.",
        f"Passed checks: {sum(bool(row['passed']) for row in rows)}/{len(rows)}.",
        "Vector coordinates and components were transformed by the same explicit D4 matrix.",
        "Composition and inverse laws were checked for every D4 product.",
    ])
    state = "completed" if passed else "failed"
    update_status("S2_exact_spatial_output_actions", state, f"{sum(bool(row['passed']) for row in rows)}/{len(rows)} checks passed")
    stage_complete(DEST / "output_action_runs.csv", {"stage": "S2", "state": state, "checks": len(rows)})
    return {"state": state, "checks": len(rows)}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    started_at, started = utc_now(), time.perf_counter()
    try:
        result = run()
    except Exception as error:
        update_status("S2_exact_spatial_output_actions", "failed", str(error))
        record_command(command=COMMAND, source=SCRIPT, seed_scope="exact", dataset_revision="procedural_spatial_outputs_v1", started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND, source=SCRIPT, seed_scope="exact", dataset_revision="procedural_spatial_outputs_v1", started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['checks']} output-action checks")


if __name__ == "__main__":
    main()
