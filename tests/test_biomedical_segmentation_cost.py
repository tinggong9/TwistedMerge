import torch

from experiments.biomedical_segmentation_cost import METHODS, _batched_inverse, _pareto
from experiments.spatial_output_common import apply_d4


def test_cost_audit_has_all_required_method_families():
    assert len(METHODS) == 9
    assert "inferred_full_retransport" in METHODS
    assert "direct_d4_equivariant_unet" in METHODS


def test_pareto_marks_undominated_rows():
    rows = [
        {"batch_size": 1, "method": "a", "dice": 0.8, "latency": 2.0},
        {"batch_size": 1, "method": "b", "dice": 0.7, "latency": 3.0},
    ]
    result = _pareto(rows, "latency")
    assert result[0]["frontier"]
    assert not result[1]["frontier"]


def test_boundary_quality_frontier_uses_boundary_metric():
    rows = [
        {"batch_size": 1, "method": "a", "boundary_dice": 0.7, "latency": 2.0},
        {"batch_size": 1, "method": "b", "boundary_dice": 0.8, "latency": 3.0},
    ]
    result = _pareto(rows, "latency", "boundary_dice")
    assert all(row["frontier"] for row in result)
    assert all(row["quality"] == "boundary_dice" for row in result)


def test_batched_inverse_recovers_per_example_chart_actions():
    images = torch.arange(8 * 3 * 7 * 7, dtype=torch.float32).reshape(8, 3, 7, 7)
    charts = torch.arange(8)
    transformed = apply_d4(images, charts)
    assert torch.equal(_batched_inverse(transformed, charts), images)
