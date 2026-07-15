import torch

from experiments.trained_chart_inference import (
    D4EquivariantChartCNN,
    apply_d4,
    run_seed,
)


def test_d4_chart_model_is_trained_cnn_and_smoke_protocol_executes():
    model = D4EquivariantChartCNN(width=4)
    assert any(isinstance(module, torch.nn.Conv2d) for module in model.modules())
    image = torch.rand(3, 1, 28, 28)
    transformed = apply_d4(image, torch.tensor([0, 3, 7]))
    assert transformed.shape == image.shape
    runs, generalization, abstention, costs = run_seed(
        0, "test", epochs=1, task_epochs=1, train_size=512, test_size=64
    )
    assert len(runs) == 14
    assert len(generalization) == 14 * 8
    assert abstention
    assert len(costs) == 14
    assert all(row["label_permutation_hash_passed"] for row in runs)
    assert any(row["method"] == "trained_d4_equivariant_cnn_chart_classifier" and row["implementation"] == "trained_neural" for row in runs)
