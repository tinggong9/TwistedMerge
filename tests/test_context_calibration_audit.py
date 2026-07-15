import numpy as np

from experiments.context_calibration_audit import metrics, temperature, vector_scale


def test_temperature_is_validation_fitted_positive_scalar():
    logits = np.array([[8.0, -2.0], [-3.0, 7.0], [2.0, 1.0], [1.0, 2.0]])
    labels = np.array([0, 1, 1, 0])
    value = temperature(logits, labels)
    assert 0.2 <= value <= 5.0


def test_vector_scale_shapes_and_metrics():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.2], [1.3, 1.0]])
    labels = np.array([0, 1, 1, 0])
    scales, bias = vector_scale(logits, labels)
    assert scales.shape == bias.shape == (2,)
    result = metrics(logits / scales + bias, labels)
    assert set(result) == {"accuracy", "nll", "brier", "ece", "classwise_ece"}
