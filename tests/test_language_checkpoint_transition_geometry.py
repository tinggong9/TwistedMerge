import numpy as np

from experiments.language_checkpoint_transition_geometry import diagnostics, maps, reduced


def test_language_subspace_diagnostics_execute():
    rng = np.random.default_rng(8); values = [rng.normal(size=(40, 16)) for _ in range(4)]
    projected, basis = reduced(values, width=8); fitted = maps(projected); residual, rank = diagnostics(fitted)
    assert basis.shape == (16, 8)
    assert residual >= 0 and rank >= 0
