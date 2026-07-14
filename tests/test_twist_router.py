from __future__ import annotations

import numpy as np

from src.twist_router import LinearTwistRouter, invariant_projection, representation_loss


def test_router_learns_context_without_prediction_labels() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(800, 2))
    branches = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(int)
    router = LinearTwistRouter(2, 2, seed=3).fit(features[:500], branches[:500], steps=700)
    accuracy = np.mean(router.predict_proba(features[500:]).argmax(axis=1) == branches[500:])
    assert accuracy > 0.95


def test_exact_group_representation_and_invariant_projection() -> None:
    identity = np.eye(2)
    flip = np.array([[0.0, 1.0], [1.0, 0.0]])
    representation = {0: identity, 1: flip}
    projection = invariant_projection([flip])
    assert representation_loss(representation, [(1, 1, 0)], projection, [1]) < 1e-12
    assert np.allclose(projection @ flip, projection)
