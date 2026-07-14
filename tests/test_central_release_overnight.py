from __future__ import annotations

import numpy as np

from src.central_reproduction import central_candidate_predictors, executed_central_candidate_logits
from src.controlled_twisted_overlaps import build_controlled_case


def test_central_predictors_ignore_test_labels() -> None:
    case = build_controlled_case("mu2_nontrivial_h2", 8, 4, 0, 40, 60, 2)
    predictors = central_candidate_predictors(case)
    first = {name: execute() for name, execute in predictors.items()}
    for face, (features, labels) in case.test_face_data.items():
        labels[:] = np.random.default_rng(7).permutation(labels)
        case.test_face_data[face] = (features, labels)
    second = executed_central_candidate_logits(case)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[name], second[name]) for name in first)
