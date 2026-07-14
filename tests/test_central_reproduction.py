import dataclasses

import numpy as np

from src.central_reproduction import executed_central_candidate_logits
from src.controlled_twisted_overlaps import build_controlled_case
from src.period_index_central import check_heisenberg_relations, heisenberg_generators, period_index_metadata


def test_central_candidate_logits_are_label_permutation_invariant():
    case = build_controlled_case("mu2_nontrivial_h2", 32, 4, 0, 64, 96, 2)
    saved = executed_central_candidate_logits(case)
    permuted = {
        face: (x, np.random.default_rng(91).permutation(y))
        for face, (x, y) in case.test_face_data.items()
    }
    rerun = executed_central_candidate_logits(dataclasses.replace(case, test_face_data=permuted))
    assert all(np.array_equal(saved[method], rerun[method]) for method in saved)


def test_period_index_cases_have_exact_d_power_k_thresholds():
    for d, k in ((2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (4, 1), (4, 2)):
        metadata = period_index_metadata(d, k)
        assert metadata.index == d**k
        assert check_heisenberg_relations(heisenberg_generators(d, k)).all_relations_hold
