"""Small offline protocol guards; fixtures are not trained research groups."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from src.kbs_revision.merge_baselines import LoRAUpdate, effective_delta
from src.kbs_small_offline import (
    CANDIDATES,
    COUNTS,
    DOMAINS,
    SealedDigits,
    choose,
    gauge_updates,
    make_splits,
    merge_factors,
    pairwise_admitted,
    simplex_summary,
    transform,
)


def _labels():
    return np.arange(1797, dtype=np.int64) % 10


def _updates():
    generator = torch.Generator().manual_seed(71014)
    return [
        [
            LoRAUpdate(
                torch.randn(64, 4, generator=generator, dtype=torch.float64) * .1,
                torch.randn(4, 64, generator=generator, dtype=torch.float64) * .1,
            )
            for _ in range(4)
        ]
        for _ in range(2)
    ]


def test_seven_roles_are_deterministic_stratified_and_id_disjoint():
    expected = dict(train=950, pilot=100, D_K=80, D_align=120,
                    D_cert=120, D_select=120, D_test=307)
    assert COUNTS == expected
    labels = _labels()
    splits = make_splits(labels)
    repeated = make_splits(labels)
    assert set(splits) == set(expected)
    ids = np.concatenate([splits[role] for role in expected])
    np.testing.assert_array_equal(np.sort(ids), np.arange(len(labels)))
    assert len(np.unique(ids)) == sum(expected.values()) == 1797
    proportions = np.bincount(labels, minlength=10) / len(labels)
    for role, count in expected.items():
        np.testing.assert_array_equal(splits[role], repeated[role])
        assert len(splits[role]) == count
        observed = np.bincount(labels[splits[role]], minlength=10)
        assert np.max(np.abs(observed - proportions * count)) <= 2
    changed = make_splits(labels, seed=20261015)
    assert any(not np.array_equal(splits[role], changed[role]) for role in expected)


def test_domains_are_fixed_bounded_and_original_id_deterministic():
    assert tuple(DOMAINS) == ("clean", "translate", "noise", "occlusion")
    images = np.random.default_rng(7).uniform(size=(5, 64)).astype(np.float32)
    ids = np.array([3, 20, 172, 180, 301])
    order = np.array([4, 1, 3])
    for domain in DOMAINS:
        result = transform(images, ids, domain)
        assert result.shape == images.shape
        assert np.isfinite(result).all()
        assert result.min() >= 0 and result.max() <= 1
        np.testing.assert_array_equal(result, transform(images, ids, domain))
        np.testing.assert_array_equal(
            result[order], transform(images[order], ids[order], domain)
        )
    np.testing.assert_array_equal(transform(images, ids, "clean"), images)
    assert any(not np.array_equal(transform(images, ids, d), images)
               for d in DOMAINS if d != "clean")


@pytest.mark.parametrize("kind", ["orthogonal", "diagonal"])
def test_exact_gauges_preserve_both_layer_updates_and_executed_predictions(kind):
    original = _updates()
    changed = gauge_updates(original, kind=kind, seed=314)
    assert len(changed) == 2
    for left, right in zip(original, changed):
        assert len(left) == len(right) == 4
        for before, after in zip(left, right):
            torch.testing.assert_close(effective_delta(before), effective_delta(after),
                                       rtol=1e-10, atol=1e-10)
    generator = torch.Generator().manual_seed(789)
    x = torch.randn(9, 64, generator=generator, dtype=torch.float64)
    base = [torch.randn(64, 64, generator=generator, dtype=torch.float64) * .1
            for _ in range(3)]
    head = torch.randn(10, 64, generator=generator, dtype=torch.float64)

    def logits(factors, specialist):
        hidden = torch.relu(x @ base[0].T)
        for layer in range(2):
            hidden = torch.relu(hidden @ (base[layer + 1] +
                                           effective_delta(factors[layer][specialist])).T)
        return hidden @ head.T

    for specialist in range(4):
        before, after = logits(original, specialist), logits(changed, specialist)
        torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-10)
        assert torch.equal(before.argmax(1), after.argmax(1))


def test_full_comparison_simplex_has_zero_constant_coefficient_h2():
    summary = simplex_summary()
    assert summary["h2"] == 0
    assert summary["d1_rank"] == 3
    assert summary["d2_rank"] == 1


def test_compact_and_dense_agree_at_matched_rank_in_both_layers():
    updates = _updates()
    compact = merge_factors(updates, "compact")
    dense = merge_factors(updates, "dense")
    assert len(compact) == len(dense) == 2
    for left, right in zip(compact, dense):
        assert left.rank == right.rank == 4
        torch.testing.assert_close(effective_delta(left), effective_delta(right),
                                   rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("value,expected", [
    (0.0, True), (.35, True), (float(np.nextafter(.35, np.inf)), False),
    (float("nan"), False), (float("inf"), False), (float("-inf"), False),
])
def test_pairwise_gate_applies_frozen_boundary_and_rejects_nonfinite(value, expected):
    assert bool(pairwise_admitted(value)) is expected


def test_test_role_requires_actual_frozen_receipt_and_correct_hash(tmp_path):
    labels = _labels()
    images = np.zeros((len(labels), 64), dtype=np.float32)
    data = SealedDigits(images, labels, make_splits(labels))
    assert len(data.get("D_select", "clean")[1]) == COUNTS["D_select"]
    with pytest.raises(PermissionError):
        data.get("D_test", "clean")
    path = tmp_path / "protocol_frozen.json"
    path.write_text(json.dumps({"phase": "frozen", "test_access": True}) + "\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((PermissionError, ValueError)):
        data.unseal(path, "0" * 64)
    with pytest.raises(PermissionError):
        data.get("D_test", "clean")
    data.unseal(path, expected)
    assert len(data.get("D_test", "clean")[1]) == COUNTS["D_test"]


@pytest.mark.parametrize("phase,access", [("pilot", True), ("frozen", False)])
def test_correct_hash_does_not_authorize_unfrozen_or_closed_test_access(tmp_path, phase, access):
    data = SealedDigits(np.zeros((1797, 64), dtype=np.float32), _labels(),
                        make_splits(_labels()))
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"phase": phase, "test_access": access}))
    with pytest.raises((PermissionError, ValueError)):
        data.unseal(path, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(PermissionError):
        data.get("D_test", "clean")


def test_candidate_universe_and_validation_ties_are_fixed_and_comparable():
    expected = ("compact", "naive", "pairwise", "global")
    assert tuple(CANDIDATES) == expected
    losses = {"compact": .3, "naive": .3, "pairwise": .2, "global": .2}
    assert choose(losses, list(CANDIDATES)) == "global"
    assert choose(losses, ["compact", "pairwise"]) == "pairwise"
    assert choose(losses, ["compact"]) == "compact"
    factors = _updates()
    for method in CANDIDATES:
        merged = merge_factors(factors, method)
        assert len(merged) == 2
        assert all(update.B.shape == (64, 4) and update.A.shape == (4, 64)
                   for update in merged)
