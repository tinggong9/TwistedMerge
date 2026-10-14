from __future__ import annotations

import json

import pytest
import torch

from src.kbs_revision.merge_baselines import (
    LoRAUpdate,
    _fixture,
    _peft_core,
    aligned_factor_average,
    compact_qr_svd,
    concatenate_exact,
    dare_ties_effective,
    dense_reference_svd,
    effective_delta,
    naive_factor_average,
    profile_factor_only,
    ties_effective,
    verify_all,
    weighted_sum,
)


@pytest.mark.parametrize("out,inp,ranks", [(17, 23, (2, 3, 1)), (3, 11, (3, 4, 2)), (3, 4, (3, 4, 2))])
@pytest.mark.parametrize("complex_dtype", [False, True])
def test_weighted_scaled_sum_compact_and_dense_parity(out, inp, ranks, complex_dtype):
    updates = _fixture(out=out, inp=inp, ranks=ranks, complex_dtype=complex_dtype)
    weights = [0.4, -0.7, 0.0]
    expected = weighted_sum(updates, weights)
    exact = concatenate_exact(updates, weights)
    assert exact.rank == sum(ranks)
    assert exact.scale == 1
    torch.testing.assert_close(effective_delta(exact), expected, rtol=1e-10, atol=1e-10)
    full = compact_qr_svd(updates, min(out, inp), weights)
    torch.testing.assert_close(effective_delta(full), expected, rtol=1e-10, atol=1e-10)
    compressed = compact_qr_svd(updates, 2, weights)
    dense = dense_reference_svd(updates, 2, weights)
    torch.testing.assert_close(effective_delta(compressed), effective_delta(dense), rtol=1e-10, atol=1e-10)


def test_gauge_preservation_under_dense_nonorthogonal_coordinates():
    updates = _fixture(ranks=(3, 3, 3))
    generator = torch.Generator().manual_seed(812)
    changed = []
    for update in updates:
        u, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
        v, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
        gauge = u @ torch.diag(torch.tensor([1, 6, 30], dtype=torch.float64)) @ v.mH
        transformed = LoRAUpdate(update.B @ gauge, torch.linalg.solve(gauge, update.A), update.scale)
        torch.testing.assert_close(effective_delta(transformed), effective_delta(update), atol=1e-10, rtol=1e-10)
        changed.append(transformed)
    torch.testing.assert_close(effective_delta(compact_qr_svd(changed, 3)), effective_delta(compact_qr_svd(updates, 3)), atol=1e-10, rtol=1e-10)


def test_repeated_cutoff_compares_error_and_admissible_subspace():
    delta = torch.diag(torch.tensor([5, 3, 3, 1], dtype=torch.float64))
    generator = torch.Generator().manual_seed(877)
    gauge, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
    update = LoRAUpdate(gauge, gauge.mH @ delta)
    compressed = effective_delta(compact_qr_svd([update], 2))
    assert float(torch.linalg.vector_norm(compressed - delta)) == pytest.approx(10**0.5, abs=1e-10)
    torch.testing.assert_close(compressed[:, 0], delta[:, 0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(compressed[:, 3], torch.zeros(4, dtype=torch.float64), atol=1e-10, rtol=1e-10)
    u, _, _ = torch.linalg.svd(compressed, full_matrices=False)
    assert float(torch.linalg.vector_norm(u[3, :2])) < 1e-10


def test_rank_deficient_zero_rank_and_zero_weight_updates():
    b = torch.arange(14, dtype=torch.float64).reshape(7, 2)
    b[:, 1] = b[:, 0]
    update = LoRAUpdate(b, torch.arange(22, dtype=torch.float64).reshape(2, 11), 0.7)
    torch.testing.assert_close(effective_delta(compact_qr_svd([update], 3)), effective_delta(update), atol=1e-10, rtol=1e-10)
    for zero in (LoRAUpdate(torch.zeros_like(b), update.A), LoRAUpdate(b[:, :0], update.A[:0])):
        assert not effective_delta(compact_qr_svd([zero], 3)).count_nonzero()
    assert not effective_delta(compact_qr_svd([update], 3, [0])).count_nonzero()
    assert compact_qr_svd([update], 0).rank == 0


def test_scaling_naive_formula_and_singleton():
    updates = _fixture(ranks=(2, 2, 2))
    weights = [0.3, -0.7, 1.1]
    expected = sum(w * u.B for u, w in zip(updates, weights)) @ sum(w * u.scale * u.A for u, w in zip(updates, weights))
    torch.testing.assert_close(effective_delta(naive_factor_average(updates, weights)), expected)
    for function in (naive_factor_average, lambda u: compact_qr_svd(u, 2), aligned_factor_average):
        torch.testing.assert_close(effective_delta(function(updates[:1])), effective_delta(updates[0]), atol=1e-10, rtol=1e-10)
    assert float(torch.linalg.vector_norm(expected - weighted_sum(updates, weights))) > 1


def test_singleton_weights_documented_cross_term_semantics():
    update = _fixture(ranks=(2,))[0]
    torch.testing.assert_close(effective_delta(naive_factor_average([update], [0.3])), 0.09 * effective_delta(update))
    torch.testing.assert_close(effective_delta(compact_qr_svd([update], 2, [0.3])), 0.3 * effective_delta(update))


def test_permutation_keeps_weights_attached_to_update():
    updates = _fixture()
    weights = [0.2, -0.8, 1.3]
    first = effective_delta(compact_qr_svd(updates, 3, weights))
    second = effective_delta(compact_qr_svd(updates[::-1], 3, weights[::-1]))
    torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)


def test_profiler_and_actual_dispatch_shapes_show_no_ambient_output():
    evidence = profile_factor_only()
    assert evidence["passed"]
    assert evidence["ambient_shape_output_count"] == 0
    assert any(row["operator"] == "aten::_linalg_svd" for row in evidence["profiler_kernels"])
    assert any(row["cpu_memory_usage_bytes"] > 0 for row in evidence["profiler_kernels"])
    json.dumps(evidence, allow_nan=False)


def test_ties_is_on_effective_updates_with_manual_kernel_reference():
    fixture = [LoRAUpdate(torch.ones(1, 1, dtype=torch.float64), torch.tensor([[4, -3, 2, -1]], dtype=torch.float64), 2), LoRAUpdate(torch.ones(1, 1, dtype=torch.float64), torch.tensor([[1, 5, -2, -6]], dtype=torch.float64))]
    expected = torch.tensor([[8, -6, 0, -6]], dtype=torch.float64)
    torch.testing.assert_close(ties_effective(fixture, density=0.5), expected)
    torch.testing.assert_close(ties_effective(fixture, [0.4, 0.8], density=0.5), _peft_core().ties([effective_delta(u) for u in fixture], torch.tensor([0.4, 0.8], dtype=torch.float64), 0.5, "total"))


def test_dare_ties_determinism_preserves_caller_rng():
    updates = _fixture(ranks=(2, 2, 2))
    before = torch.random.get_rng_state()
    first = dare_ties_effective(updates, seed=19, density=0.7)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(first, dare_ties_effective(updates, seed=19, density=0.7))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        expected = _peft_core().dare_ties([effective_delta(u) for u in updates], torch.ones(3, dtype=torch.float64), 0.7, "total")
    torch.testing.assert_close(first, expected)


def test_invalid_baselines_fail_explicitly():
    updates = _fixture()
    with pytest.raises(ValueError, match="equal ranks"):
        naive_factor_average(updates)
    with pytest.raises(ValueError, match="finite real weight"):
        compact_qr_svd(updates, 2, [1, float("nan"), 2])
    with pytest.raises(ValueError, match="nonnegative integer"):
        compact_qr_svd(updates, -1)
    with pytest.raises(ValueError, match="density"):
        dare_ties_effective(updates, density=0)


def test_measured_cpu_report_retains_stress_failures_and_is_serializable():
    report = verify_all(include_timings=False)
    assert report["passed"], [row for row in report["checks"] if not row["passed"]]
    assert len(report["stress_panel"]) == 4
    assert report["stress_panel"][-1].get("passed") is False or report["stress_panel"][-1]["status"] == "failed"
    json.dumps(report, allow_nan=False)
