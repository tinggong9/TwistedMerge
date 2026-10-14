from __future__ import annotations

import json

import pytest
import torch

from src.kbs_revision import external_methods as external
from src.kbs_revision.merge_baselines import LoRAUpdate, _fixture, compact_qr_svd, effective_delta, weighted_sum


def test_exact_reviewed_external_tensor_references_pass_and_report_scope():
    report = external.verify_external_parity()
    assert report["passed"], report["checks"]
    assert len(report["checks"]) >= 12
    assert sum(check.get("official_tensor_kernel", False) for check in report["checks"]) == 8
    assert report["neural_evaluation"] == report["full_upstream_model_runner"] == "not_run"
    assert len(report["stress_panel"]) == 8
    assert any(row["passed"] is False for row in report["stress_panel"])
    definitions = report["sources"]["methods"]
    assert {definition["license"] for definition in definitions} == {"MIT", "Apache-2.0"}
    assert all(len(definition["commit"]) == 40 for definition in definitions)
    assert all("reviewed_kernels" not in definition for definition in definitions)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("density", [1., .5, .25])
def test_independent_official_nonlinear_core_parity_with_heterogeneous_ranks(density):
    updates = _fixture(ranks=(2, 3, 1))
    # Scale one is the original alpha/r=1 source setup; weighted extension is
    # tested separately, rather than counted as official pretrained-run parity.
    updates = [LoRAUpdate(u.B, u.A) for u in updates]
    actual = effective_delta(external.core_space_merge(updates, density=density))
    expected = external.official_core_reference(updates, density=density)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_knots_global_nonlinear_masking_differs_from_independent_layer_masking():
    layers = {"q_proj": _fixture(seed=43, out=13, inp=11, ranks=(2, 2, 2)),
              "o_proj": _fixture(seed=66, out=17, inp=19, ranks=(2, 2, 2))}
    actual = external.knots_merge_layers(layers, density=.25)
    reference = external.official_knots_reference(layers, density=.25)
    assert any(float(torch.linalg.vector_norm(effective_delta(actual[name]) -
                     effective_delta(external.knots_merge(updates, density=.25)))) > 1e-3
               for name, updates in layers.items())
    for name in layers:
        torch.testing.assert_close(effective_delta(actual[name]), reference[name], atol=2e-6, rtol=2e-6)


def test_core_space_ties_is_distinct_from_weighted_sum_compression():
    updates = _fixture(ranks=(2, 2, 2))
    actual = effective_delta(external.core_space_merge(updates, density=.5))
    compact = effective_delta(compact_qr_svd(updates, min(actual.shape), [1.] * len(updates)))
    assert float(torch.linalg.vector_norm(actual - compact)) > 1.


@pytest.mark.parametrize("function,tolerance", [(external.knots_merge, 2e-6), (external.core_space_merge, 1e-10)])
def test_explicitly_named_corrected_weighting_matches_scaled_effective_sum(function, tolerance):
    updates = _fixture(ranks=(2, 3, 1))
    weights = [.3, -.4, 0.]
    with pytest.raises(ValueError, match="corrected_per_task"):
        function(updates, weights=weights)
    actual = effective_delta(function(updates, weights=weights, merge_method="sum", weight_semantics="corrected_per_task"))
    torch.testing.assert_close(actual.double(), weighted_sum(updates, weights).double(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("function", [external.knots_merge, external.core_space_merge])
def test_official_common_multiplier_is_linear_after_elected_mask(function):
    updates = _fixture(ranks=(2, 2, 2))
    original = effective_delta(function(updates, density=.5))
    multiplied = effective_delta(function(updates, weights=[-.25] * 3, scaling=2., density=.5))
    torch.testing.assert_close(multiplied, -.5 * original, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("function", [external.knots_merge, external.core_space_merge])
@pytest.mark.parametrize("rank", [0, 1, 3])
def test_output_compression_caps_factor_capacity_and_matches_dense_svd(function, rank):
    updates = _fixture(ranks=(2, 2, 2))
    full = function(updates, density=.5)
    compressed = function(updates, rank=rank, density=.5)
    expected = effective_delta(external.dense_reference_svd([full], rank, [1.]))
    assert compressed.rank <= rank
    assert compressed.scale == 1.
    torch.testing.assert_close(effective_delta(compressed), expected, rtol=2e-6, atol=2e-6)


def test_knots_absolute_singular_cutoff_is_measured_information_loss():
    delta = torch.diag(torch.tensor([2., 1e-6], dtype=torch.float64))
    update = LoRAUpdate(torch.eye(2, dtype=torch.float64), delta)
    output = external.knots_merge([update], merge_method="sum")
    assert output.rank == 1
    assert float(torch.linalg.vector_norm(effective_delta(output).double() - delta)) == pytest.approx(1e-6)


def test_core_space_rectangular_saturated_and_zero_rank_factors():
    updates = _fixture(out=3, inp=4, ranks=(3, 4, 2))
    actual = external.core_space_merge(updates, merge_method="sum")
    torch.testing.assert_close(effective_delta(actual), weighted_sum(updates, [1.] * len(updates)), rtol=1e-10, atol=1e-10)
    zero = LoRAUpdate(updates[0].B[:, :0], updates[0].A[:0])
    assert not effective_delta(external.core_space_merge([zero])).count_nonzero()
    assert not effective_delta(external.knots_merge([zero])).count_nonzero()


def test_reference_manifest_mutation_cannot_execute_changed_excerpt(tmp_path, monkeypatch):
    altered = tmp_path / "altered.json"
    altered.write_text(external._MANIFEST.read_text().replace('"schema_version": 1', '"schema_version": 2'))
    monkeypatch.setattr(external, "_MANIFEST", altered)
    with pytest.raises(RuntimeError, match="manifest SHA256 mismatch"):
        external.external_method_definitions()


def test_external_methods_reject_bad_arguments_and_inconsistent_tasks():
    updates = _fixture(ranks=(2, 2, 2))
    for function in (external.knots_merge, external.core_space_merge):
        with pytest.raises(ValueError, match="density"):
            function(updates, density=0)
        with pytest.raises(ValueError, match="rank"):
            function(updates, rank=True)
        with pytest.raises(ValueError, match="real float32"):
            function(_fixture(complex_dtype=True))
        with pytest.raises(ValueError, match="weight_semantics"):
            function(updates, weight_semantics="unknown")
    with pytest.raises(ValueError, match="same task"):
        external.knots_merge_layers({"q_proj": updates, "o_proj": updates[:2]})
