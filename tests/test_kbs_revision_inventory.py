"""Historical reanalysis regression checks; no network or checkpoint loading."""
import json
from pathlib import Path

import pytest

from src.kbs_revision import evidence_inventory as evidence


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def record():
    return evidence.inventory(ROOT)


def test_inventory_is_offline_and_strict_json(record, monkeypatch):
    monkeypatch.setattr(evidence, "_fetch", lambda *_: pytest.fail("Offline inventory made a network call"))
    result = evidence.inventory(ROOT)
    json.dumps(result, allow_nan=False)
    assert result["external"]["status"] == "not_checked"
    assert all(h["status"] == "legacy_reproduced" for h in result["historical"])
    assert result["checkpoints"]["checkpoint_deserialization_performed"] is False


def test_historical_recomputation_matches_source_tables(record):
    claims = {h["claim_id"]: h for h in record["historical"]}
    diagnostics = claims["legacy_natural_diagnostic_prediction"]
    assert diagnostics["n_checkpoint_collections"] == 120
    for value in diagnostics["recomputation"].values():
        assert value["absolute_recompute_error"] < 1e-12
    selector = claims["legacy_matched_selector_attribution"]
    assert selector["n_training_groups"] == 10
    assert selector["n_exact_settings"] == 40
    assert selector["recomputation"]["maximum_ci_recompute_error"] < 1e-12
    assert selector["recomputation"]["paired_mean_accuracy_delta"] < 0
    assert len(selector["per_group"]) == 10


def test_negative_evidence_and_oracle_boundary_are_retained(record):
    claims = {h["claim_id"]: h for h in record["historical"]}
    lora = claims["legacy_single_feature_layer_lora"]
    assert lora["n_training_groups"] == 5
    assert lora["n_primary_rows"] == 2100
    assert lora["recomputation"]["cycle_aware_decision_counts"] == {"fallback_full_delta_svd": 300}
    assert lora["recomputation"]["full_delta_svd_mean_accuracy_advantage_over_global"] > 0
    assert claims["legacy_no_natural_certificate"]["n_certified_natural_candidates"] == 0
    assert claims["legacy_tetrahedral_oracle_correction"]["mu2_edge_witness_count"] == 0
    assert claims["legacy_tetrahedral_oracle_correction"]["ground_truth_construction_type"] == "oracle_face_assignment"
    assert record["provenance_resolution"]["older_is_ancestor_of_later"] is True


def test_missing_source_is_a_blocker_not_a_reproduction(tmp_path, monkeypatch):
    # Check the claim-level error wrapper without requiring nonexistent repo
    # checkout manifests in the downstream checkpoint inventory.
    monkeypatch.setattr(evidence, "_checkpoint_inventory", lambda *_: {})
    result = evidence.inventory(tmp_path)
    assert all(h["status"] == "blocked" for h in result["historical"])
    assert all("failure_reason" in h for h in result["historical"])
    assert all(i["can_reuse"] is False for i in result["items"])
