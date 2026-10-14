import json
from pathlib import Path

from src.kbs_revision.evidence_inventory import inventory
from src.kbs_revision.finite_certificates import verify_all as algebra_checks, gate_ablation_controls
from src.kbs_revision.merge_baselines import verify_all as baseline_checks
from src.kbs_revision.policies import gate_controls
from src.kbs_revision.reporting import report


def test_reports_consume_actual_cpu_and_legacy_schemas(tmp_path):
    root = Path(__file__).resolve().parents[1]
    values = {"provenance.json": {"source_commit": "a" * 40, "code_hashes": {}},
              "inventory.json": inventory(root), "algebraic_checks.json": algebra_checks(),
              "baseline_parity.json": baseline_checks(include_timings=False),
              "cpu_shape_profile.json": {"rows": []},
              "pilot/readiness.json": {"blockers": ["no language backend"]},
              "verification_cost.json": {"wall_seconds": 1},
              "gate_controls.json": gate_controls()}
    for name, value in values.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    report(root, tmp_path)
    assert len((tmp_path / "per_episode.csv").read_text().splitlines()) == 1
    assert "17 & 18" in (tmp_path / "tables/controlled_algebra.tex").read_text()
    assert "stress_failure_retained" in (tmp_path / "failures.csv").read_text()
    assert "modern_policy_utility" in (tmp_path / "claim_ledger.json").read_text()
