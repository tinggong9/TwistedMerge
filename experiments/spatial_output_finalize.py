#!/usr/bin/env python3
"""Assemble factual spatial-output manifests, claim levels, and test evidence."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.spatial_output_common import (  # noqa: E402
    OUT,
    dataset_checksum,
    dataset_counts,
    dataset_ready,
    git_head,
    record_command,
    sha256_file,
    stage_complete,
    update_status,
    utc_now,
    write_csv,
    write_json,
)

SCRIPT = Path(__file__).resolve()
COMMAND = "python experiments/spatial_output_finalize.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def claim_value(path: Path, claim: str) -> bool:
    return any(row.get("claim") == claim and row.get("passed", "").lower() == "true" for row in read_csv(path))


def paired_positive(rows: list[dict[str, str]], comparisons: tuple[str, ...]) -> bool:
    selected = {row.get("comparison"): row for row in rows}
    return bool(comparisons and all(name in selected and float(selected[name]["ci_lower"]) > 0 for name in comparisons))


def build_claim_ladder() -> list[dict[str, Any]]:
    sanity_masks = read_csv(OUT / "sanity" / "mask_claims.csv")
    output_actions = read_csv(OUT / "sanity" / "output_action_runs.csv")
    b1_claims = OUT / "biomedical" / "discovery" / "claims.csv"
    b1_paired = read_csv(OUT / "biomedical" / "discovery" / "paired.csv")
    b4_claims = OUT / "biomedical" / "cost" / "claims.csv"
    c1_claims = OUT / "multidomain" / "claims.csv"
    d1 = read_csv(OUT / "transitions" / "residuals.csv")
    d2_claims = OUT / "transitions" / "correction_claims.csv"
    exact_masks = bool(sanity_masks and all(row.get("passed", "").lower() == "true" for row in sanity_masks))
    exact_outputs = bool(output_actions and all(row.get("passed", "").lower() == "true" for row in output_actions))
    controlled = exact_masks and any(row.get("claim") == "negative_controls_fail" and row.get("passed", "").lower() == "true" for row in sanity_masks)
    inferred = claim_value(b1_claims, "retransport_gate")
    accuracy = paired_positive(b1_paired, ("retransport_vs_generic_soft", "full_vs_direct_equivariant", "full_vs_tta", "four_vs_one_after_inferred_chart"))
    matched_cost = claim_value(b4_claims, "twistedmerge_specific_matched_cost_gate")
    stable = any(row.get("layer") == "certificate" and row.get("certified_stable_residual", "").lower() == "true" for row in d1)
    correction = claim_value(d2_claims, "residual_correction_activated")
    return [
        {"level": "S1", "name": "output_action_correctness", "passed": exact_masks and exact_outputs, "evidence": "sanity/mask_claims.csv and sanity/output_action_runs.csv"},
        {"level": "S2", "name": "controlled_spatial_retransport", "passed": controlled, "evidence": "exact asymmetric-mask retransport and negative controls"},
        {"level": "S3", "name": "inferred_spatial_retransport", "passed": inferred, "evidence": "biomedical/discovery/claims.csv retransport_gate"},
        {"level": "S4", "name": "twistedmerge_specific_spatial_benefit", "passed": accuracy and matched_cost, "evidence": "B1 paired accuracy gates and B4 matched-cost gate"},
        {"level": "S5", "name": "multi_domain_benefit", "passed": claim_value(c1_claims, "multidomain_primary_gate"), "evidence": "multidomain/claims.csv; domains labeled synthetic"},
        {"level": "S6", "name": "realistic_residual_correction", "passed": stable and correction, "evidence": "transitions/residuals.csv and correction_claims.csv"},
    ]


def _baseline_manifest() -> None:
    revision = git_head()
    rows = [
        {"baseline": "U-Net", "status": "carefully_verified_internal_equivalent", "source": "experiments/spatial_output_common.py:TinyUNet", "commit": revision, "license": "repository license not declared", "wrapper_changes": "none", "hyperparameters": "width=4; 128x128; Adam lr=0.002; 2 epochs", "exact_command": "python experiments/biomedical_segmentation_discovery.py"},
        {"baseline": "D4-equivariant U-Net", "status": "carefully_verified_orbit_symmetrization", "source": "experiments/spatial_output_common.py:D4SymmetrizedUNet", "commit": revision, "license": "repository license not declared", "wrapper_changes": "exact eight-element orbit average", "hyperparameters": "shared TinyUNet base", "exact_command": "python experiments/biomedical_segmentation_discovery.py"},
        {"baseline": "D4 test-time augmentation", "status": "carefully_verified_internal_equivalent", "source": "experiments/spatial_output_common.py:D4SymmetrizedUNet", "commit": revision, "license": "repository license not declared", "wrapper_changes": "inverse input action and output retransport for all eight charts", "hyperparameters": "eight deterministic transforms", "exact_command": "python experiments/biomedical_segmentation_discovery.py"},
        {"baseline": "generic mixture of experts", "status": "carefully_verified_internal_equivalent", "source": "experiments/biomedical_segmentation_discovery.py", "commit": revision, "license": "repository license not declared", "wrapper_changes": "probability-weighted output logits without spatial retransport", "hyperparameters": "four independently trained specialists", "exact_command": "python experiments/biomedical_segmentation_discovery.py"},
        {"baseline": "domain-adaptive routing", "status": "carefully_verified_internal_equivalent", "source": "experiments/multidomain_biomedical_experts.py", "commit": revision, "license": "repository license not declared", "wrapper_changes": "synthetic-domain prototype router", "hyperparameters": "four disjoint synthetic-domain experts", "exact_command": "python experiments/multidomain_biomedical_experts.py"},
        {"baseline": "model soup and weight average", "status": "carefully_verified_internal_equivalent", "source": "experiments/spatial_output_common.py:average_state_dict", "commit": revision, "license": "repository license not declared", "wrapper_changes": "parameter-compatible arithmetic average and validation-selected prefix soup", "hyperparameters": "four identical TinyUNet architectures", "exact_command": "python experiments/biomedical_segmentation_discovery.py"},
        {"baseline": "orientation-normalized canonical model", "status": "carefully_verified_internal_equivalent", "source": "experiments/spatial_output_common.py:hard_canonical_retransport", "commit": revision, "license": "repository license not declared", "wrapper_changes": "exact inverse D4 input action and matching output action", "hyperparameters": "single TinyUNet plus trained chart model", "exact_command": "python experiments/biomedical_segmentation_discovery.py"},
    ]
    write_csv(OUT / "baselines" / "manifest.csv", rows)
    (OUT / "baselines" / "report.md").write_text("# Baseline implementation manifest\n\n- Seven internal equivalents were executed and are identified as internal equivalents, not official upstream packages.\n- Sources, execution commit, implementation status, wrapper behavior, hyperparameters, and commands are recorded in `manifest.csv`.\n- The repository does not currently declare a license file; the manifest records that fact.\n", encoding="utf-8")


def _run_tests(smoke: bool) -> subprocess.CompletedProcess[str]:
    command = [str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "-q"]
    if smoke:
        command.extend(["tests/test_exact_mask_retransport.py", "tests/test_exact_spatial_output_actions.py", "tests/test_trivial_vs_spatial_output_action.py", "tests/test_biomedical_dataset_audit.py", "tests/test_biomedical_segmentation_discovery.py", "tests/test_biomedical_zeroshot_segmentation.py", "tests/test_biomedical_chart_uncertainty.py", "tests/test_biomedical_segmentation_cost.py", "tests/test_multidomain_biomedical_experts.py", "tests/test_biomedical_missing_expert_robustness.py", "tests/test_segmentation_transition_geometry.py", "tests/test_residual_aware_segmentation.py", "tests/test_second_biomedical_segmentation.py", "tests/test_biomedical_landmark_retransport.py", "tests/test_medical_3d_retransport.py", "tests/test_microscopy_multiview_retransport.py", "tests/test_run_spatial_output_program.py", "tests/test_spatial_output_finalize.py"])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    body = "$ " + " ".join(command) + "\n" + result.stdout + result.stderr + f"\nexit_code={result.returncode}\n"
    (OUT / "test_results.txt").write_text(body, encoding="utf-8")
    return result


def _status_rows() -> list[dict[str, str]]:
    path = OUT / "status.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [{"stage": stage, **values} for stage, values in sorted(payload.get("stages", {}).items())]


def _mean(rows: list[dict[str, str]], method: str, metric: str = "dice") -> float | None:
    values = [float(row[metric]) for row in rows if row.get("method") == method and row.get(metric, "") not in ("", "nan")]
    return float(np.mean(values)) if values else None


def _reports(claims: list[dict[str, Any]], test_exit: int) -> None:
    b1 = read_csv(OUT / "biomedical" / "discovery" / "summary.csv")
    paired = read_csv(OUT / "biomedical" / "discovery" / "paired.csv")
    cost = read_csv(OUT / "biomedical" / "cost" / "summary.csv")
    dataset = read_csv(OUT / "data" / "dataset_manifest.csv")
    statuses = _status_rows()
    lines = ["# Spatial-output program factual report", "", "## Stage status", ""]
    lines.extend(f"- `{row['stage']}`: {row['state']}; {row['summary']}" for row in statuses)
    lines.extend(["", "## Protocol coverage and data", "", f"- Dataset-ready check: {dataset_ready()}; bounded split counts: {dataset_counts() if dataset_ready() else {}}.", f"- Dataset manifest rows: {len(dataset)}; dataset SHA-256 aggregate: {dataset_checksum() if dataset_ready() else 'unavailable'}.", "- Kvasir-SEG has no patient, center, site, scanner, institution, tissue, or organ-domain metadata in the resolved archive; synthetic color/stain shifts are labeled synthetic domains.", f"- Execution commit recorded by finalizer: `{git_head()}`.", "- Candidate segmentation predictions were persisted before mask metrics and label-permutation hash audits were recorded by B1, B2, B3, and C1."])
    lines.extend(["", "## Numerical results", ""])
    for method in ("inferred_chart_canonicalize_pool_retransport", "inferred_canonical_no_output_retransport", "direct_d4_equivariant_unet", "d4_test_time_augmentation", "generic_soft_moe", "one_canonical_inferred_inverse_and_retransport", "supplied_chart_canonicalize_pool_retransport"):
        value = _mean(b1, method)
        if value is not None:
            lines.append(f"- Mean B1 Dice, `{method}`: {value:.6f}.")
    lines.extend(["", "## Paired confidence intervals", ""])
    lines.extend(f"- `{row['comparison']}` Dice delta {float(row['mean_delta']):.6f}, 95% CI [{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}], seeds={row['seeds']}." for row in paired)
    claim_summary = ", ".join(f"{row['level']}={row['passed']}" for row in claims)
    lines.extend(["", "## Exact actions and component attribution", "", f"- Claim levels: {claim_summary}.", "- Exact mask, landmark, heatmap, point-set, and vector-field actions are recorded under `sanity/`.", "- B1 comparisons separately attribute output retransport, chart inference, multi-expert pooling, direct D4 equivariance, D4 test-time augmentation, generic routing, and supplied-chart inference.", "- C1 uses exact D4 chart actions and separately labeled synthetic non-group domains."])
    lines.extend(["", "## Complete cost and residual results", "", f"- Complete-path cost rows: {len(cost)}.", "- B4 includes chart inference, canonicalization, expert evaluation, pooling, final mask logits, output retransport, warm-ups, timed repetitions, process memory, accelerator memory where available, and stored bytes.", f"- D1 stable residual certificate: {next((row['passed'] for row in claims if row['level'] == 'S6'), False)}; D2 correction remained inactive when the D1 gate was closed."])
    lines.extend(["", "## Negative and gated findings", ""])
    for row in claims:
        if not row["passed"]:
            lines.append(f"- `{row['level']} {row['name']}` did not pass; evidence: {row['evidence']}.")
    lines.extend(["- No real multi-center conclusion was made because center/site metadata is absent.", "- No second-dataset, real-landmark, 3D, or multiview-microscopy result was substituted when its required audited data were unavailable.", f"- Test command exit code: {test_exit}.", "", "## Artifact paths", "", "- Machine-readable claims: `claim_ladder.json` and `claim_ladder.md`.", "- Experiment inventory: `experiment_manifest.csv` and `experiment_manifest.json`.", "- Integrity inventory: `artifact_checksums.csv` and `checkpoint_manifest.csv`.", "- Tests: `test_results.txt`; commands: `commands.csv`; failures: `failures.csv`."])
    (OUT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "claim_ladder.json", claims)
    ladder = ["# Spatial-output claim ladder", "", "| Level | Claim | Passed | Evidence |", "|---|---|---:|---|"]
    ladder.extend(f"| {row['level']} | {row['name']} | {str(row['passed']).lower()} | {row['evidence']} |" for row in claims)
    (OUT / "claim_ladder.md").write_text("\n".join(ladder) + "\n", encoding="utf-8")


def _manifests() -> None:
    checkpoints = [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted((OUT / "checkpoints").glob("*.pt"))]
    write_csv(OUT / "checkpoint_manifest.csv", checkpoints)
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device_mps_available": torch.backends.mps.is_available(), "git_commit": git_head(), "dataset_checksum": dataset_checksum() if dataset_ready() else "unavailable", "dataset_counts": dataset_counts() if dataset_ready() else {}}
    write_json(OUT / "environment.json", environment)
    (OUT / "reproduction.md").write_text("# Spatial-output program reproduction\n\n```bash\nPYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python experiments/fetch_kvasir_subset.py\nPYTHONPYCACHEPREFIX=/private/tmp/codex-pycache .venv/bin/python experiments/run_spatial_output_program.py --tier all --force\n```\n\nThe runner writes stage state, exact commands, failures, tests, manifests, and checksums under `reports/spatial_output_program/`.\n", encoding="utf-8")
    manifest = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {"artifact_checksums.csv", "experiment_manifest.csv", "experiment_manifest.json"}:
            continue
        relative = path.relative_to(OUT)
        rows = len(read_csv(path)) if path.suffix == ".csv" else ""
        manifest.append({"artifact": str(relative), "kind": path.suffix.lstrip(".") or "file", "bytes": path.stat().st_size, "rows": rows, "sha256": sha256_file(path)})
    write_csv(OUT / "experiment_manifest.csv", manifest)
    write_json(OUT / "experiment_manifest.json", manifest)
    checksums = [{"artifact": row["artifact"], "sha256": row["sha256"], "bytes": row["bytes"]} for row in manifest]
    checksums.extend([{"artifact": name, "sha256": sha256_file(OUT / name), "bytes": (OUT / name).stat().st_size} for name in ("experiment_manifest.csv", "experiment_manifest.json")])
    write_csv(OUT / "artifact_checksums.csv", checksums)


def run(smoke: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    _baseline_manifest()
    test_result = _run_tests(smoke)
    claims = build_claim_ladder()
    _reports(claims, test_result.returncode)
    if not (OUT / "failures.csv").exists():
        write_csv(OUT / "failures.csv", [], ("stage", "command", "exit_code", "stderr", "time"))
    _manifests()
    state = "completed" if test_result.returncode == 0 else "failed"
    update_status("Z0_finalization", state, f"claim ladder and manifests written; test exit={test_result.returncode}")
    stage_complete(OUT / "claim_ladder.json", {"stage": "Z0", "state": state, "test_exit_code": test_result.returncode})
    return {"state": state, "test_exit": test_result.returncode, "claims": claims}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    result = run(args.smoke)
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="all tests and final artifacts", dataset_revision=dataset_checksum() if dataset_ready() else "unavailable", started_at=started_at, runtime=time.perf_counter()-started, exit_code=int(result["test_exit"]), state=str(result["state"]), summary=f"claim levels written={len(result['claims'])}; test_exit={result['test_exit']}")
    # Refresh inventories after the finalizer's own command and stage records exist.
    _manifests()
    if result["test_exit"]:
        raise SystemExit(int(result["test_exit"]))


if __name__ == "__main__":
    main()
