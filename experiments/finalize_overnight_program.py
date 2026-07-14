#!/usr/bin/env python3
"""Write final multi-stage manifest, status, tests, and combined research report."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"
ARXIV_COMMIT = "d71d1a3651a4c4c23a3e6e80c834b729d6a8aa2e"
TIER_BC_COMMIT = "66f7eaf811a57d4e94e1dd699c1025dc2fa4ea03"


def entry(stage, name, commit, script, command, config, raw, summary, table, plot, status, eligible, wording):
    return {"stage": stage, "id": name, "actual_execution_commit": commit, "script": script, "exact_command": command, "config": config, "raw_csv": raw, "summary_csv": summary, "latex_table": table, "plot": plot, "evidence_status": status, "paper_eligibility": eligible, "safe_wording": wording}


def manifest_entries() -> list[dict]:
    python = sys.executable
    return [
        entry(0, "provenance_audit", ARXIV_COMMIT, "experiments/overnight_stage0_audit.py", f"{python} experiments/overnight_stage0_audit.py --run-tests", "reports/overnight_program/stage0_git_state.json", "reports/overnight_program/stage0_artifact_status.csv", None, None, None, "supported", True, "Target-injected nonabelian accuracy artifacts are quarantined; per-artifact provenance is required."),
        entry(1, "fresh_practical_selector", ARXIV_COMMIT, "experiments/final_practical_selector.py", f"{python} experiments/final_practical_selector.py --mode full", "reports/overnight_program/practical_selector_config.json", "reports/overnight_program/practical_selector_runs.csv", "reports/overnight_program/practical_selector_summary.csv", "reports/overnight_program/tables/practical_selector_main.tex", "reports/overnight_program/plots/practical_selector_accuracy.pdf", "executed_negative_result", True, "Fresh MNIST selector is slightly below ordinary greedy soup; no central or nonabelian lift activates."),
        entry(2, "hodge_lr_components", ARXIV_COMMIT, "experiments/hodge_lr_smoke.py", f"{python} experiments/hodge_lr_smoke.py", "reports/overnight_program/hodge_lr_smoke_config.json", None, None, None, None, "component_smoke", False, "Transition, Hodge, low-rank, routing, gating, and distillation components pass unit smokes; no natural-data gain claim."),
        entry(3, "context_two_loop", ARXIV_COMMIT, "experiments/context_dependent_two_loop_holonomy.py", f"{python} experiments/context_dependent_two_loop_holonomy.py --mode full", "reports/overnight_program/two_loop_context_config.json", "reports/overnight_program/two_loop_context_runs.csv", "reports/overnight_program/two_loop_context_summary.csv", "reports/overnight_program/tables/two_loop_context_main.tex", "reports/overnight_program/plots/two_loop_context_accuracy.pdf", "executed_controlled_positive", True, "Controlled context-dependent S3/D4 task supports noncommuting structure and chart-aware accuracy, not natural checkpoint holonomy."),
        entry(4, "central_period_index", ARXIV_COMMIT, "experiments/central_release_overnight.py", f"{python} experiments/central_release_overnight.py --mode full", "reports/overnight_program/central_reproduction_manifest.json", "reports/overnight_program/central_mu2_runs.csv", "reports/overnight_program/central_mu2_summary.csv", "reports/overnight_program/tables/central_mu2.tex", None, "supported_controlled_and_exact", True, "Controlled mu2 supplied-context evidence and exact finite-Heisenberg thresholds are supported."),
        entry(5, "arxiv_release", ARXIV_COMMIT, "experiments/build_overnight_release.py", f"{python} experiments/build_overnight_release.py --release arxiv", "reports/overnight_program/arxiv_release_manifest.json", None, None, None, None, "verified_clean_release", True, "The arXiv-eligible subset shares one clean execution commit and passes forbidden-evidence scanning."),
        entry(6, "quaternion_pose", TIER_BC_COMMIT, "experiments/quaternion_projective_pose_merge.py", f"{python} experiments/quaternion_projective_pose_merge.py --mode smoke", "reports/overnight_program/quaternion_pose_config.json", "reports/overnight_program/quaternion_pose_runs.csv", "reports/overnight_program/quaternion_pose_summary.csv", "reports/overnight_program/tables/quaternion_pose.tex", "reports/overnight_program/plots/quaternion_pose.pdf", "synthetic_smoke_negative", False, "Generated quaternion smoke detects lift signs but the two-sheet method does not beat strict sign-invariant synchronization."),
        entry(7, "natural_twist", TIER_BC_COMMIT, "experiments/natural_twist_discovery.py", f"{python} experiments/natural_twist_discovery.py --mode smoke", "reports/overnight_program/natural_twist_config.json", "reports/overnight_program/natural_twist_runs.csv", "reports/overnight_program/natural_twist_summary.csv", "reports/overnight_program/tables/natural_twist.tex", "reports/overnight_program/plots/natural_twist.pdf", "natural_promotion_unsupported", False, "MNIST cycle proxy does not add held-out predictive value beyond validation features; no natural twist is promoted."),
        entry(8, "pretrained_vision", TIER_BC_COMMIT, "experiments/full_pretrained_vision_merging.py", f"{python} experiments/full_pretrained_vision_merging.py --mode smoke", "reports/overnight_program/pretrained_vision_config.json", "reports/overnight_program/pretrained_vision_runs.csv", "reports/overnight_program/pretrained_vision_summary.csv", "reports/overnight_program/tables/pretrained_vision.tex", "reports/overnight_program/plots/pretrained_vision.pdf", "one_seed_frozen_backbone_smoke", False, "One-seed frozen ResNet-18 head smoke is feasibility evidence, not a full pretrained benchmark."),
        entry(9, "lora_holonomy", TIER_BC_COMMIT, "experiments/lora_holonomy_merging.py", f"{python} experiments/lora_holonomy_merging.py --mode smoke", "reports/overnight_program/lora_holonomy_config.json", "reports/overnight_program/lora_holonomy_runs.csv", "reports/overnight_program/lora_holonomy_summary.csv", "reports/overnight_program/tables/lora_holonomy.tex", None, "synthetic_adapter_smoke", False, "Four synthetic adapters show gauge invariance but no persistent cycle holonomy; open pretrained LoRA remains blocked."),
        entry(10, "federated_frame", TIER_BC_COMMIT, "experiments/federated_sensor_frame_merge.py", f"{python} experiments/federated_sensor_frame_merge.py --mode smoke", "reports/overnight_program/federated_frame_config.json", "reports/overnight_program/federated_frame_runs.csv", "reports/overnight_program/federated_frame_summary.csv", "reports/overnight_program/tables/federated_frame.tex", None, "real_mnist_controlled_frame_smoke", False, "Exact rotated MNIST frames are removable; synchronization helps and no persistent lift is certified."),
        entry(11, "transformer", TIER_BC_COMMIT, "experiments/shared_base_transformer_merging.py", f"{python} experiments/shared_base_transformer_merging.py --mode smoke", "reports/overnight_program/transformer_config.json", "reports/overnight_program/transformer_runs.csv", "reports/overnight_program/transformer_summary.csv", "reports/overnight_program/tables/transformer_merging.tex", None, "local_transformer_smoke", False, "Four local tiny-Transformer checkpoints are merged; no open pretrained-model claim is allowed."),
        entry(12, "capacity_latency", TIER_BC_COMMIT, "experiments/capacity_latency_robustness.py", f"{python} experiments/capacity_latency_robustness.py", "reports/overnight_program/capacity_latency_config.json", "reports/overnight_program/capacity_latency.csv", None, "reports/overnight_program/tables/capacity_latency.tex", "reports/overnight_program/plots/capacity_latency.pdf", "cross_benchmark_smoke_audit", False, "Measured available capacity/latency and controls are reported; missing FLOPs, memory, and sensitivity sweeps are not imputed."),
    ]


def main() -> None:
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    entries = manifest_entries()
    missing = []
    for item in entries:
        for key in ("config", "raw_csv", "summary_csv", "latex_table", "plot"):
            value = item.get(key)
            if value and not (ROOT / value).exists():
                missing.append(value)
    if missing:
        raise RuntimeError(f"final release inputs missing: {missing}")
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, text=True, capture_output=True)
    (OUT / "final_test_results.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("final full test suite failed")

    final_manifest = {
        "release_commit_containing_all_code": current_commit,
        "arxiv_eligible_common_execution_commit": ARXIV_COMMIT,
        "tier_bc_smoke_execution_commit": TIER_BC_COMMIT,
        "single_common_execution_commit_for_all_stages": False,
        "single_commit_note": "ArXiv-eligible Stages 0-5 share a clean commit; later non-paper Tier B/C smokes were added and rerun at a second clean commit.",
        "per_artifact_execution_commit_verified": True,
        "strongest_supported_claim_level": 1,
        "entries": entries,
    }
    (OUT / "final_release_manifest.json").write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
    lines = ["# Final per-artifact evidence manifest", "", f"Code HEAD: `{current_commit}`", "", "| stage | id | evidence status | paper eligible | execution commit |", "|---:|---|---|---|---|"]
    for item in entries:
        lines.append(f"| {item['stage']} | {item['id']} | {item['evidence_status']} | {item['paper_eligibility']} | `{item['actual_execution_commit']}` |")
    (OUT / "final_release_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    environment = {"code_head": current_commit, "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(), "torch": __import__("torch").__version__, "environment": {key: os.environ.get(key) for key in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")}, "missing_optional_packages": [name for name in ("transformers", "datasets", "peft") if __import__("importlib").util.find_spec(name) is None]}
    (OUT / "final_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    practical = pd.read_csv(OUT / "practical_selector_paired_stats.csv").query("method == 'twistedmerge_exact_gauge_soup_selector'").iloc[0]
    context = pd.read_csv(OUT / "two_loop_context_paired_stats.csv").query("method == 'supplied_context_oracle'").iloc[0]
    quaternion = pd.read_csv(OUT / "quaternion_pose_claims.csv").query("claim == 'two_sheet_lift_beats_best_strict'").iloc[0]
    natural = json.loads((OUT / "natural_twist_config.json").read_text())
    lora_max = pd.read_csv(OUT / "lora_holonomy_residuals.csv").cycle_residual_fro.max()
    federated = pd.read_csv(OUT / "federated_frame_runs.csv").set_index("method")
    pretrained = json.loads((OUT / "pretrained_vision_config.json").read_text())
    transformer = json.loads((OUT / "transformer_config.json").read_text())
    hodge = json.loads((OUT / "hodge_lr_smoke.json").read_text())
    capacity = json.loads((OUT / "capacity_latency_config.json").read_text())
    choices = pd.read_csv(OUT / "practical_selector_choices.csv").selected_method.value_counts().to_dict()
    tests_last = (OUT / "final_test_results.txt").read_text().strip().splitlines()[-1]

    report = f"""# Final overnight TwistedMerge research report

## 1. Executive verdict

- **arXiv:** ready only for the controlled-framework revision after deleting invalid target-injected empirical tables and inserting the clean Tier A evidence. The clean arXiv subset is verified at `{ARXIV_COMMIT}`.
- **ICLR:** not ready. Full pretrained vision, natural twist discovery, external-baseline integration, and matched systems controls are incomplete.
- **JMLR:** not ready. Broad multi-dataset, adapter, language, robustness, and systems evidence is missing.

## 2. Strongest supported claim

**Claim ladder Level 1.** TwistedMerge provides a descent-theoretic taxonomy, exact controlled central evidence, finite-Heisenberg thresholds, structural noncommuting certificates, and a controlled context-dependent accuracy construction. No natural twist or broad practical-superiority claim is supported.

## 3. Stage decisions

| Stage | Decision | Reason |
|---:|---|---|
| 0 | supported | provenance repaired; target-injected accuracy quarantined |
| 1 | supported negative | fresh selector is below ordinary greedy soup; no lift activation |
| 2 | supported with limitations | components/unit smoke pass; no natural accuracy evidence |
| 3 | supported controlled | S3/D4 structure and controlled chart-aware accuracy pass all gates |
| 4 | supported controlled/exact | mu2 reproduction and all seven period-index cases pass |
| 5 | supported | clean per-artifact arXiv manifest and tests pass |
| 6 | unsupported positive claim | synthetic quaternion lift ties best strict baseline |
| 7 | unsupported promotion | residual proxy does not improve held-out prediction |
| 8 | descriptive/blocked full | one-seed frozen ResNet smoke only |
| 9 | descriptive/blocked full | synthetic adapters; no persistent cycle; dependencies missing |
| 10 | supported with limitations | real MNIST clients, but only exact removable frames |
| 11 | descriptive/blocked full | local tiny Transformer, not open pretrained |
| 12 | descriptive/incomplete | available costs audited; systems matrix incomplete |

## 4. Commands and commits

Stages 0--5 were executed from clean commit `{ARXIV_COMMIT}`. Stages 6--11 were rerun from clean commit `{TIER_BC_COMMIT}`. Exact commands and per-artifact paths are in `final_release_manifest.json`. The final suite result is `{tests_last}`.

## 5. Main numerical results

- Practical selector versus ordinary greedy soup: {practical.mean_accuracy_delta:+.6f}, 95% CI [{practical.accuracy_delta_ci_low:+.6f}, {practical.accuracy_delta_ci_high:+.6f}], W/T/L {int(practical.wins)}/{int(practical.ties)}/{int(practical.losses)}.
- Controlled supplied-context S3/D4 lift versus strict synchronization: {context.mean_accuracy_delta:+.6f}, 95% CI [{context.ci_low:+.6f}, {context.ci_high:+.6f}].
- Quaternion lift superiority CI: [{quaternion.paired_ci_low:+.6f}, {quaternion.paired_ci_high:+.6f}] (gate failed).
- Natural discovery promotion: `{natural['natural_twist_promoted']}`.
- LoRA maximum cycle residual: {lora_max:.3e}.
- Federated raw/synchronized MNIST accuracy: {federated.loc['fedavg_raw_frame_weights', 'accuracy']:.4f}/{federated.loc['pairwise_synchronization', 'accuracy']:.4f}.
- Pretrained vision seeds completed: {pretrained['seeds_completed']}; selector source is recorded in `pretrained_vision_choices.csv`.
- Tiny Transformer checkpoints: {transformer['fine_tuned_checkpoints']}; selector source `{transformer['selector_source']}`.

## 6. Structural certificates

The context benchmark recovers noncommuting S3/D4 generators and verifies the regular action. Controlled mu2 preserves centrality/closure/coboundary records. Seven finite-Heisenberg cases verify scalar order, d^k thresholds, failed ranks, and direct sums. Quaternion cycle signs are generated-smoke structure. Natural MNIST, LoRA, federated exact frames, pretrained vision, and Transformer smokes do not certify a persistent natural obstruction.

## 7. Leakage tests

All accuracy stages save executed logits/predictions and verify that subsequent label/target permutation does not change the saved bytes. Every reported leakage flag is true. The old target-dependent S3/D4 artifacts remain invalid.

## 8. Capacity and latency

`capacity_latency.csv` contains 684 score rows over nine fixed lambda pairs. False-positive lift rate across audited negative gates is {capacity['false_positive_rate']:.4f}. FLOPs, most peak memory, batch-size sensitivity, and false-negative rate are explicitly missing, not imputed.

## 9. Candidate selection

Stage 1 selector counts are `{json.dumps(choices, sort_keys=True)}`. Central and nonabelian activation counts are both zero. The ResNet smoke selected `{pd.read_csv(OUT / 'pretrained_vision_choices.csv').selector_source_method.iloc[0]}`; the tiny Transformer selected `{transformer['selector_source']}`.

## 10. Negative results

The practical selector loses slightly to ordinary greedy soup; no natural lift activates; the quaternion two-sheet representation does not beat strict sign-invariant synchronization; natural cycle features worsen held-out degradation prediction; LoRA cycles close numerically; exact federated frames are removable; pretrained vision and language full programs are blocked.

## 11. Evidence for real twist-like residuals

There is strong controlled algebraic/structural evidence and a controlled context-dependent accuracy result. There is no confirmed persistent projective or holonomy residual in independently trained natural checkpoints.

## 12. Correction mechanisms

- Strict synchronization: useful in rotated-frame MNIST and practical MLP merging.
- Hodge correction/low-rank lift: implemented and unit-tested; no natural promotion.
- Invariant pooling: succeeds in the controlled S3/D4 task; quaternion lift gives no advantage over strict invariant methods.
- Adaptive routing: succeeds with context in the controlled task; open-domain generalization remains unsupported.
- Distillation: component KL falls from {hodge['distillation_initial_kl']:.6g} to {hodge['distillation_final_kl']:.6g}; no broad pretrained distillation claim.

## 13. Exact safe wording

- Controlled H2(mu2): an exact planted overlap construction with a supplied-context q=2 representation; not a learned natural router.
- Period-index: finite-Heisenberg relations certify threshold d^k and direct sums in the executed cases.
- Noncommuting holonomy: controlled S3/D4 generators and action laws are verified.
- Context accuracy: chart-aware prediction improves in the fixed controlled teacher task.
- Quaternion: generated smoke detects lift signs but no strict-baseline advantage.
- Practical selector: fresh inference is slightly below greedy soup and selects no lift.
- Learned router: supported only in controlled inference-context tasks.
- Natural discovery: no promotion.
- Pretrained vision: one-seed frozen-backbone feasibility only.
- LoRA: algebra smoke only; no open pretrained claim.
- Transformer: local architecture smoke only; no pretrained claim.
- Practical superiority: unsupported.

## 14. Paste-ready LaTeX tables

`practical_selector_main.tex`, `practical_selector_choices.tex`, `two_loop_context_main.tex`, `two_loop_context_structural.tex`, `two_loop_context_capacity.tex`, `central_mu2.tex`, `period_index.tex`, `quaternion_pose.tex`, `natural_twist.tex`, `pretrained_vision.tex`, `lora_holonomy.tex`, `federated_frame.tex`, `transformer_merging.tex`, and `capacity_latency.tex` under `reports/overnight_program/tables/`.

## 15. Abstract variants

### Controlled-framework version (supported)

We develop TwistedMerge, a descent-theoretic framework for diagnosing when model parameters can be merged after symmetry alignment and when residual transition structure requires a charted representation. We provide an executed controlled central H2(mu2) witness, exact finite-Heisenberg representation thresholds, structural noncommuting S3/D4 certificates, and a controlled context-dependent accuracy construction. A fresh MNIST selector rerun compares strict synchronization, monomial gauges, soups, and validation-only selection without activating unsupported lifts. Natural-checkpoint and broad practical-superiority claims remain open.

### Real-twist positive version (conditional; unsupported)

TwistedMerge detects stable projective or holonomy residuals in independently trained checkpoints and corrects only their low-rank persistent component, improving held-out accuracy over synchronization, soups, routing, and capacity controls. Do not use until the natural-data gates pass.

### Practical-superiority version (conditional; unsupported)

TwistedMerge adaptively selects ordinary merging or certified lifts and improves mean or worst-case performance across vision and language at matched cost. Do not use until full pretrained benchmarks pass.

## 16. Conclusion variants

### Controlled-framework conclusion (supported)

The evidence supports the controlled taxonomy, exact structural certificates, and a controlled chart-dependent accuracy advantage. It does not support natural twist discovery or broad practical superiority; the practical selector remains conservative and activates no unsupported lift.

### Real-twist conclusion (conditional)

If natural residuals pass null, stability, prediction, correction, accuracy, capacity, and budget gates, the framework would support obstruction-aware accuracy beyond controlled constructions. Present evidence does not.

### Practical-superiority conclusion (conditional)

If full vision, adapter, and language runs produce positive paired intervals at matched systems cost, TwistedMerge could be described as broadly regime-adaptive. Present evidence does not meet this bar.

## 17. Prioritized manuscript revision map

1. Delete/quarantine invalid target-injected accuracy tables and figures.
2. Replace aggregated selector numbers with the fresh negative result.
3. Insert clean central/period-index and controlled context-dependent tables.
4. Add explicit negative findings and Level-1 claim boundary.
5. Keep Tier B/C smokes out of headline paper numbers; cite them only as feasibility/blocker evidence.

The manuscript itself was not edited.
"""
    (OUT / "final_overnight_research_report.md").write_text(report, encoding="utf-8")
    shutil.copy2(OUT / "arxiv_paper_numbers.tex", OUT / "paper_numbers.tex")
    revision = (OUT / "arxiv_revision_map.md").read_text(encoding="utf-8") + "\n\n## Final overnight addendum\n\nTier B/C smokes do not raise the claim level. Use the controlled-framework abstract/conclusion only.\n"
    (OUT / "paper_revision_map.md").write_text(revision, encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    decisions = {item["stage"]: item["evidence_status"] for item in entries}
    status = {"schema_version": 1, "updated_at": now, "stages": {str(stage): {"name": item["id"], "status": "completed", "decision": item["evidence_status"], "execution_commit": item["actual_execution_commit"], "updated_at": now} for stage, item in ((item["stage"], item) for item in entries)}}
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    status_lines = ["# Overnight TwistedMerge Program Status", "", "| stage | name | status | decision | execution commit |", "|---:|---|---|---|---|"]
    for item in entries:
        status_lines.append(f"| {item['stage']} | {item['id']} | completed | {item['evidence_status']} | `{item['actual_execution_commit']}` |")
    (OUT / "status.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    print(json.dumps({"entries": len(entries), "strongest_claim_level": 1, "tests": tests_last}, indent=2))


if __name__ == "__main__":
    main()
