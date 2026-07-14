#!/usr/bin/env python3
"""Executed Stage 2 smoke and immediate design/evidence report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"
sys.path.insert(0, str(ROOT))

from src.twist_distillation import distill_linear_student  # noqa: E402
from src.twist_router import LinearTwistRouter  # noqa: E402
from src.twist_subspace import bootstrap_rank_stability, extract_twist_subspace, subspace_cost  # noqa: E402
from src.twistedmerge_hodge_lr import (  # noqa: E402
    conservative_confidence_gate,
    cycle_residual,
    dispatch_correction,
    estimate_transition,
    weighted_hodge_decomposition,
)


def main() -> None:
    execution_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty_worktree_at_execution = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    OUT.mkdir(parents=True, exist_ok=True)
    tests = [
        "tests/test_twistedmerge_hodge_lr.py",
        "tests/test_twist_subspace.py",
        "tests/test_twist_router.py",
        "tests/test_twist_distillation.py",
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, text=True, capture_output=True
    )
    (OUT / "hodge_lr_unit_tests.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise SystemExit(completed.returncode)

    rng = np.random.default_rng(2026)
    source = rng.normal(size=(600, 4))
    rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    target = source @ rotation.T
    transition = estimate_transition(source[:300], target[:300], family="orthogonal", heldout=(source[300:], target[300:]))
    cycle = cycle_residual(rotation, rotation)
    b1 = np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]], dtype=float)
    b2 = np.ones((3, 1))
    hodge = weighted_hodge_decomposition(b1, b2, np.array([0.8, -0.1, 0.4]), edge_weights=[1, 2, 3])
    residual_stack = np.stack([np.eye(4) + 0.2 * np.outer(rng.normal(size=4), np.array([1, 0, 0, 0])) for _ in range(30)])
    subspace = extract_twist_subspace(residual_stack, epsilon=0.05)
    stability = bootstrap_rank_stability(residual_stack, epsilon=0.05, samples=100, seed=2)

    features = rng.normal(size=(1000, 3))
    branch_target = (features[:, 0] > 0).astype(int)
    router = LinearTwistRouter(3, 2, seed=1).fit(features[:600], branch_target[:600], steps=600)
    router_accuracy = float(np.mean(router.predict_proba(features[600:]).argmax(1) == branch_target[600:]))
    teacher_logits = features @ rng.normal(size=(3, 3))
    _, distillation_history = distill_linear_student(features[:700], teacher_logits[:700], steps=600, learning_rate=0.2)
    gate = conservative_confidence_gate([0.01, -0.01, 0.00, 0.02])
    dispatch = dispatch_correction(residual_norm=1.0, harmonic_norm=0.4, gate=gate)

    smoke = {
        "transition_calibration_error": transition.calibration_error,
        "transition_heldout_error": transition.heldout_error,
        "transition_rank": transition.rank,
        "transition_condition_number": transition.condition_number,
        "cycle_distance_to_identity": cycle.distance_to_identity,
        "cycle_distance_to_real_center": cycle.distance_to_real_center,
        "cycle_effective_rank": cycle.effective_residual_rank,
        "hodge_reconstruction_error": hodge.reconstruction_error,
        "hodge_max_abs_orthogonality_error": max(
            abs(hodge.gradient_harmonic_inner), abs(hodge.gradient_coexact_inner), abs(hodge.harmonic_coexact_inner)
        ),
        "twist_rank": subspace.chosen_rank,
        "twist_explained_energy": subspace.explained_energy,
        "twist_rank_bootstrap": stability,
        "twist_cost": subspace_cost(10000, 4, subspace.chosen_rank, 2),
        "router_heldout_branch_accuracy": router_accuracy,
        "distillation_initial_kl": distillation_history[0],
        "distillation_final_kl": distillation_history[-1],
        "confidence_gate_activate": gate.activate,
        "dispatcher_mode": dispatch.mode,
        "dispatcher_activate_lift": dispatch.activate_lift,
    }
    (OUT / "hodge_lr_smoke.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    (OUT / "hodge_lr_smoke_config.json").write_text(
        json.dumps(
            {
                "stage": 2,
                "execution_commit": execution_commit,
                "dirty_worktree_at_execution": dirty_worktree_at_execution,
                "command": " ".join([sys.executable, *sys.argv]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    design = """# TwistedMerge-Hodge/LR design

The implementation estimates functional transition maps in permutation, positive-monomial, orthogonal, whitened-linear/CCA, low-rank, LoRA-basis, and block-orthogonal families. Diagnostics separate calibration fit, held-out fit, rank, conditioning, inverse consistency, cycle distance, centrality distance, and residual spectrum.

Weighted edge cochains are decomposed as a removable gradient component, a harmonic component, and the weighted-adjoint face/coexact component. The code verifies boundary-of-boundary, weighted orthogonality, and exact reconstruction. A harmonic numerical component is called persistent cycle structure, not an H^2 obstruction, unless external closure and coefficient assumptions are supplied.

Only the SVD-selected residual subspace is eligible for a lift. The dispatcher defaults to strict synchronization for removable structure and the validated ordinary family for uncertified structure. Central and noncentral lifts require structural certificates, representation-rank checks where applicable, and a positive lower confidence bound on validation gain. Routing uses inference-available features; distillation consumes teacher probabilities and not labels.
"""
    (OUT / "hodge_lr_design.md").write_text(design, encoding="utf-8")
    report = f"""# Stage 2 smoke report

All {len(tests)} focused test files passed. Orthogonal transition held-out relative error was {transition.heldout_error:.3e}; weighted Hodge reconstruction error was {hodge.reconstruction_error:.3e}, and maximum weighted orthogonality error was {smoke['hodge_max_abs_orthogonality_error']:.3e}. The extracted residual rank was q={subspace.chosen_rank} with {subspace.explained_energy:.6f} explained energy. The inference-feature router achieved {router_accuracy:.4f} held-out branch accuracy. Distillation KL fell from {distillation_history[0]:.6f} to {distillation_history[-1]:.6f}.

The deliberately inconclusive gain sample did not pass the conservative gate, so the dispatcher returned `{dispatch.mode}` and created no branches. This is the intended false-positive control. These are component smoke results, not a natural-data accuracy claim.
"""
    (OUT / "hodge_lr_smoke_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(smoke, indent=2))


if __name__ == "__main__":
    main()
