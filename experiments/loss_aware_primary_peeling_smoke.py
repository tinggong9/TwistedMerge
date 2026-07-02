#!/usr/bin/env python
"""Loss-aware no-lift primary residual peeling smoke test.

This experiment keeps the v2 quotient cochain solve as the algebraic baseline,
but searches over representative banks and validation-gates a small fixed set of
loss-aware corrected-map candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.primary_residual_peeling_smoke_v2 import (  # noqa: E402
    build_loaders_and_models,
    choose_settings,
    cycle_residual,
    evaluate_c2m3_from_pairwise,
    load_run_metrics,
    load_triangle_maps,
    md_table,
    parse_csv,
    prime_peeling_plan,
    reconstruct_pairwise_perms,
    safe_float,
    summarize_relation_set,
)
from src.loss_aware_primary_peeling import (  # noqa: E402
    LossAwareCorrectionResult,
    LossAwareObjectiveWeights,
    RepresentativeCandidate,
    assemble_loss_aware_corrections,
    build_representative_bank,
    correction_result_with_q_residual,
    cumulative_update_allowed_loss_aware,
    directed_edge_labels,
    no_lift_capacity_metadata,
    permutation_cycle_residual,
    validation_selection_decision,
)
from src.primary_holonomy import fit_primary_quotient  # noqa: E402
from src.primary_residual_peeling import (  # noqa: E402
    compose_perm,
    invert_perm,
    is_valid_permutation,
    quotient_defect_labels_from_pairwise,
    quotient_residual_from_labels,
    relations_from_pairwise,
    solve_best_edge_cochain_mod_p,
    triangle_defects_from_pairwise,
)

MAIN_COLUMNS = [
    "dataset",
    "run_id",
    "prime",
    "candidate_id",
    "candidate_role",
    "peel_mode",
    "quotient_residual_before",
    "quotient_residual_after",
    "quotient_residual_reduction",
    "permutation_cycle_residual_before",
    "permutation_cycle_residual_after",
    "permutation_cycle_residual_reduction",
    "representative_displacement_mean",
    "representative_displacement_max",
    "inverse_consistency_violation",
    "alignment_cost_proxy",
    "objective_value",
    "implemented_corrected_merge",
    "validation_accuracy",
    "test_accuracy",
    "baseline_validation_accuracy",
    "baseline_test_accuracy",
    "validation_delta_vs_baseline",
    "test_delta_vs_baseline",
    "wrong_prime_control_validation_accuracy",
    "shuffled_control_validation_accuracy",
    "random_control_validation_accuracy",
    "no_quotient_control_validation_accuracy",
    "capacity_multiplier",
    "inference_multiplier",
    "uses_test_for_selection",
    "selected_by_validation",
    "claim_status",
    "na_reason",
]

BANK_COLUMNS = [
    "dataset",
    "run_id",
    "prime",
    "label",
    "candidate_index",
    "source",
    "fit_label",
    "fit_label_verified",
    "disagreement_from_identity",
    "order",
    "is_valid_permutation",
    "used_in_selected_candidate",
]

MAP_COLUMNS = [
    "dataset",
    "run_id",
    "prime",
    "candidate_id",
    "candidate_role",
    "edge",
    "edge_label",
    "representative_source",
    "representative_displacement",
    "original_map",
    "correction_map",
    "corrected_map",
    "map_valid",
]


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def command_text(argv: list[str]) -> str:
    return " ".join([".venv/bin/python", "experiments/loss_aware_primary_peeling_smoke.py", *argv])


def permutation_json(perm: Iterable[int]) -> str:
    return json.dumps(np.asarray(perm, dtype=int).tolist(), separators=(",", ":"))


def exact_edges_from_solution(solution, n_models: int, p: int) -> dict[tuple[int, int], int]:
    return directed_edge_labels(solution.edge_labels, n_models, p)


def candidate_roles() -> list[tuple[str, LossAwareObjectiveWeights]]:
    return [
        ("minimal_displacement", LossAwareObjectiveWeights(quotient_residual=1.0, permutation_cycle_residual=0.0, representative_displacement=1.0, inverse_consistency=0.5)),
        ("best_quotient_residual", LossAwareObjectiveWeights(quotient_residual=10.0, permutation_cycle_residual=0.0, representative_displacement=0.05, inverse_consistency=0.1)),
        ("best_permutation_cycle", LossAwareObjectiveWeights(quotient_residual=1.0, permutation_cycle_residual=10.0, representative_displacement=0.05, inverse_consistency=0.1)),
        ("combined_objective", LossAwareObjectiveWeights(quotient_residual=4.0, permutation_cycle_residual=4.0, representative_displacement=0.5, inverse_consistency=0.5)),
    ]


def no_lift_meta() -> dict:
    return no_lift_capacity_metadata()


def metrics_or_nan(metrics: dict | None, key: str) -> float:
    if not metrics:
        return float("nan")
    return safe_float(metrics.get(key))


def evaluate_candidate(setting, bundle, result: LossAwareCorrectionResult) -> dict | None:
    if not result.implemented:
        return None
    try:
        return evaluate_c2m3_from_pairwise(setting, bundle, result.corrected)
    except Exception:
        return None


def shuffled_label_edges(edge_labels: dict[tuple[int, int], int], n_models: int, p: int, seed: int) -> dict[tuple[int, int], int]:
    rng = np.random.default_rng(seed)
    directed = [(i, j) for i in range(int(n_models)) for j in range(int(n_models)) if i != j]
    values = [int(edge_labels[edge]) % int(p) for edge in directed]
    rng.shuffle(values)
    out = {(idx, idx): 0 for idx in range(int(n_models))}
    for edge, value in zip(directed, values):
        out[edge] = int(value) % int(p)
    return out


def no_quotient_edges(n_models: int) -> dict[tuple[int, int], int]:
    return {(i, j): 0 for i in range(int(n_models)) for j in range(int(n_models))}


def random_same_displacement_result(pairwise, base: LossAwareCorrectionResult, n_models: int, seed: int) -> LossAwareCorrectionResult:
    width = len(next(iter(pairwise.values())))
    identity = np.arange(width, dtype=int)
    rng = np.random.default_rng(seed)
    corrections = {(idx, idx): identity.copy() for idx in range(int(n_models))}
    for i in range(int(n_models)):
        for j in range(int(n_models)):
            if i == j:
                continue
            moved = int(np.sum(base.corrections[(i, j)] != identity)) if (i, j) in base.corrections else 0
            perm = identity.copy()
            for _ in range(max(0, int(round(moved / 2)))):
                a, b = rng.choice(width, size=2, replace=False)
                perm[a], perm[b] = perm[b], perm[a]
            corrections[(i, j)] = perm
    corrected = {edge: compose_perm(invert_perm(corrections[edge]), pairwise[edge]) for edge in pairwise}
    before = permutation_cycle_residual(pairwise, n_models)
    after = permutation_cycle_residual(corrected, n_models)
    return LossAwareCorrectionResult(
        corrections=corrections,
        corrected=corrected,
        selected_candidates={},
        edge_labels=dict(base.edge_labels),
        quotient_residual_after=float("nan"),
        permutation_cycle_residual_before=before,
        permutation_cycle_residual_after=after,
        representative_displacement_mean=base.representative_displacement_mean,
        representative_displacement_max=base.representative_displacement_max,
        inverse_consistency_violation=float("nan"),
        alignment_cost_proxy=0.0,
        objective_value=float("nan"),
        implemented=True,
        status="random_same_displacement_control",
        candidate_role="random_same_displacement_control",
    )


def no_correction_result(pairwise, edge_labels: dict[tuple[int, int], int], n_models: int, quotient_residual: float) -> LossAwareCorrectionResult:
    width = len(next(iter(pairwise.values())))
    identity = np.arange(width, dtype=int)
    identity_candidate = {
        (i, j): RepresentativeCandidate(0, identity.copy(), "identity", 0, True, 0.0, 1, True)
        for i in range(int(n_models))
        for j in range(int(n_models))
    }
    corrections = {(i, j): identity.copy() for i in range(int(n_models)) for j in range(int(n_models))}
    before = permutation_cycle_residual(pairwise, n_models)
    return LossAwareCorrectionResult(
        corrections=corrections,
        corrected={edge: np.asarray(value, dtype=int).copy() for edge, value in pairwise.items()},
        selected_candidates=identity_candidate,
        edge_labels=dict(edge_labels),
        quotient_residual_after=float(quotient_residual),
        permutation_cycle_residual_before=before,
        permutation_cycle_residual_after=before,
        representative_displacement_mean=0.0,
        representative_displacement_max=0.0,
        inverse_consistency_violation=0.0,
        alignment_cost_proxy=0.0,
        objective_value=float(quotient_residual),
        implemented=True,
        status="no_correction_baseline",
        candidate_role="no_correction_baseline",
    )


def bank_rows(setting, prime: int, bank, selected_keys: set[tuple[int, tuple[int, ...]]]) -> list[dict]:
    rows = []
    for label, candidates in sorted(bank.items()):
        for idx, candidate in enumerate(candidates):
            key = (int(label), tuple(int(v) for v in candidate.perm))
            rows.append(
                {
                    "dataset": setting.dataset,
                    "run_id": setting.run_id,
                    "prime": int(prime),
                    "label": int(label),
                    "candidate_index": int(idx),
                    "source": candidate.source,
                    "fit_label": candidate.fit_label if candidate.fit_label is not None else np.nan,
                    "fit_label_verified": candidate.fit_label_verified if candidate.fit_label_verified is not None else np.nan,
                    "disagreement_from_identity": candidate.disagreement_from_identity,
                    "order": candidate.order,
                    "is_valid_permutation": candidate.is_valid_permutation,
                    "used_in_selected_candidate": key in selected_keys,
                }
            )
    return rows


def map_rows(setting, prime: int, candidate_id: str, role: str, pairwise, result: LossAwareCorrectionResult) -> list[dict]:
    rows = []
    for edge in sorted(pairwise):
        corr = result.corrections.get(edge)
        out = result.corrected.get(edge)
        candidate = result.selected_candidates.get(edge)
        rows.append(
            {
                "dataset": setting.dataset,
                "run_id": setting.run_id,
                "prime": int(prime),
                "candidate_id": candidate_id,
                "candidate_role": role,
                "edge": f"{edge[0]}->{edge[1]}",
                "edge_label": int(result.edge_labels.get(edge, 0)),
                "representative_source": candidate.source if candidate else "",
                "representative_displacement": candidate.disagreement_from_identity if candidate else np.nan,
                "original_map": permutation_json(pairwise[edge]),
                "correction_map": permutation_json(corr) if corr is not None else "[]",
                "corrected_map": permutation_json(out) if out is not None else "[]",
                "map_valid": bool(corr is not None and out is not None and is_valid_permutation(corr) and is_valid_permutation(out)),
            }
        )
    return rows


def candidate_row(setting, prime: int, candidate_id: str, role: str, result: LossAwareCorrectionResult, metrics, baseline, controls: dict[str, dict | None], q_before: float) -> dict:
    row = {
        "dataset": setting.dataset,
        "run_id": setting.run_id,
        "prime": int(prime),
        "candidate_id": candidate_id,
        "candidate_role": role,
        "peel_mode": "loss_aware_peel_p_only",
        "quotient_residual_before": float(q_before),
        "quotient_residual_after": result.quotient_residual_after,
        "quotient_residual_reduction": float(q_before - result.quotient_residual_after) if math.isfinite(result.quotient_residual_after) else float("nan"),
        "permutation_cycle_residual_before": result.permutation_cycle_residual_before,
        "permutation_cycle_residual_after": result.permutation_cycle_residual_after,
        "permutation_cycle_residual_reduction": result.permutation_cycle_residual_before - result.permutation_cycle_residual_after,
        "representative_displacement_mean": result.representative_displacement_mean,
        "representative_displacement_max": result.representative_displacement_max,
        "inverse_consistency_violation": result.inverse_consistency_violation,
        "alignment_cost_proxy": result.alignment_cost_proxy,
        "objective_value": result.objective_value,
        "implemented_corrected_merge": bool(metrics is not None),
        "validation_accuracy": metrics_or_nan(metrics, "validation_accuracy"),
        "test_accuracy": metrics_or_nan(metrics, "test_accuracy"),
        "baseline_validation_accuracy": metrics_or_nan(baseline, "validation_accuracy"),
        "baseline_test_accuracy": metrics_or_nan(baseline, "test_accuracy"),
        "wrong_prime_control_validation_accuracy": metrics_or_nan(controls.get("wrong_prime_control"), "validation_accuracy"),
        "shuffled_control_validation_accuracy": metrics_or_nan(controls.get("shuffled_control"), "validation_accuracy"),
        "random_control_validation_accuracy": metrics_or_nan(controls.get("random_control"), "validation_accuracy"),
        "no_quotient_control_validation_accuracy": metrics_or_nan(controls.get("no_quotient_control"), "validation_accuracy"),
        **no_lift_meta(),
        "uses_test_for_selection": False,
        "selected_by_validation": False,
        "claim_status": "",
        "na_reason": "",
    }
    row["validation_delta_vs_baseline"] = row["validation_accuracy"] - row["baseline_validation_accuracy"]
    row["test_delta_vs_baseline"] = row["test_accuracy"] - row["baseline_test_accuracy"]
    if not result.implemented:
        row["na_reason"] = result.status
    elif metrics is None:
        row["na_reason"] = "corrected_merge_metric_unavailable"
    elif not math.isfinite(row["quotient_residual_after"]) or row["quotient_residual_after"] >= row["quotient_residual_before"]:
        row["na_reason"] = "metric_produced_but_not_claimable"
    elif row["permutation_cycle_residual_after"] > row["permutation_cycle_residual_before"] + 1e-12:
        row["na_reason"] = "quotient_peel_not_permutation_safe"
    selected, status = validation_selection_decision(row)
    row["selected_by_validation"] = bool(selected)
    row["claim_status"] = status if selected else (row["na_reason"] or status)
    return row


def evaluate_controls(setting, bundle, pairwise, bank, edge_labels, prime: int, primes: list[int], source_order: int, combined: LossAwareCorrectionResult, max_beam_size: int) -> dict[str, dict | None]:
    controls: dict[str, dict | None] = {}
    wrong_prime = next((int(p) for p in primes if int(p) != int(prime) and int(source_order) % int(p) != 0), None)
    if wrong_prime is not None:
        wrong_labels = {edge: int(value) % wrong_prime for edge, value in edge_labels.items()}
        wrong_bank = build_representative_bank(None, pairwise, triangle_defects_from_pairwise(pairwise, setting.n_models).values(), setting.width, wrong_prime, 8)
        wrong = assemble_loss_aware_corrections(pairwise, wrong_labels, wrong_bank, setting.n_models, wrong_prime, LossAwareObjectiveWeights(), max_beam_size=max_beam_size, candidate_role="wrong_prime_loss_aware_control")
        controls["wrong_prime_control"] = evaluate_candidate(setting, bundle, wrong)
    else:
        controls["wrong_prime_control"] = None

    shuffled = assemble_loss_aware_corrections(
        pairwise,
        shuffled_label_edges(edge_labels, setting.n_models, prime, setting.seed + prime + 17),
        bank,
        setting.n_models,
        prime,
        LossAwareObjectiveWeights(),
        max_beam_size=max_beam_size,
        candidate_role="shuffled_label_loss_aware_control",
    )
    controls["shuffled_control"] = evaluate_candidate(setting, bundle, shuffled)
    random_result = random_same_displacement_result(pairwise, combined, setting.n_models, setting.seed + prime + 31)
    controls["random_control"] = evaluate_candidate(setting, bundle, random_result)
    no_quotient = assemble_loss_aware_corrections(
        pairwise,
        no_quotient_edges(setting.n_models),
        bank,
        setting.n_models,
        prime,
        LossAwareObjectiveWeights(quotient_residual=0.0, permutation_cycle_residual=1.0, representative_displacement=1.0, inverse_consistency=1.0),
        max_beam_size=max_beam_size,
        candidate_role="no_quotient_constraint_control",
    )
    controls["no_quotient_control"] = evaluate_candidate(setting, bundle, no_quotient)
    return controls


def evaluate_setting(setting, group, bundle, primes: list[int], max_candidates_per_label: int, max_beam_size: int):
    pairwise = reconstruct_pairwise_perms(group)
    baseline = evaluate_c2m3_from_pairwise(setting, bundle, pairwise)
    rows = []
    candidate_rows = []
    bank_out = []
    corrected_maps = []
    plan = prime_peeling_plan(setting.primary_source_order, primes)
    cumulative_pairwise = {edge: value.copy() for edge, value in pairwise.items()}
    current_validation = baseline["validation_accuracy"]

    for peel in plan:
        prime = int(peel["prime"])
        if not peel["eligible"]:
            continue
        relations_pairwise = pairwise
        relations = list(triangle_defects_from_pairwise(relations_pairwise, setting.n_models).values())
        fit = fit_primary_quotient(list(relations_from_pairwise(relations_pairwise, setting.n_models)), prime, random_restarts=8, seed=setting.seed + prime)
        defect_labels = quotient_defect_labels_from_pairwise(relations_pairwise, fit, setting.n_models, prime)
        q_before = quotient_residual_from_labels(defect_labels, prime)
        solution = solve_best_edge_cochain_mod_p(defect_labels, setting.n_models, prime)
        edge_labels = exact_edges_from_solution(solution, setting.n_models, prime)
        bank = build_representative_bank(fit, relations_pairwise, relations, setting.width, prime, max_candidates_per_label=max_candidates_per_label)
        role_results = []
        baseline_result = no_correction_result(relations_pairwise, edge_labels, setting.n_models, q_before)
        role_results.append((f"p{prime}_candidate_0", "no_correction_baseline", baseline_result, baseline))
        for idx, (role, weights) in enumerate(candidate_roles(), start=1):
            result = assemble_loss_aware_corrections(relations_pairwise, edge_labels, bank, setting.n_models, prime, weights, max_beam_size=max_beam_size, candidate_role=role)
            result = correction_result_with_q_residual(result, solution.quotient_residual_after)
            metrics = evaluate_candidate(setting, bundle, result)
            role_results.append((f"p{prime}_candidate_{idx}", role, result, metrics))
        combined = next(result for _cid, role, result, _metrics in role_results if role == "combined_objective")
        controls = evaluate_controls(setting, bundle, relations_pairwise, bank, edge_labels, prime, primes, setting.primary_source_order, combined, max_beam_size)
        selected_keys = set()
        prime_rows = []
        for candidate_id, role, result, metrics in role_results:
            row = candidate_row(setting, prime, candidate_id, role, result, metrics, baseline, controls, q_before)
            prime_rows.append(row)
            candidate_rows.append({**row, "edge_cochain_solve_status": solution.solve_status, "representative_selection_status": result.status})
            corrected_maps.extend(map_rows(setting, prime, candidate_id, role, relations_pairwise, result))
            if row["selected_by_validation"]:
                for candidate in result.selected_candidates.values():
                    selected_keys.add((int(candidate.label), tuple(int(v) for v in candidate.perm)))
        rows.extend(prime_rows)
        bank_out.extend(bank_rows(setting, prime, bank, selected_keys))
        best_selected = next((row for row in prime_rows if row["selected_by_validation"]), None)
        if best_selected is not None:
            selected_result = next(result for cid, _role, result, _metrics in role_results if cid == best_selected["candidate_id"])
            q_ok = best_selected["quotient_residual_after"] < best_selected["quotient_residual_before"]
            p_ok = best_selected["permutation_cycle_residual_after"] <= best_selected["permutation_cycle_residual_before"] + 1e-12
            v_ok = best_selected["validation_accuracy"] > current_validation
            if cumulative_update_allowed_loss_aware(q_ok, p_ok, v_ok):
                cumulative_pairwise = {edge: value.copy() for edge, value in selected_result.corrected.items()}
                current_validation = best_selected["validation_accuracy"]
    return rows, candidate_rows, bank_out, corrected_maps


def paired_stats(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    if rows.empty:
        return pd.DataFrame()
    for (dataset, role), group in rows.groupby(["dataset", "candidate_role"], sort=True):
        vals = pd.to_numeric(group["validation_delta_vs_baseline"], errors="coerce").dropna()
        tests = pd.to_numeric(group["test_delta_vs_baseline"], errors="coerce").dropna()
        out.append(
            {
                "dataset": dataset,
                "candidate_role": role,
                "n_rows": int(len(group)),
                "n_finite_validation_delta": int(len(vals)),
                "mean_validation_delta_vs_baseline": float(vals.mean()) if len(vals) else float("nan"),
                "best_validation_delta_vs_baseline": float(vals.max()) if len(vals) else float("nan"),
                "mean_test_delta_vs_baseline": float(tests.mean()) if len(tests) else float("nan"),
                "selected_by_validation_rows": int(group["selected_by_validation"].fillna(False).sum()),
                "claim_status": "real_positive" if group["selected_by_validation"].fillna(False).any() else "real_negative_or_control_blocked",
            }
        )
    return pd.DataFrame(out)


def run_status(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "implementation_invalid"
    if rows["selected_by_validation"].fillna(False).any():
        return "real_positive"
    implemented = rows[rows["implemented_corrected_merge"].fillna(False)]
    if implemented.empty:
        return "diagnostic_only"
    finite = pd.to_numeric(implemented["validation_accuracy"], errors="coerce").notna()
    if finite.any():
        return "real_negative"
    return "diagnostic_only"


def rigid_v2_best_delta(reports_dir: Path) -> float:
    path = reports_dir / "csv" / "primary_residual_peeling_smoke_v2.csv"
    if not path.exists():
        return float("nan")
    try:
        data = pd.read_csv(path)
    except Exception:
        return float("nan")
    if "method" in data.columns:
        data = data[data["method"].astype(str).str.contains("peeled", na=False)]
    if "validation_delta_vs_baseline" not in data.columns:
        return float("nan")
    vals = pd.to_numeric(data["validation_delta_vs_baseline"], errors="coerce").dropna()
    return float(vals.max()) if len(vals) else float("nan")


def exact_cochain_solve_count(candidates: pd.DataFrame) -> int:
    if candidates.empty or "edge_cochain_solve_status" not in candidates.columns:
        return 0
    exact = candidates[candidates["edge_cochain_solve_status"].astype(str).str.contains("exact", na=False)]
    if exact.empty:
        return 0
    cols = [col for col in ["dataset", "run_id", "prime"] if col in exact.columns]
    return int(exact[cols].drop_duplicates().shape[0]) if cols else int(len(exact))


def write_report(args, settings, rows: pd.DataFrame, candidates: pd.DataFrame, stats: pd.DataFrame, banks: pd.DataFrame) -> None:
    status = run_status(rows)
    selected_rows = rows[rows["selected_by_validation"].fillna(False)] if not rows.empty else pd.DataFrame()
    eligible = rows.groupby("dataset")["prime"].apply(lambda vals: ",".join(str(int(v)) for v in sorted(set(vals)))).to_dict() if not rows.empty else {}
    bank_sizes = banks.groupby(["dataset", "prime", "label"]).size().reset_index(name="n_candidates") if not banks.empty else pd.DataFrame()
    best_delta = pd.to_numeric(rows["validation_delta_vs_baseline"], errors="coerce").max() if not rows.empty else float("nan")
    v2_delta = rigid_v2_best_delta(args.reports_dir)
    delta_vs_v2 = float(best_delta - v2_delta) if math.isfinite(float(best_delta)) and math.isfinite(float(v2_delta)) else float("nan")
    exact_solves = exact_cochain_solve_count(candidates)
    control_passed = int(selected_rows["selected_by_validation"].sum()) if not selected_rows.empty else 0
    text = f"""# Loss-Aware Primary Peeling Smoke Test

Audit classification: `{status}`

Generated by `experiments/loss_aware_primary_peeling_smoke.py`.

## Exact Command

```bash
{command_text(sys.argv[1:])}
```

## Git State

- Git commit: `{git_output('rev-parse', '--short', 'HEAD')}`
- Dirty status (tracked files only): `{git_output('status', '--short', '--untracked-files=no') or 'clean'}`

## Scope

- Two-setting smoke test only.
- No-lift representative peeling; capacity and inference multipliers stay `1.0`.
- Selection is validation-only; test metrics are report-only.
- A positive claim requires quotient residual reduction, permutation safety, and beating baseline plus all controls by validation.

## Selected Settings

{md_table(pd.DataFrame([{**setting.__dict__} for setting in settings]), ['dataset', 'run_id', 'setting_id', 'n_models', 'width', 'matching', 'primary_source_order', 'model_source'])}

## Eligible Primes

`{json.dumps(eligible, sort_keys=True)}`

## Representative Bank Sizes

{md_table(bank_sizes, ['dataset', 'prime', 'label', 'n_candidates'], 80)}

## Candidate Corrections

{md_table(rows, ['dataset', 'prime', 'candidate_id', 'candidate_role', 'quotient_residual_before', 'quotient_residual_after', 'permutation_cycle_residual_before', 'permutation_cycle_residual_after', 'representative_displacement_mean', 'inverse_consistency_violation', 'objective_value', 'validation_accuracy', 'baseline_validation_accuracy', 'validation_delta_vs_baseline', 'selected_by_validation', 'claim_status', 'na_reason'], 120)}

## Paired Stats

{md_table(stats, ['dataset', 'candidate_role', 'n_rows', 'n_finite_validation_delta', 'mean_validation_delta_vs_baseline', 'best_validation_delta_vs_baseline', 'selected_by_validation_rows', 'claim_status'], 80)}

## Final Console Summary

Audit classification: `{status}`

Selected settings:
{chr(10).join(f'- {setting.dataset}: {setting.run_id}' for setting in settings)}

Eligible primes: `{json.dumps(eligible, sort_keys=True)}`

Representative bank sizes: `{len(banks)}` rows

Exact cochain solves: `{exact_solves}`

Candidate corrections evaluated: `{len(rows)}`

Validation-selected candidates: `{len(selected_rows)}`

Best validation delta vs baseline: `{best_delta if math.isfinite(float(best_delta)) else 'not_available'}`

Best validation delta vs rigid v2 peeling: `{delta_vs_v2 if math.isfinite(float(delta_vs_v2)) else 'not_available'}`

Controls passed: `{control_passed}`

Final interpretation:
- `{'Primary quotient/cochain structure produced a validation-selected loss-aware no-lift candidate.' if status == 'real_positive' else 'Primary quotient/cochain structure is useful diagnostically, but no-lift representative peeling is not yet a reliable accuracy-improving merge method on these real settings.'}`
"""
    (args.reports_dir / "loss_aware_primary_peeling_smoke_report.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="4,3")
    parser.add_argument("--settings-per-dataset", type=int, default=1)
    parser.add_argument("--prime-list", default="2,3,5,7,17,19,43")
    parser.add_argument("--prefer-n-models", type=int, default=4)
    parser.add_argument("--max-candidates-per-label", type=int, default=16)
    parser.add_argument("--max-beam-size", type=int, default=64)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-group-order", type=int, default=50000)
    parser.add_argument("--max-generators", type=int, default=6)
    parser.add_argument("--max-exact-order", type=int, default=50000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "csv").mkdir(parents=True, exist_ok=True)
    datasets = parse_csv(args.datasets, str)
    primes = parse_csv(args.prime_list, int)
    maps = load_triangle_maps(args.reports_dir, set(datasets), set(parse_csv(args.model_counts, int)))
    runs = load_run_metrics(args.reports_dir)
    chosen = choose_settings(maps, runs, datasets, args.prefer_n_models, args.settings_per_dataset)
    if len(chosen) < len(datasets):
        missing = sorted(set(datasets) - set(chosen))
        raise RuntimeError(f"missing selected settings: {missing}")

    settings = []
    all_rows = []
    all_candidates = []
    all_banks = []
    all_maps = []
    for dataset in datasets:
        group = chosen[dataset]
        setting = summarize_relation_set(group, args.reports_dir, args.max_group_order, args.max_generators, args.max_exact_order)
        settings.append(setting)
        bundle = build_loaders_and_models(setting, runs, args.reports_dir, args.data_dir, args.device)
        rows, candidates, banks, corrected = evaluate_setting(setting, group, bundle, primes, args.max_candidates_per_label, args.max_beam_size)
        all_rows.extend(rows)
        all_candidates.extend(candidates)
        all_banks.extend(banks)
        all_maps.extend(corrected)

    rows = pd.DataFrame(all_rows)
    candidates = pd.DataFrame(all_candidates)
    banks = pd.DataFrame(all_banks)
    corrected = pd.DataFrame(all_maps)
    for col in MAIN_COLUMNS:
        if col not in rows:
            rows[col] = np.nan
    rows = rows[MAIN_COLUMNS + [c for c in rows.columns if c not in MAIN_COLUMNS]].copy()
    for col in MAIN_COLUMNS:
        if col not in candidates:
            candidates[col] = np.nan
    candidates = candidates[MAIN_COLUMNS + ["edge_cochain_solve_status", "representative_selection_status"]].copy()
    for col in BANK_COLUMNS:
        if col not in banks:
            banks[col] = np.nan
    banks = banks[BANK_COLUMNS].copy()
    for col in MAP_COLUMNS:
        if col not in corrected:
            corrected[col] = np.nan
    corrected = corrected[MAP_COLUMNS].copy()
    stats = paired_stats(rows)

    rows.to_csv(args.reports_dir / "csv" / "loss_aware_primary_peeling_smoke.csv", index=False, lineterminator="\n")
    candidates.to_csv(args.reports_dir / "csv" / "loss_aware_primary_peeling_smoke_candidates.csv", index=False, lineterminator="\n")
    banks.to_csv(args.reports_dir / "csv" / "loss_aware_primary_peeling_smoke_representative_bank.csv", index=False, lineterminator="\n")
    corrected.to_csv(args.reports_dir / "csv" / "loss_aware_primary_peeling_smoke_corrected_maps.csv", index=False, lineterminator="\n")
    stats.to_csv(args.reports_dir / "csv" / "loss_aware_primary_peeling_smoke_paired_stats.csv", index=False, lineterminator="\n")
    write_report(args, settings, rows, candidates, stats, banks)

    status = run_status(rows)
    selected = rows[rows["selected_by_validation"].fillna(False)]
    best_delta = pd.to_numeric(rows["validation_delta_vs_baseline"], errors="coerce").max() if len(rows) else float("nan")
    v2_delta = rigid_v2_best_delta(args.reports_dir)
    delta_vs_v2 = float(best_delta - v2_delta) if math.isfinite(float(best_delta)) and math.isfinite(float(v2_delta)) else float("nan")
    exact_solves = exact_cochain_solve_count(candidates)
    eligible = rows.groupby("dataset")["prime"].apply(lambda vals: sorted(set(map(int, vals)))).to_dict() if len(rows) else {}
    print(f"Audit classification: {status}")
    print("Selected settings:")
    for setting in settings:
        print(f"- {setting.dataset}: {setting.run_id}")
    print("Eligible primes:")
    for dataset in datasets:
        print(f"- {dataset}: {eligible.get(dataset, [])}")
    print(f"Representative bank sizes: {len(banks)} rows")
    print(f"Exact cochain solves: {exact_solves}")
    print(f"Candidate corrections evaluated: {len(rows)}")
    print(f"Validation-selected candidates: {len(selected)}")
    print(f"Best validation delta vs baseline: {best_delta if math.isfinite(float(best_delta)) else 'not_available'}")
    print(f"Best validation delta vs rigid v2 peeling: {delta_vs_v2 if math.isfinite(float(delta_vs_v2)) else 'not_available'}")
    print(f"Controls passed: {len(selected)}")
    print("Final interpretation:")
    if status == "real_positive":
        print("- loss-aware representative peeling produced a validation-selected no-lift candidate")
    elif status == "real_negative":
        print("- primary quotient/cochain structure is useful diagnostically, but no-lift representative peeling is not reliable here")
    else:
        print(f"- {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
