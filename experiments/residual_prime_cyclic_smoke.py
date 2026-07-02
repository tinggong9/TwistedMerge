#!/usr/bin/env python
"""Peel-until-prime residual cyclic certification smoke test.

This experiment is intentionally conservative.  It enumerates peel paths whose
observed residual order is prime and then separates exact cyclic certification,
quotient-level certification, and observed-prime-LCM-only diagnostics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nonabelian_holonomy import infer_holonomy_group  # noqa: E402
from src.primary_holonomy import (  # noqa: E402
    fit_primary_quotient,
    observed_holonomy_order_lcm,
    p_adic_valuation,
    relation_count_status,
    triangle_relation_from_perms,
)

TRIANGLE_COLUMNS = [
    "setting_id",
    "run_id",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "seed",
    "alignment_source",
    "alignment_noise_fraction",
    "triangle_type",
    "triangle",
    "i",
    "j",
    "k",
    "p_ij",
    "p_jk",
    "p_ki",
    "triangle_perm",
]

RUN_COLUMNS = [
    "setting_id",
    "run_id",
    "dataset",
    "architecture",
    "n_models",
    "width",
    "domain_shift",
    "matching",
    "seed",
    "method",
    "val_accuracy",
    "val_loss",
    "test_accuracy",
    "test_loss",
    "is_ensemble_or_extra_capacity",
    "capacity_multiplier",
    "inference_multiplier",
]

PREFERRED_RUN_IDS = {
    "mnist": "mnist_mlp_N4_W64_input_noise_monomial_activation_seed4200",
    "fashion_mnist": "fashion_mnist_mlp_N4_W64_input_noise_monomial_activation_seed4200",
}

MAIN_COLUMNS = [
    "dataset",
    "run_id",
    "setting_id",
    "n_models",
    "width",
    "matching",
    "domain_shift",
    "relation_count",
    "relation_count_status",
    "observed_holonomy_order_lcm",
    "group_closure_status",
    "group_order_if_exact",
    "group_exponent_if_exact",
    "primary_source_order",
    "primary_source_order_source",
    "source_order_factorization",
    "peel_path_id",
    "peeled_primes",
    "peeled_prime_powers",
    "remaining_order_before_path",
    "residual_order_candidate",
    "residual_order_is_prime",
    "residual_prime",
    "residual_factorization",
    "certification_status",
    "certification_method",
    "certification_relation_violation_rate",
    "certification_confidence",
    "full_residual_group_order_certified",
    "cyclic_quotient_certified",
    "planned_case_status",
    "planned_method",
    "implemented_corrected_merge",
    "implemented_branch_lift",
    "branch_lift_status",
    "validation_accuracy",
    "test_accuracy",
    "baseline_method",
    "baseline_validation_accuracy",
    "baseline_test_accuracy",
    "validation_delta_vs_baseline",
    "test_delta_vs_baseline",
    "wrong_prime_control_validation_accuracy",
    "shuffled_control_validation_accuracy",
    "random_residual_control_validation_accuracy",
    "validation_delta_vs_wrong_prime_control",
    "validation_delta_vs_shuffled_control",
    "validation_delta_vs_random_residual_control",
    "capacity_multiplier",
    "inference_multiplier",
    "uses_test_for_selection",
    "selected_by_validation",
    "claim_status",
    "na_reason",
]

CERTIFICATION_COLUMNS = [
    "dataset",
    "run_id",
    "peel_path_id",
    "residual_order_candidate",
    "residual_prime",
    "group_closure_status",
    "group_order_if_exact",
    "group_exponent_if_exact",
    "quotient_fit_status",
    "quotient_relation_violation_rate",
    "quotient_nontrivial_rate",
    "quotient_entropy",
    "quotient_confidence",
    "certification_status",
    "certification_method",
    "claim_boundary",
]


@dataclass(frozen=True)
class SelectedSetting:
    dataset: str
    run_id: str
    setting_id: str
    architecture: str
    n_models: int
    width: int
    domain_shift: str
    matching: str
    seed: int
    relation_count: int
    relation_count_status: str
    observed_holonomy_order_lcm: int
    group_closure_status: str
    group_order_if_exact: int | None
    group_exponent_if_exact: int | None
    primary_source_order: int
    primary_source_order_source: str


@dataclass(frozen=True)
class CertificationEvidence:
    residual_order_candidate: int
    group_closure_status: str = "not_computed"
    group_order_if_exact: int | None = None
    quotient_relation_violation_rate: float | None = None
    quotient_nontrivial_rate: float | None = None
    quotient_entropy: float | None = None
    quotient_confidence: float | None = None
    quotient_fit_status: str = "not_attempted"
    relation_tolerance: float = 1e-9
    confidence_threshold: float = 0.75
    nontrivial_threshold: float = 0.0


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_perm(value) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not value.strip() or value == "nan":
        return None
    try:
        arr = tuple(int(item) for item in json.loads(value))
    except Exception:
        return None
    return arr if arr and sorted(arr) == list(range(len(arr))) else None


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def command_text(argv: list[str]) -> str:
    return " ".join([".venv/bin/python", "experiments/residual_prime_cyclic_smoke.py", *argv])


def is_prime(value: int | float | None) -> bool:
    if value is None or not np.isfinite(float(value)):
        return False
    n = int(value)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n))
    for factor in range(3, limit + 1, 2):
        if n % factor == 0:
            return False
    return True


def factorization_dict(value: int | float | None) -> dict[int, int]:
    if value is None or not np.isfinite(float(value)):
        return {}
    n = abs(int(value))
    if n <= 1:
        return {n: 1}
    out: dict[int, int] = {}
    factor = 2
    while factor * factor <= n:
        while n % factor == 0:
            out[factor] = out.get(factor, 0) + 1
            n //= factor
        factor += 1 if factor == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def factorization_string(value: int | float | None) -> str:
    factors = factorization_dict(value)
    if not factors:
        return ""
    terms = []
    for prime, power in sorted(factors.items()):
        if prime in {0, 1}:
            terms.append(str(prime))
        elif power == 1:
            terms.append(str(prime))
        else:
            terms.append(f"{prime}^{power}")
    return " * ".join(terms)


def remove_full_prime_powers(order: int, primes: Iterable[int]) -> tuple[int, list[str]]:
    remaining = int(order)
    powers = []
    for prime in primes:
        p = int(prime)
        mult = p_adic_valuation(remaining, p)
        if mult <= 0:
            continue
        while remaining % p == 0:
            remaining //= p
        powers.append(f"{p}^{mult}")
    return int(remaining), powers


def enumerate_prime_residual_paths(order: int, prime_list: Sequence[int], include_composite_diagnostics: bool = True) -> list[dict]:
    """Enumerate peel subsets that leave a prime residual plus key composite diagnostics."""

    source = int(order)
    eligible = [int(p) for p in prime_list if p_adic_valuation(source, int(p)) > 0]
    seen: set[tuple[int, ...]] = set()
    rows = []

    def add_path(path: Sequence[int], path_kind: str) -> None:
        key = tuple(int(p) for p in path)
        if key in seen:
            return
        seen.add(key)
        residual, powers = remove_full_prime_powers(source, key)
        prime_residual = is_prime(residual)
        rows.append(
            {
                "peel_path_id": f"path_{len(rows):03d}",
                "path_kind": path_kind,
                "peeled_primes": ",".join(str(p) for p in key),
                "peeled_prime_powers": ",".join(powers),
                "remaining_order_before_path": int(source),
                "residual_order_candidate": int(residual),
                "residual_order_is_prime": bool(prime_residual),
                "residual_prime": int(residual) if prime_residual else np.nan,
                "residual_factorization": factorization_string(residual),
            }
        )

    for size in range(0, len(eligible) + 1):
        for combo in itertools.combinations(eligible, size):
            residual, _powers = remove_full_prime_powers(source, combo)
            if is_prime(residual):
                add_path(combo, "prime_residual_candidate")

    if include_composite_diagnostics:
        small_prefix = tuple(p for p in [2, 3, 5, 7] if p in eligible)
        if small_prefix:
            residual, _powers = remove_full_prime_powers(source, small_prefix)
            if not is_prime(residual):
                add_path(small_prefix, "small_prime_prefix_composite_diagnostic")
        add_path((), "unpeeled_diagnostic")

    rows.sort(
        key=lambda row: (
            0 if row["residual_order_is_prime"] else 1,
            int(row["residual_order_candidate"]),
            row["peeled_primes"],
        )
    )
    for idx, row in enumerate(rows):
        row["peel_path_id"] = f"path_{idx:03d}"
    return rows


def certify_prime_residual(evidence: CertificationEvidence) -> dict:
    residual = int(evidence.residual_order_candidate)
    if not is_prime(residual):
        return {
            "certification_status": "not_prime_residual",
            "certification_method": "residual_order_candidate_composite",
            "certification_relation_violation_rate": np.nan,
            "certification_confidence": np.nan,
            "full_residual_group_order_certified": False,
            "cyclic_quotient_certified": False,
        }
    if (
        evidence.group_closure_status == "exact_closure"
        and evidence.group_order_if_exact is not None
        and int(evidence.group_order_if_exact) == residual
    ):
        return {
            "certification_status": "certified_full_residual_Cp",
            "certification_method": "exact_group_order_prime",
            "certification_relation_violation_rate": 0.0,
            "certification_confidence": 1.0,
            "full_residual_group_order_certified": True,
            "cyclic_quotient_certified": False,
        }
    violation = safe_float(evidence.quotient_relation_violation_rate)
    confidence = safe_float(evidence.quotient_confidence)
    nontrivial = safe_float(evidence.quotient_nontrivial_rate)
    if (
        np.isfinite(violation)
        and np.isfinite(confidence)
        and np.isfinite(nontrivial)
        and violation <= float(evidence.relation_tolerance)
        and confidence >= float(evidence.confidence_threshold)
        and nontrivial > float(evidence.nontrivial_threshold)
    ):
        return {
            "certification_status": "certified_cyclic_Cp_quotient",
            "certification_method": "cyclic_quotient_Cp_fit",
            "certification_relation_violation_rate": float(violation),
            "certification_confidence": float(confidence),
            "full_residual_group_order_certified": False,
            "cyclic_quotient_certified": True,
        }
    return {
        "certification_status": "observed_prime_lcm_only",
        "certification_method": "observed_prime_lcm_only",
        "certification_relation_violation_rate": violation if np.isfinite(violation) else np.nan,
        "certification_confidence": confidence if np.isfinite(confidence) else np.nan,
        "full_residual_group_order_certified": False,
        "cyclic_quotient_certified": False,
    }


def planned_case_decision(certification_status: str, implemented_corrected_merge: bool, implemented_branch_lift: bool) -> tuple[bool, str]:
    if certification_status == "not_prime_residual":
        return False, "not_prime_residual"
    if certification_status == "observed_prime_lcm_only":
        return False, "observed_lcm_prime_but_not_certified"
    if certification_status == "certification_failed":
        return False, "certification_failed"
    if certification_status in {"certified_full_residual_Cp", "certified_cyclic_Cp_quotient"}:
        if implemented_corrected_merge or implemented_branch_lift:
            return True, "implemented_candidate_available_requires_validation_gate"
        return False, "certified_but_merge_rerun_not_implemented"
    return False, "certification_failed"


def capacity_multiplier_for_plan(planned_method: str, residual_prime: int | float | None) -> float:
    if planned_method == "no_lift_cyclic_prime_correction":
        return 1.0
    if planned_method == "branch_lift" and residual_prime is not None and np.isfinite(float(residual_prime)):
        return float(int(residual_prime))
    return 1.0


def selection_decision(row: dict) -> tuple[bool, str]:
    if bool(row.get("uses_test_for_selection", False)):
        return False, "blocked_test_metric_selection_forbidden"
    if str(row.get("planned_case_status", "")) != "implemented_candidate_available_requires_validation_gate":
        return False, str(row.get("planned_case_status", "certification_failed"))
    val = safe_float(row.get("validation_accuracy"))
    baseline = safe_float(row.get("baseline_validation_accuracy"))
    wrong = safe_float(row.get("wrong_prime_control_validation_accuracy"))
    shuffled = safe_float(row.get("shuffled_control_validation_accuracy"))
    random_control = safe_float(row.get("random_residual_control_validation_accuracy"))
    if not np.isfinite(val):
        return False, "missing_validation_metric"
    if not np.isfinite(baseline) or val <= baseline:
        return False, "not_selected_fails_unpeeled_baseline_gate"
    for name, control in [
        ("wrong_prime", wrong),
        ("shuffled", shuffled),
        ("random_residual", random_control),
    ]:
        if not np.isfinite(control):
            return False, f"not_selected_missing_{name}_control"
        if val <= control:
            return False, f"not_selected_fails_{name}_control"
    return True, "cyclic_prime_smoke_positive_validation_selected"


def missing_metric_na_reason(row: dict) -> str:
    metric_cols = [
        "validation_accuracy",
        "test_accuracy",
        "validation_delta_vs_baseline",
        "test_delta_vs_baseline",
        "wrong_prime_control_validation_accuracy",
        "shuffled_control_validation_accuracy",
        "random_residual_control_validation_accuracy",
    ]
    missing = any(not np.isfinite(safe_float(row.get(col))) for col in metric_cols)
    if not missing:
        return ""
    status = str(row.get("planned_case_status", ""))
    cert = str(row.get("certification_status", ""))
    if status in {
        "not_prime_residual",
        "observed_lcm_prime_but_not_certified",
        "certified_but_merge_rerun_not_implemented",
        "certification_failed",
    }:
        return status
    if cert == "observed_prime_lcm_only":
        return "observed_lcm_prime_but_not_certified"
    if not bool(row.get("implemented_branch_lift", False)) and str(row.get("planned_method")) == "branch_lift":
        return "no_real_Cp_prediction_row"
    return "control_not_available"


def load_triangle_maps(reports_dir: Path, datasets: set[str], model_counts: set[int]) -> pd.DataFrame:
    artifact_dir = reports_dir / "csv" / "fixed_setting_large_artifacts"
    shards = sorted(artifact_dir.glob("fixed_setting_triangle_maps_part_*.csv.gz"))
    if not shards:
        shards = sorted(artifact_dir.glob("*triangle_maps_part_*.csv.gz"))
    if not shards:
        raise FileNotFoundError(f"no triangle map shards found under {artifact_dir}")
    frames = [pd.read_csv(path, usecols=lambda col: col in TRIANGLE_COLUMNS) for path in shards]
    maps = pd.concat(frames, ignore_index=True, sort=False)
    maps = maps[maps["triangle_type"].astype(str).eq("permutation")].copy()
    maps = maps[maps["alignment_source"].astype(str).eq("observed")].copy()
    maps = maps[pd.to_numeric(maps["alignment_noise_fraction"], errors="coerce").fillna(0.0).eq(0.0)].copy()
    if datasets:
        maps = maps[maps["dataset"].astype(str).isin(datasets)].copy()
    if model_counts:
        maps = maps[pd.to_numeric(maps["n_models"], errors="coerce").isin(model_counts)].copy()
    maps = maps.sort_values(["dataset", "n_models", "width", "domain_shift", "matching", "run_id", "triangle"])
    return maps.reset_index(drop=True)


def load_run_metrics(reports_dir: Path) -> pd.DataFrame:
    path = reports_dir / "csv" / "fixed_setting_verification_runs.csv"
    if not path.exists():
        return pd.DataFrame(columns=RUN_COLUMNS)
    runs = pd.read_csv(path, usecols=lambda col: col in RUN_COLUMNS)
    for col in ["val_accuracy", "val_loss", "test_accuracy", "test_loss", "capacity_multiplier", "inference_multiplier"]:
        if col in runs:
            runs[col] = pd.to_numeric(runs[col], errors="coerce")
    return runs


def relations_from_group(group: pd.DataFrame) -> tuple:
    relations = []
    for _, row in group.iterrows():
        p_ij = safe_perm(row.get("p_ij"))
        p_jk = safe_perm(row.get("p_jk"))
        p_ki = safe_perm(row.get("p_ki"))
        hol = safe_perm(row.get("triangle_perm"))
        if p_ij is None or p_jk is None or p_ki is None or hol is None:
            continue
        relations.append(triangle_relation_from_perms(p_ij, p_jk, p_ki, hol))
    return tuple(relations)


def summarize_relation_set(group: pd.DataFrame, max_group_order: int, max_generators: int, max_exact_order: int) -> tuple[SelectedSetting, tuple]:
    first = group.iloc[0]
    relations = relations_from_group(group)
    observed_lcm = observed_holonomy_order_lcm(relations)
    edges = []
    holonomies = []
    for relation in relations:
        edges.extend([relation.first, relation.second, relation.third])
        holonomies.append(relation.holonomy)
    summary = infer_holonomy_group(edges, holonomies, max_group_order=max_group_order, max_generators=max_generators, max_exact_order=max_exact_order)
    group_exact = summary.group_status == "exact_closure" and not summary.group.truncated and summary.group.order <= max_exact_order
    group_order = int(summary.group.order) if group_exact else None
    group_exponent = int(summary.group_exponent) if summary.group_exponent else None
    primary = group_exponent if group_exponent else int(observed_lcm)
    source = "group_exponent_if_exact" if group_exponent else "observed_holonomy_order_lcm"
    setting = SelectedSetting(
        dataset=str(first["dataset"]),
        run_id=str(first["run_id"]),
        setting_id=str(first["setting_id"]),
        architecture=str(first.get("architecture", "")),
        n_models=int(first["n_models"]),
        width=int(first["width"]),
        domain_shift=str(first.get("domain_shift", "")),
        matching=str(first.get("matching", "")),
        seed=int(first.get("seed", 0)),
        relation_count=int(len(relations)),
        relation_count_status=relation_count_status(len(relations), 4),
        observed_holonomy_order_lcm=int(observed_lcm),
        group_closure_status=str(summary.group_status),
        group_order_if_exact=group_order,
        group_exponent_if_exact=group_exponent,
        primary_source_order=int(primary),
        primary_source_order_source=source,
    )
    return setting, relations


def choose_settings(maps: pd.DataFrame, datasets: list[str], prefer_n_models: int, settings_per_dataset: int) -> dict[str, pd.DataFrame]:
    selected: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        group = maps[maps["dataset"].astype(str).eq(str(dataset))].copy()
        if group.empty:
            continue
        preferred = PREFERRED_RUN_IDS.get(str(dataset))
        if preferred and preferred in set(group["run_id"].astype(str)):
            selected[str(dataset)] = group[group["run_id"].astype(str).eq(preferred)].copy()
            continue
        candidates = []
        for run_id, run_group in group.groupby("run_id", sort=True):
            first = run_group.iloc[0]
            score = (
                0 if int(first["n_models"]) == int(prefer_n_models) else 1,
                0 if int(first["width"]) in {64, 128} else 1,
                int(first["width"]),
                str(first.get("domain_shift", "")),
                str(first.get("matching", "")),
                str(run_id),
            )
            candidates.append((score, str(run_id), run_group))
        candidates.sort(key=lambda item: item[0])
        for _score, _run_id, run_group in candidates[: int(settings_per_dataset)]:
            selected[str(dataset)] = run_group.copy()
            break
    return selected


def best_method(run_rows: pd.DataFrame, exact: Iterable[str] = (), contains: Iterable[str] = ()) -> dict | None:
    if run_rows.empty:
        return None
    mask = pd.Series(False, index=run_rows.index)
    exact_set = {str(item) for item in exact}
    if exact_set:
        mask |= run_rows["method"].astype(str).isin(exact_set)
    contains_list = [str(item) for item in contains]
    if contains_list:
        mask |= run_rows["method"].astype(str).map(lambda value: any(token in value for token in contains_list))
    subset = run_rows[mask].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["val_accuracy", "val_loss"], ascending=[False, True])
    return subset.iloc[0].to_dict()


def metric(row: dict | None, column: str) -> float:
    if row is None:
        return np.nan
    return safe_float(row.get(column))


def branch_lift_row_for_prime(run_rows: pd.DataFrame, residual_prime: int | float | None) -> dict | None:
    if residual_prime is None or not np.isfinite(float(residual_prime)):
        return None
    p = int(residual_prime)
    methods = [f"twisted_rank_lift_{p}", f"prime_C{p}_branch_lift", f"rank_lift_{p}"]
    if p == 2:
        methods.insert(0, "twisted_rank_lift_2")
    return best_method(run_rows, exact=methods)


def quotient_fit_for_candidate(relations: tuple, residual_prime: int | float | None, seed: int) -> dict:
    if residual_prime is None or not np.isfinite(float(residual_prime)):
        return {
            "quotient_fit_status": "not_attempted_nonprime_residual",
            "quotient_relation_violation_rate": np.nan,
            "quotient_nontrivial_rate": np.nan,
            "quotient_entropy": np.nan,
            "quotient_confidence": np.nan,
        }
    p = int(residual_prime)
    fit = fit_primary_quotient(relations, p, random_restarts=8, seed=seed + p)
    return {
        "quotient_fit_status": fit.quotient_fit_status,
        "quotient_relation_violation_rate": float(fit.relation_violation_rate),
        "quotient_nontrivial_rate": float(fit.quotient_holonomy_nontrivial_rate),
        "quotient_entropy": float(fit.quotient_holonomy_entropy),
        "quotient_confidence": float(fit.quotient_assignment_confidence),
    }


def build_smoke_rows(setting: SelectedSetting, relations: tuple, run_rows: pd.DataFrame, prime_list: list[int]) -> tuple[list[dict], list[dict]]:
    baseline = best_method(run_rows, exact=["c2m3_synchronized", "c2m3_permutation"])
    if baseline is None:
        baseline = best_method(run_rows, exact=["greedy_soup"])
    paths = enumerate_prime_residual_paths(setting.primary_source_order, prime_list)
    rows = []
    cert_rows = []
    for path in paths:
        residual_prime = path["residual_prime"]
        qfit = quotient_fit_for_candidate(relations, residual_prime, setting.seed) if path["residual_order_is_prime"] else quotient_fit_for_candidate(relations, None, setting.seed)
        evidence = CertificationEvidence(
            residual_order_candidate=int(path["residual_order_candidate"]),
            group_closure_status=setting.group_closure_status,
            group_order_if_exact=setting.group_order_if_exact,
            quotient_relation_violation_rate=qfit["quotient_relation_violation_rate"],
            quotient_nontrivial_rate=qfit["quotient_nontrivial_rate"],
            quotient_entropy=qfit["quotient_entropy"],
            quotient_confidence=qfit["quotient_confidence"],
            quotient_fit_status=qfit["quotient_fit_status"],
        )
        cert = certify_prime_residual(evidence)
        lift = branch_lift_row_for_prime(run_rows, residual_prime)
        implemented_branch = lift is not None
        implemented_corrected = False
        planned_method = "no_lift_cyclic_prime_correction"
        _selectable, planned_status = planned_case_decision(cert["certification_status"], implemented_corrected, implemented_branch)
        if implemented_branch and cert["certification_status"] in {"certified_full_residual_Cp", "certified_cyclic_Cp_quotient"}:
            planned_method = "branch_lift"
        capacity = capacity_multiplier_for_plan(planned_method, residual_prime)
        row = {
            **setting.__dict__,
            "source_order_factorization": factorization_string(setting.primary_source_order),
            **path,
            **cert,
            "planned_case_status": planned_status,
            "planned_method": planned_method,
            "implemented_corrected_merge": implemented_corrected,
            "implemented_branch_lift": implemented_branch,
            "branch_lift_status": "real_Cp_prediction_row_found" if implemented_branch else "no_real_Cp_prediction_row",
            "validation_accuracy": metric(lift, "val_accuracy") if implemented_branch else np.nan,
            "test_accuracy": metric(lift, "test_accuracy") if implemented_branch else np.nan,
            "baseline_method": baseline.get("method") if baseline else "",
            "baseline_validation_accuracy": metric(baseline, "val_accuracy"),
            "baseline_test_accuracy": metric(baseline, "test_accuracy"),
            "validation_delta_vs_baseline": np.nan,
            "test_delta_vs_baseline": np.nan,
            "wrong_prime_control_validation_accuracy": np.nan,
            "shuffled_control_validation_accuracy": np.nan,
            "random_residual_control_validation_accuracy": np.nan,
            "validation_delta_vs_wrong_prime_control": np.nan,
            "validation_delta_vs_shuffled_control": np.nan,
            "validation_delta_vs_random_residual_control": np.nan,
            "capacity_multiplier": float(capacity),
            "inference_multiplier": float(capacity),
            "uses_test_for_selection": False,
            "selected_by_validation": False,
        }
        if implemented_branch:
            row["validation_delta_vs_baseline"] = row["validation_accuracy"] - row["baseline_validation_accuracy"]
            row["test_delta_vs_baseline"] = row["test_accuracy"] - row["baseline_test_accuracy"]
        selected, claim = selection_decision(row)
        row["selected_by_validation"] = bool(selected)
        row["claim_status"] = claim if selected else planned_status
        row["na_reason"] = missing_metric_na_reason(row)
        rows.append(row)
        cert_rows.append(
            {
                "dataset": setting.dataset,
                "run_id": setting.run_id,
                "peel_path_id": path["peel_path_id"],
                "residual_order_candidate": path["residual_order_candidate"],
                "residual_prime": path["residual_prime"],
                "group_closure_status": setting.group_closure_status,
                "group_order_if_exact": setting.group_order_if_exact if setting.group_order_if_exact is not None else np.nan,
                "group_exponent_if_exact": setting.group_exponent_if_exact if setting.group_exponent_if_exact is not None else np.nan,
                **qfit,
                "certification_status": cert["certification_status"],
                "certification_method": cert["certification_method"],
                "claim_boundary": certification_claim_boundary(cert["certification_status"]),
            }
        )
    return rows, cert_rows


def certification_claim_boundary(status: str) -> str:
    if status == "certified_full_residual_Cp":
        return "full residual cyclic only because exact residual group order is p"
    if status == "certified_cyclic_Cp_quotient":
        return "certifies only a cyclic C_p quotient, not that the full residual group is cyclic"
    if status == "observed_prime_lcm_only":
        return "observed prime LCM only; does not rule out C_p^r or exponent-p nonabelian structure"
    if status == "not_prime_residual":
        return "composite residual diagnostic only"
    return "certification failed; no cyclic claim"


def selected_settings_frame(settings: list[SelectedSetting], prime_list: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **setting.__dict__,
                "source_order_factorization": factorization_string(setting.primary_source_order),
                "prime_list_used": ",".join(str(p) for p in prime_list),
            }
            for setting in settings
        ]
    )


def paired_stats(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    if rows.empty:
        return pd.DataFrame()
    for (dataset, status), group in rows.groupby(["dataset", "certification_status"], dropna=False, sort=True):
        out.append(
            {
                "dataset": dataset,
                "certification_status": status,
                "n_rows": int(len(group)),
                "n_prime_residual_rows": int(group["residual_order_is_prime"].fillna(False).sum()),
                "implemented_corrected_merge_rows": int(group["implemented_corrected_merge"].fillna(False).sum()),
                "implemented_branch_lift_rows": int(group["implemented_branch_lift"].fillna(False).sum()),
                "selected_by_validation_rows": int(group["selected_by_validation"].fillna(False).sum()),
                "mean_validation_delta_vs_baseline": float(pd.to_numeric(group["validation_delta_vs_baseline"], errors="coerce").mean()),
                "claim_status": "diagnostic_smoke_summary",
            }
        )
    return pd.DataFrame(out)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if not np.isfinite(value) else f"{value:.6g}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(args: argparse.Namespace, settings: list[SelectedSetting], rows: pd.DataFrame, cert: pd.DataFrame, stats: pd.DataFrame) -> None:
    selected = selected_settings_frame(settings, parse_csv(args.prime_list, int))
    prime_candidates = rows[rows["residual_order_is_prime"].fillna(False)].copy()
    full = cert[cert["certification_status"].astype(str).eq("certified_full_residual_Cp")]
    quotient = cert[cert["certification_status"].astype(str).eq("certified_cyclic_Cp_quotient")]
    lcm_only = cert[cert["certification_status"].astype(str).eq("observed_prime_lcm_only")]
    not_prime = cert[cert["certification_status"].astype(str).eq("not_prime_residual")]
    corrected = rows[rows["implemented_corrected_merge"].fillna(False)]
    branch = rows[rows["implemented_branch_lift"].fillna(False)]
    selected_rows = rows[rows["selected_by_validation"].fillna(False)]
    if len(selected_rows):
        final = "cyclic-prime smoke positive"
        blocked = "not blocked"
    elif len(full) or len(quotient):
        final = "diagnostic only"
        blocked = "certified residual/quotient exists but no corrected merge or C_p lift metrics passed gates"
    else:
        final = "diagnostic only"
        blocked = "prime residual candidates are observed-LCM-only or composite diagnostics"
    text = f"""# Residual Prime Cyclic Smoke Test

Generated by `experiments/residual_prime_cyclic_smoke.py`.

## Exact Command

```bash
{command_text(sys.argv[1:])}
```

## Git State

- Git commit: `{git_output("rev-parse", "--short", "HEAD")}`
- Dirty status (tracked files only): `{git_output("status", "--short", "--untracked-files=no") or "clean"}`

## Scope And Safe Wording

- This is a two-setting smoke test.
- The statement "prime residual order implies cyclic" is used only when the actual residual group/order or certified quotient has order p.
- Observed prime LCM alone is not enough to certify a cyclic residual group.
- Positive rows are hypothesis-generating only.
- This does not prove real Brauer/projective or period-index structure.
- This does not prove broad model-merging improvement.

## Selected Settings

{md_table(selected, ["dataset", "run_id", "setting_id", "n_models", "width", "matching", "relation_count", "relation_count_status", "observed_holonomy_order_lcm", "group_closure_status", "group_order_if_exact", "group_exponent_if_exact", "primary_source_order"])}

## Source Order And Factorization

{md_table(rows.drop_duplicates(["dataset", "run_id"]), ["dataset", "run_id", "observed_holonomy_order_lcm", "primary_source_order", "primary_source_order_source", "source_order_factorization"], 20)}

## Peel Paths Considered

{md_table(rows, ["dataset", "peel_path_id", "peeled_primes", "peeled_prime_powers", "remaining_order_before_path", "residual_order_candidate", "residual_order_is_prime", "residual_prime", "residual_factorization", "certification_status"], 80)}

## Residual Prime Candidates

{md_table(prime_candidates, ["dataset", "peel_path_id", "peeled_primes", "residual_prime", "certification_status", "certification_method", "planned_case_status", "branch_lift_status", "claim_status", "na_reason"], 80)}

## Certification Table

{md_table(cert, ["dataset", "peel_path_id", "residual_order_candidate", "residual_prime", "group_closure_status", "group_order_if_exact", "quotient_relation_violation_rate", "quotient_confidence", "certification_status", "certification_method", "claim_boundary"], 80)}

## Planned Cyclic-Prime Method Table

{md_table(rows, ["dataset", "peel_path_id", "planned_method", "implemented_corrected_merge", "implemented_branch_lift", "validation_accuracy", "baseline_method", "baseline_validation_accuracy", "validation_delta_vs_baseline", "capacity_multiplier", "inference_multiplier", "selected_by_validation", "claim_status", "na_reason"], 80)}

## Controls

{md_table(rows, ["dataset", "peel_path_id", "wrong_prime_control_validation_accuracy", "shuffled_control_validation_accuracy", "random_residual_control_validation_accuracy", "validation_delta_vs_wrong_prime_control", "validation_delta_vs_shuffled_control", "validation_delta_vs_random_residual_control", "na_reason"], 80)}

## Paired Stats

{md_table(stats, ["dataset", "certification_status", "n_rows", "n_prime_residual_rows", "implemented_corrected_merge_rows", "implemented_branch_lift_rows", "selected_by_validation_rows", "mean_validation_delta_vs_baseline", "claim_status"], 80)}

## Final Console Summary

Selected settings:
- MNIST: `{", ".join(selected[selected["dataset"].eq("mnist")]["run_id"].astype(str).tolist()) or "none"}`
- Fashion-MNIST: `{", ".join(selected[selected["dataset"].eq("fashion_mnist")]["run_id"].astype(str).tolist()) or "none"}`

Source orders:
- MNIST: `{", ".join(selected[selected["dataset"].eq("mnist")]["primary_source_order"].astype(str).tolist()) or "none"}`
- Fashion-MNIST: `{", ".join(selected[selected["dataset"].eq("fashion_mnist")]["primary_source_order"].astype(str).tolist()) or "none"}`

Prime residual candidates:
- `{len(prime_candidates)}` rows: `{", ".join(prime_candidates["dataset"].astype(str) + ":" + prime_candidates["residual_prime"].fillna("").astype(str)) if len(prime_candidates) else "none"}`

Certification:
- full residual C_p: `{len(full)}`
- cyclic C_p quotient: `{len(quotient)}`
- observed prime LCM only: `{len(lcm_only)}`
- failed/not-prime: `{len(not_prime)}`

Planned cyclic-prime rows:
- corrected merge implemented: `{'yes' if len(corrected) else 'no'}` / `{len(corrected)}`
- branch lift implemented: `{'yes' if len(branch) else 'no'}` / `{len(branch)}`

Accuracy deltas:
- vs baseline: `not_available_without_implemented_corrected_merge_or_Cp_lift`
- vs controls: `not_available_controls_not_run`

Final interpretation:
- `{final}`
- what blocked success: `{blocked}`
- recommended next experiment: `implement an audited C_p quotient correction adapter, then rerun C2M3/monomial synchronization with validation-only controls`
"""
    (args.reports_dir / "residual_prime_cyclic_smoke_report.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="4,3")
    parser.add_argument("--settings-per-dataset", type=int, default=1)
    parser.add_argument("--prime-list", default="2,3,5,7,11,13,17,19,23,29,31,43")
    parser.add_argument("--prefer-n-models", type=int, default=4)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-group-order", type=int, default=50000)
    parser.add_argument("--max-generators", type=int, default=6)
    parser.add_argument("--max-exact-order", type=int, default=50000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "csv").mkdir(parents=True, exist_ok=True)
    datasets = parse_csv(args.datasets, str)
    prime_list = parse_csv(args.prime_list, int)
    maps = load_triangle_maps(args.reports_dir, set(datasets), set(parse_csv(args.model_counts, int)))
    runs = load_run_metrics(args.reports_dir)
    chosen = choose_settings(maps, datasets, args.prefer_n_models, args.settings_per_dataset)
    if len(chosen) < len(datasets):
        missing = sorted(set(datasets) - set(chosen))
        raise RuntimeError(f"could not select settings for datasets: {missing}")

    settings = []
    all_rows = []
    all_cert = []
    for dataset in datasets:
        group = chosen[dataset]
        setting, relations = summarize_relation_set(group, args.max_group_order, args.max_generators, args.max_exact_order)
        settings.append(setting)
        run_rows = runs[runs["run_id"].astype(str).eq(setting.run_id)].copy()
        rows, cert = build_smoke_rows(setting, relations, run_rows, prime_list)
        all_rows.extend(rows)
        all_cert.extend(cert)

    rows = pd.DataFrame(all_rows)
    cert = pd.DataFrame(all_cert)
    for col in MAIN_COLUMNS:
        if col not in rows:
            rows[col] = np.nan
    for col in CERTIFICATION_COLUMNS:
        if col not in cert:
            cert[col] = np.nan
    rows = rows[MAIN_COLUMNS + [col for col in rows.columns if col not in MAIN_COLUMNS]].copy()
    cert = cert[CERTIFICATION_COLUMNS + [col for col in cert.columns if col not in CERTIFICATION_COLUMNS]].copy()
    stats = paired_stats(rows)
    selected = selected_settings_frame(settings, prime_list)

    rows.to_csv(args.reports_dir / "csv" / "residual_prime_cyclic_smoke.csv", index=False, lineterminator="\n")
    selected.to_csv(args.reports_dir / "csv" / "residual_prime_cyclic_smoke_selected_settings.csv", index=False, lineterminator="\n")
    cert.to_csv(args.reports_dir / "csv" / "residual_prime_cyclic_smoke_certification.csv", index=False, lineterminator="\n")
    stats.to_csv(args.reports_dir / "csv" / "residual_prime_cyclic_smoke_paired_stats.csv", index=False, lineterminator="\n")
    write_report(args, settings, rows, cert, stats)

    prime_candidates = rows[rows["residual_order_is_prime"].fillna(False)].copy()
    full = cert[cert["certification_status"].astype(str).eq("certified_full_residual_Cp")]
    quotient = cert[cert["certification_status"].astype(str).eq("certified_cyclic_Cp_quotient")]
    lcm_only = cert[cert["certification_status"].astype(str).eq("observed_prime_lcm_only")]
    not_prime = cert[cert["certification_status"].astype(str).eq("not_prime_residual")]
    corrected = rows[rows["implemented_corrected_merge"].fillna(False)]
    branch = rows[rows["implemented_branch_lift"].fillna(False)]
    selected_rows = rows[rows["selected_by_validation"].fillna(False)]

    print("Selected settings:")
    for setting in settings:
        print(f"- {setting.dataset}: {setting.run_id} (N={setting.n_models}, W={setting.width}, {setting.matching})")
    print("\nSource orders:")
    for setting in settings:
        print(f"- {setting.dataset}: {setting.primary_source_order} = {factorization_string(setting.primary_source_order)}")
    print("\nPrime residual candidates:")
    if prime_candidates.empty:
        print("- none")
    else:
        for _, row in prime_candidates.iterrows():
            print(f"- {row['dataset']} {row['peel_path_id']}: peel {row['peeled_primes']} -> {int(row['residual_prime'])} ({row['certification_status']})")
    print("\nCertification:")
    print(f"- full residual C_p: {len(full)}")
    print(f"- cyclic C_p quotient: {len(quotient)}")
    print(f"- observed prime LCM only: {len(lcm_only)}")
    print(f"- failed/not-prime: {len(not_prime)}")
    print("\nPlanned cyclic-prime rows:")
    print(f"- corrected merge implemented: {'yes' if len(corrected) else 'no'}/{len(corrected)}")
    print(f"- branch lift implemented: {'yes' if len(branch) else 'no'}/{len(branch)}")
    print("\nAccuracy deltas:")
    print("- vs baseline: not_available_without_implemented_corrected_merge_or_Cp_lift")
    print("- vs controls: not_available_controls_not_run")
    print("\nFinal interpretation:")
    if len(selected_rows):
        print("- cyclic-prime smoke positive")
        print("- what blocked success: not blocked")
    else:
        print("- diagnostic only")
        print("- what blocked success: no certified implemented corrected merge or C_p lift metrics passed gates")
    print("- recommended next experiment: implement an audited C_p quotient correction adapter and rerun validation-only controls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
