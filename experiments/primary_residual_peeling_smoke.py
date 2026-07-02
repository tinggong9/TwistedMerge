#!/usr/bin/env python
"""No-lift p-primary residual peeling smoke test.

This is deliberately small and conservative.  It fits quotient-level primary
residual corrections from existing triangle permutation artifacts, but it does
not fabricate model-level corrected merges when no safe representative
correction path is available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
]

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
    "group_exponent_if_exact",
    "primary_source_order",
    "primary_source_order_source",
    "prime",
    "prime_index",
    "p_adic_multiplicity",
    "eligible",
    "remaining_order_before",
    "remaining_order_after",
    "peel_mode",
    "cumulative_primes",
    "quotient_fit_status",
    "quotient_relation_violation_rate",
    "edge_correction_status",
    "corrected_cycle_residual_before",
    "corrected_cycle_residual_after",
    "correction_reduces_residual",
    "method",
    "baseline_method",
    "implemented_corrected_merge",
    "validation_accuracy",
    "test_accuracy",
    "baseline_validation_accuracy",
    "baseline_test_accuracy",
    "validation_delta_vs_baseline",
    "test_delta_vs_baseline",
    "wrong_prime_control_validation_accuracy",
    "wrong_prime_control_test_accuracy",
    "shuffled_control_validation_accuracy",
    "shuffled_control_test_accuracy",
    "random_residual_control_validation_accuracy",
    "random_residual_control_test_accuracy",
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
    group_exponent_if_exact: int | None
    primary_source_order: int
    primary_source_order_source: str


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_perm(value) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not value.strip() or value == "nan":
        return None
    try:
        arr = tuple(int(item) for item in json.loads(value))
    except Exception:
        return None
    return arr if arr and sorted(arr) == list(range(len(arr))) else None


def p_adic_multiplicity(order: int | float | None, prime: int) -> int:
    return p_adic_valuation(order, prime)


def peel_once(remaining_order: int, prime: int) -> dict:
    before = int(max(1, remaining_order))
    multiplicity = p_adic_multiplicity(before, prime)
    eligible = multiplicity > 0
    after = before
    if eligible:
        while after % int(prime) == 0:
            after //= int(prime)
    return {
        "prime": int(prime),
        "p_adic_multiplicity": int(multiplicity),
        "eligible": bool(eligible),
        "remaining_order_before": int(before),
        "remaining_order_after": int(after),
    }


def prime_peeling_plan(primary_source_order: int, primes: Iterable[int]) -> list[dict]:
    remaining = int(max(1, primary_source_order))
    rows = []
    cumulative = []
    for idx, prime in enumerate(primes):
        row = peel_once(remaining, int(prime))
        row["prime_index"] = int(idx)
        if row["eligible"]:
            cumulative.append(str(int(prime)))
        row["cumulative_primes"] = ",".join(cumulative)
        rows.append(row)
        remaining = int(row["remaining_order_after"])
    return rows


def correction_is_safe(
    eligible: bool,
    quotient_relation_violation_rate: float,
    corrected_cycle_residual_before: float,
    corrected_cycle_residual_after: float,
) -> bool:
    if not bool(eligible):
        return False
    values = [
        quotient_relation_violation_rate,
        corrected_cycle_residual_before,
        corrected_cycle_residual_after,
    ]
    if not all(np.isfinite(float(value)) for value in values):
        return False
    return bool(
        float(quotient_relation_violation_rate) <= 0.01
        and float(corrected_cycle_residual_after) < float(corrected_cycle_residual_before)
    )


def no_lift_capacity_metadata() -> dict:
    return {"capacity_multiplier": 1.0, "inference_multiplier": 1.0}


def selection_decision(row: dict) -> tuple[bool, str]:
    if bool(row.get("uses_test_for_selection", False)):
        return False, "blocked_test_metric_selection_forbidden"
    if not bool(row.get("eligible", False)):
        return False, "prime_not_eligible"
    if not bool(row.get("correction_reduces_residual", False)):
        return False, "no_safe_quotient_fit"
    if not bool(row.get("implemented_corrected_merge", False)):
        return False, "merge_rerun_not_implemented"
    val = safe_float(row.get("validation_accuracy"))
    baseline = safe_float(row.get("baseline_validation_accuracy"))
    wrong = safe_float(row.get("wrong_prime_control_validation_accuracy"))
    shuffled = safe_float(row.get("shuffled_control_validation_accuracy"))
    random_resid = safe_float(row.get("random_residual_control_validation_accuracy"))
    if not np.isfinite(val):
        return False, "missing_corrected_validation_metric"
    if not np.isfinite(baseline) or val <= baseline:
        return False, "not_selected_fails_unpeeled_baseline_gate"
    if not np.isfinite(wrong):
        return False, "not_selected_missing_wrong_prime_control"
    if val <= wrong:
        return False, "not_selected_fails_wrong_prime_control"
    if not np.isfinite(shuffled):
        return False, "not_selected_missing_shuffled_control"
    if val <= shuffled:
        return False, "not_selected_fails_shuffled_control"
    if not np.isfinite(random_resid):
        return False, "not_selected_missing_random_residual_control"
    if val <= random_resid:
        return False, "not_selected_fails_random_residual_control"
    return True, "smoke_positive_validation_selected"


def na_reason_for_metrics(row: dict, default_reason: str) -> str:
    has_val = np.isfinite(safe_float(row.get("validation_accuracy")))
    has_test = np.isfinite(safe_float(row.get("test_accuracy")))
    if has_val and has_test:
        return ""
    return default_reason


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def command_text(argv: list[str]) -> str:
    return " ".join([".venv/bin/python", "experiments/primary_residual_peeling_smoke.py", *argv])


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
    return maps.sort_values(["dataset", "n_models", "width", "domain_shift", "matching", "run_id", "triangle"])


def load_run_metrics(reports_dir: Path) -> pd.DataFrame:
    path = reports_dir / "csv" / "fixed_setting_verification_runs.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    runs = pd.read_csv(path, usecols=lambda col: col in RUN_COLUMNS)
    for col in ["val_accuracy", "val_loss", "test_accuracy", "test_loss"]:
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


def summarize_relation_set(group: pd.DataFrame, max_group_order: int, max_generators: int, max_exact_order: int) -> SelectedSetting:
    first = group.iloc[0]
    relations = relations_from_group(group)
    observed_lcm = observed_holonomy_order_lcm(relations)
    edges = []
    holonomies = []
    for relation in relations:
        edges.extend([relation.first, relation.second, relation.third])
        holonomies.append(relation.holonomy)
    summary = infer_holonomy_group(
        edges,
        holonomies,
        max_group_order=int(max_group_order),
        max_generators=int(max_generators),
        max_exact_order=int(max_exact_order),
    )
    group_exponent = summary.group_exponent
    return SelectedSetting(
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
        group_exponent_if_exact=int(group_exponent) if group_exponent else None,
        primary_source_order=int(group_exponent) if group_exponent else int(observed_lcm),
        primary_source_order_source="group_exponent_if_exact" if group_exponent else "observed_holonomy_order_lcm",
    )


def choose_settings(
    maps: pd.DataFrame,
    datasets: list[str],
    prefer_n_models: int,
    settings_per_dataset: int,
) -> dict[str, pd.DataFrame]:
    selected = {}
    preferred_run_ids = {
        "mnist": "mnist_mlp_N4_W64_input_noise_monomial_activation_seed4200",
        "fashion_mnist": "fashion_mnist_mlp_N4_W64_input_noise_monomial_activation_seed4200",
    }
    for dataset in datasets:
        group = maps[maps["dataset"].astype(str).eq(str(dataset))].copy()
        if group.empty:
            continue
        preferred = preferred_run_ids.get(str(dataset))
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
        for _, _run_id, run_group in candidates[: int(settings_per_dataset)]:
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


def baseline_rows(setting: SelectedSetting, run_rows: pd.DataFrame) -> list[dict]:
    baselines = [
        ("baseline_greedy_soup", "greedy_soup", best_method(run_rows, exact=["greedy_soup"])),
        ("baseline_c2m3_permutation", "c2m3_synchronized", best_method(run_rows, exact=["c2m3_synchronized", "c2m3_permutation"])),
        ("baseline_monomial_scale", "monomial_scale", best_method(run_rows, contains=["monomial"])),
    ]
    rows = []
    for method, baseline_method, source in baselines:
        row = {
            **setting.__dict__,
            "prime": 0,
            "prime_index": -1,
            "p_adic_multiplicity": 0,
            "eligible": False,
            "remaining_order_before": setting.primary_source_order,
            "remaining_order_after": setting.primary_source_order,
            "peel_mode": "baseline",
            "cumulative_primes": "",
            "quotient_fit_status": "not_applicable_baseline",
            "quotient_relation_violation_rate": np.nan,
            "edge_correction_status": "not_applicable_baseline",
            "corrected_cycle_residual_before": np.nan,
            "corrected_cycle_residual_after": np.nan,
            "correction_reduces_residual": False,
            "method": method,
            "baseline_method": baseline_method,
            "implemented_corrected_merge": False,
            "validation_accuracy": metric(source, "val_accuracy"),
            "test_accuracy": metric(source, "test_accuracy"),
            "baseline_validation_accuracy": metric(source, "val_accuracy"),
            "baseline_test_accuracy": metric(source, "test_accuracy"),
            "validation_delta_vs_baseline": 0.0 if source is not None else np.nan,
            "test_delta_vs_baseline": 0.0 if source is not None else np.nan,
            "wrong_prime_control_validation_accuracy": np.nan,
            "wrong_prime_control_test_accuracy": np.nan,
            "shuffled_control_validation_accuracy": np.nan,
            "shuffled_control_test_accuracy": np.nan,
            "random_residual_control_validation_accuracy": np.nan,
            "random_residual_control_test_accuracy": np.nan,
            "validation_delta_vs_wrong_prime_control": np.nan,
            "validation_delta_vs_shuffled_control": np.nan,
            "validation_delta_vs_random_residual_control": np.nan,
            **no_lift_capacity_metadata(),
            "uses_test_for_selection": False,
            "selected_by_validation": False,
            "claim_status": "baseline_reference",
            "na_reason": "" if source is not None else "missing_baseline_metric",
        }
        rows.append(row)
    return rows


def quotient_diagnostic(relations: tuple, peel: dict, seed: int) -> dict:
    if not peel["eligible"]:
        return {
            "quotient_fit_status": "prime_not_eligible",
            "quotient_relation_violation_rate": np.nan,
            "edge_correction_status": "prime_not_eligible",
            "corrected_cycle_residual_before": np.nan,
            "corrected_cycle_residual_after": np.nan,
            "correction_reduces_residual": False,
        }
    fit = fit_primary_quotient(relations, int(peel["prime"]), random_restarts=8, seed=seed)
    before = float(fit.quotient_holonomy_nontrivial_rate)
    after = float(fit.relation_violation_rate)
    safe = correction_is_safe(True, after, before, after)
    return {
        "quotient_fit_status": fit.quotient_fit_status,
        "quotient_relation_violation_rate": after,
        "edge_correction_status": "quotient_edge_correction_found" if safe else "no_safe_quotient_fit",
        "corrected_cycle_residual_before": before,
        "corrected_cycle_residual_after": after,
        "correction_reduces_residual": safe,
    }


def peel_rows(setting: SelectedSetting, run_rows: pd.DataFrame, relations: tuple, primes: list[int]) -> list[dict]:
    baseline_c2m3 = best_method(run_rows, exact=["c2m3_synchronized", "c2m3_permutation"])
    baseline_mono = best_method(run_rows, contains=["monomial"])
    rows = []
    plan = prime_peeling_plan(setting.primary_source_order, primes)
    eligible_seen = []
    for peel in plan:
        if peel["eligible"]:
            eligible_seen.append(int(peel["prime"]))
        diag = quotient_diagnostic(relations, peel, seed=setting.seed + int(peel["prime"]))
        base_meta = {**setting.__dict__, **peel, **diag, **no_lift_capacity_metadata(), "uses_test_for_selection": False}
        method_specs = [
            ("peel_p_then_c2m3", "c2m3_synchronized", baseline_c2m3, str(peel["prime"])),
            ("peel_p_then_monomial", "monomial_scale", baseline_mono, str(peel["prime"])),
            ("cumulative_peel_then_c2m3", "c2m3_synchronized", baseline_c2m3, ",".join(str(p) for p in eligible_seen)),
            ("cumulative_peel_then_monomial", "monomial_scale", baseline_mono, ",".join(str(p) for p in eligible_seen)),
            ("wrong_prime_peel_control", "c2m3_synchronized", baseline_c2m3, str(peel["prime"])),
            ("shuffled_prime_quotient_control", "c2m3_synchronized", baseline_c2m3, str(peel["prime"])),
            ("random_same_residual_norm_peel_control", "c2m3_synchronized", baseline_c2m3, str(peel["prime"])),
        ]
        for method, baseline_method, baseline, cumulative in method_specs:
            if not peel["eligible"]:
                na_reason = "prime_not_eligible"
            elif not diag["correction_reduces_residual"]:
                na_reason = "no_safe_quotient_fit"
            elif baseline is None:
                na_reason = "missing_baseline_metric"
            else:
                na_reason = "merge_rerun_not_implemented"
            row = {
                **base_meta,
                "peel_mode": "peel_p_only" if method.startswith("peel_p") else ("cumulative" if method.startswith("cumulative") else "control"),
                "cumulative_primes": cumulative,
                "method": method,
                "baseline_method": baseline_method,
                "implemented_corrected_merge": False,
                "validation_accuracy": np.nan,
                "test_accuracy": np.nan,
                "baseline_validation_accuracy": metric(baseline, "val_accuracy"),
                "baseline_test_accuracy": metric(baseline, "test_accuracy"),
                "validation_delta_vs_baseline": np.nan,
                "test_delta_vs_baseline": np.nan,
                "wrong_prime_control_validation_accuracy": np.nan,
                "wrong_prime_control_test_accuracy": np.nan,
                "shuffled_control_validation_accuracy": np.nan,
                "shuffled_control_test_accuracy": np.nan,
                "random_residual_control_validation_accuracy": np.nan,
                "random_residual_control_test_accuracy": np.nan,
                "validation_delta_vs_wrong_prime_control": np.nan,
                "validation_delta_vs_shuffled_control": np.nan,
                "validation_delta_vs_random_residual_control": np.nan,
                "selected_by_validation": False,
                "na_reason": na_reason,
            }
            selected, status = selection_decision(row)
            row["selected_by_validation"] = bool(selected)
            row["claim_status"] = "diagnostic_only" if status == "merge_rerun_not_implemented" else status
            rows.append(row)
    return rows


def paired_stats(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    data = rows[~rows["method"].astype(str).str.startswith("baseline_")].copy()
    for (method, baseline), group in data.groupby(["method", "baseline_method"], dropna=False, sort=True):
        deltas = pd.to_numeric(group["validation_delta_vs_baseline"], errors="coerce").dropna()
        out.append(
            {
                "method": method,
                "baseline_method": baseline,
                "n_rows": int(len(group)),
                "n_finite_validation_delta": int(len(deltas)),
                "mean_validation_delta_vs_baseline": float(deltas.mean()) if len(deltas) else np.nan,
                "implemented_corrected_merge_rows": int(group["implemented_corrected_merge"].fillna(False).sum()),
                "selected_by_validation_rows": int(group["selected_by_validation"].fillna(False).sum()),
                "claim_status": "diagnostic_only_no_corrected_merge_accuracy",
            }
        )
    return pd.DataFrame(out)


def selected_settings_frame(settings: list[SelectedSetting], primes: list[int]) -> pd.DataFrame:
    return pd.DataFrame([{**setting.__dict__, "prime_list_used": ",".join(str(p) for p in primes)} for setting in settings])


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


def write_report(args, settings: list[SelectedSetting], rows: pd.DataFrame, stats: pd.DataFrame) -> None:
    selected = selected_settings_frame(settings, parse_csv(args.prime_list, int))
    eligible = (
        rows[(rows["eligible"].fillna(False)) & (rows["method"].astype(str).eq("peel_p_then_c2m3"))]
        .groupby("dataset")["prime"]
        .apply(lambda vals: ",".join(str(int(v)) for v in vals))
        .to_dict()
    )
    safe = rows[
        rows["correction_reduces_residual"].fillna(False)
        & rows["method"].astype(str).eq("peel_p_then_c2m3")
    ]
    implemented = rows[rows["implemented_corrected_merge"].fillna(False)]
    selected_rows = rows[rows["selected_by_validation"].fillna(False)]
    best_c2m3 = rows[rows["method"].astype(str).eq("peel_p_then_c2m3")]["validation_delta_vs_baseline"].dropna()
    best_mono = rows[rows["method"].astype(str).eq("peel_p_then_monomial")]["validation_delta_vs_baseline"].dropna()
    best_cum = rows[rows["method"].astype(str).str.startswith("cumulative")]["validation_delta_vs_baseline"].dropna()
    controls_passed = bool(False)
    final_status = "diagnostic-only; correction implementation or merge rerun path missing"
    text = f"""# Primary Residual Peeling Smoke Test

Generated by `experiments/primary_residual_peeling_smoke.py`.

## Exact Command

```bash
{command_text(sys.argv[1:])}
```

## Git State

- Git commit: `{git_output("rev-parse", "--short", "HEAD")}`
- Dirty status (tracked files only): `{git_output("status", "--short", "--untracked-files=no") or "clean"}`

## Scope

- This is a two-setting smoke test.
- Positive rows are hypothesis-generating only.
- This is no-lift primary residual peeling, not a branch/rank lift.
- This does not prove real Brauer/projective or period-index structure.
- This does not prove broad model-merging improvement.
- If no corrected merge is implemented, the result is diagnostic only.

## Prime List

`{args.prime_list}`

## Selected Settings

{md_table(selected, ["dataset", "run_id", "setting_id", "n_models", "width", "matching", "relation_count", "relation_count_status", "observed_holonomy_order_lcm", "group_closure_status", "primary_source_order"])}

## Eligible Primes

`{json.dumps(eligible, sort_keys=True)}`

## Quotient And Edge-Correction Diagnostics

{md_table(rows[rows["method"].astype(str).eq("peel_p_then_c2m3")], ["dataset", "prime", "p_adic_multiplicity", "eligible", "quotient_fit_status", "quotient_relation_violation_rate", "edge_correction_status", "corrected_cycle_residual_before", "corrected_cycle_residual_after", "correction_reduces_residual", "na_reason"], 40)}

## Method Rows

{md_table(rows, ["dataset", "prime", "peel_mode", "method", "baseline_method", "implemented_corrected_merge", "validation_accuracy", "test_accuracy", "baseline_validation_accuracy", "validation_delta_vs_baseline", "capacity_multiplier", "inference_multiplier", "selected_by_validation", "claim_status", "na_reason"], 80)}

## Paired Stats

{md_table(stats, ["method", "baseline_method", "n_rows", "n_finite_validation_delta", "mean_validation_delta_vs_baseline", "implemented_corrected_merge_rows", "selected_by_validation_rows", "claim_status"], 40)}

## Final Console-Style Summary

Selected settings:
- MNIST: `{", ".join(selected[selected["dataset"].eq("mnist")]["run_id"].astype(str).tolist()) or "none"}`
- Fashion-MNIST: `{", ".join(selected[selected["dataset"].eq("fashion_mnist")]["run_id"].astype(str).tolist()) or "none"}`

Eligible primes:
- MNIST: `{eligible.get("mnist", "")}`
- Fashion-MNIST: `{eligible.get("fashion_mnist", "")}`

Safe quotient fits:
- count: `{len(safe)}`

Safe edge corrections:
- count: `{len(safe)}`

Corrected merge rows implemented:
- count: `{len(implemented)}`

Validation-selected peeled methods:
- count: `{len(selected_rows)}`
- details: `{", ".join(selected_rows["method"].astype(str).tolist()) if len(selected_rows) else "none"}`

Best validation deltas:
- peeled C2M3 vs unpeeled C2M3: `{float(best_c2m3.max()) if len(best_c2m3) else "not_run"}`
- peeled monomial vs unpeeled monomial: `{float(best_mono.max()) if len(best_mono) else "not_run"}`
- cumulative peeling vs baseline: `{float(best_cum.max()) if len(best_cum) else "not_run"}`

Controls:
- wrong-prime control passed/failed: `not_run`
- shuffled-quotient control passed/failed: `not_run`
- random-residual control passed/failed: `not_run`

Final interpretation:
- `{final_status}`
- what blocked success: `no audited representative quotient-correction-to-permutation/monomial-map path and no corrected merge rerun path`
- recommended next experiment: `add a model-level correction adapter that converts safe quotient edge corrections into layerwise maps, then rerun C2M3 first with saved checkpoints and validation-only controls`

Control gates passed: `{controls_passed}`
"""
    (args.reports_dir / "primary_residual_peeling_smoke_report.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="4,3")
    parser.add_argument("--settings-per-dataset", type=int, default=1)
    parser.add_argument("--prime-list", default="2,3,5,7,11,13,17,23,29,31")
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
    primes = parse_csv(args.prime_list, int)
    maps = load_triangle_maps(args.reports_dir, set(datasets), set(parse_csv(args.model_counts, int)))
    runs = load_run_metrics(args.reports_dir)
    chosen = choose_settings(maps, datasets, args.prefer_n_models, args.settings_per_dataset)
    if len(chosen) < len(datasets):
        missing = sorted(set(datasets) - set(chosen))
        raise RuntimeError(f"could not select settings for datasets: {missing}")

    settings = []
    all_rows = []
    for dataset in datasets:
        group = chosen[dataset]
        setting = summarize_relation_set(group, args.max_group_order, args.max_generators, args.max_exact_order)
        settings.append(setting)
        run_rows = runs[runs["run_id"].astype(str).eq(setting.run_id)].copy()
        relations = relations_from_group(group)
        all_rows.extend(baseline_rows(setting, run_rows))
        all_rows.extend(peel_rows(setting, run_rows, relations, primes))

    rows = pd.DataFrame(all_rows)
    for col in MAIN_COLUMNS:
        if col not in rows:
            rows[col] = np.nan
    rows = rows[MAIN_COLUMNS + [col for col in rows.columns if col not in MAIN_COLUMNS]].copy()
    stats = paired_stats(rows)
    selected = selected_settings_frame(settings, primes)

    rows.to_csv(args.reports_dir / "csv" / "primary_residual_peeling_smoke.csv", index=False, lineterminator="\n")
    stats.to_csv(args.reports_dir / "csv" / "primary_residual_peeling_smoke_paired_stats.csv", index=False, lineterminator="\n")
    selected.to_csv(args.reports_dir / "csv" / "primary_residual_peeling_smoke_selected_settings.csv", index=False, lineterminator="\n")
    write_report(args, settings, rows, stats)

    eligible = (
        rows[(rows["eligible"].fillna(False)) & (rows["method"].astype(str).eq("peel_p_then_c2m3"))]
        .groupby("dataset")["prime"]
        .apply(lambda vals: list(map(int, vals)))
        .to_dict()
    )
    safe = rows[
        rows["correction_reduces_residual"].fillna(False)
        & rows["method"].astype(str).eq("peel_p_then_c2m3")
    ]
    implemented = rows[rows["implemented_corrected_merge"].fillna(False)]
    selected_rows = rows[rows["selected_by_validation"].fillna(False)]
    print("Selected settings:")
    for setting in settings:
        print(f"- {setting.dataset}: {setting.run_id} (N={setting.n_models}, W={setting.width}, {setting.matching})")
    print("\nEligible primes:")
    for dataset in datasets:
        print(f"- {dataset}: {eligible.get(dataset, [])}")
    print("\nSafe quotient fits:")
    print(f"- count: {len(safe)}")
    print("\nSafe edge corrections:")
    print(f"- count: {len(safe)}")
    print("\nCorrected merge rows implemented:")
    print(f"- count: {len(implemented)}")
    print("\nValidation-selected peeled methods:")
    print(f"- count: {len(selected_rows)}")
    print(f"- details: {', '.join(selected_rows['method'].astype(str).tolist()) if len(selected_rows) else 'none'}")
    print("\nBest validation deltas:")
    print("- peeled C2M3 vs unpeeled C2M3: not_run")
    print("- peeled monomial vs unpeeled monomial: not_run")
    print("- cumulative peeling vs baseline: not_run")
    print("\nControls:")
    print("- wrong-prime control passed/failed: not_run")
    print("- shuffled-quotient control passed/failed: not_run")
    print("- random-residual control passed/failed: not_run")
    print("\nFinal interpretation:")
    print("- diagnostic-only; correction implementation or merge rerun path missing")
    print("- what blocked success: no audited quotient-correction-to-map adapter")
    print("- recommended next experiment: implement corrected C2M3 map adapter and rerun from checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
