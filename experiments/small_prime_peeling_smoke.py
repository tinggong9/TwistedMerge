#!/usr/bin/env python
"""Two-setting smoke test for small-prime holonomy peeling.

This script is intentionally diagnostic.  It checks whether small primes divide
the observed holonomy order/exponent and only reports accuracy when an existing
prediction-level branch row is present.
"""

from __future__ import annotations

import argparse
import json
import math
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
    "alignment_source",
    "alignment_noise_fraction",
    "method",
    "val_accuracy",
    "val_loss",
    "test_accuracy",
    "test_loss",
    "uses_validation_data",
    "is_single_model",
    "is_soup",
    "is_ensemble_or_extra_capacity",
    "capacity_multiplier",
    "inference_multiplier",
    "parameter_count_multiplier",
    "inference_time_multiplier",
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
    "prime",
    "prime_index",
    "p_adic_multiplicity",
    "eligible",
    "skip_reason",
    "remaining_order_before",
    "remaining_order_after",
    "candidate_method",
    "implemented_real_lift",
    "accuracy_status",
    "validation_accuracy",
    "test_accuracy",
    "best_fallback_validation_accuracy",
    "best_fallback_test_accuracy",
    "greedy_soup_validation_accuracy",
    "greedy_soup_test_accuracy",
    "c2m3_validation_accuracy",
    "c2m3_test_accuracy",
    "monomial_validation_accuracy",
    "monomial_test_accuracy",
    "random_same_branch_control_validation_accuracy",
    "random_same_branch_control_test_accuracy",
    "wrong_prime_control_validation_accuracy",
    "wrong_prime_control_test_accuracy",
    "validation_delta_vs_best_fallback",
    "validation_delta_vs_random_same_branch_control",
    "validation_delta_vs_wrong_prime_control",
    "test_delta_vs_best_fallback",
    "capacity_multiplier",
    "inference_multiplier",
    "uses_test_for_selection",
    "selected_by_validation",
    "claim_status",
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


def peel_prime_once_full_multiplicity(remaining_order: int, prime: int) -> dict:
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
        "skip_reason": "" if eligible else "p_not_dividing_remaining_order",
        "remaining_order_before": int(before),
        "remaining_order_after": int(after),
    }


def prime_peeling_plan(primary_source_order: int, primes: Iterable[int]) -> list[dict]:
    remaining = int(max(1, primary_source_order))
    rows = []
    for idx, prime in enumerate(primes):
        row = peel_prime_once_full_multiplicity(remaining, int(prime))
        row["prime_index"] = int(idx)
        rows.append(row)
        remaining = int(row["remaining_order_after"])
    return rows


def factor_axis_label(value: int) -> str:
    n = int(value)
    if n in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}:
        return f"C{n}_prime_axis"
    if n == 6:
        return "mixed_2_plus_3_not_primary_axis"
    return "non_prime_or_unsupported_axis"


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def command_text(argv: list[str]) -> str:
    parts = [".venv/bin/python", "experiments/small_prime_peeling_smoke.py", *argv]
    return " ".join(parts)


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
        raise FileNotFoundError(path)
    runs = pd.read_csv(path, usecols=lambda col: col in RUN_COLUMNS)
    for col in [
        "val_accuracy",
        "val_loss",
        "test_accuracy",
        "test_loss",
        "capacity_multiplier",
        "inference_multiplier",
        "parameter_count_multiplier",
        "inference_time_multiplier",
    ]:
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
    primary_source_order = int(group_exponent) if group_exponent else int(observed_lcm)
    source = "group_exponent_if_exact" if group_exponent else "observed_holonomy_order_lcm"
    relation_count = int(len(relations))
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
        relation_count=relation_count,
        relation_count_status=relation_count_status(relation_count, 4),
        observed_holonomy_order_lcm=int(observed_lcm),
        group_closure_status=str(summary.group_status),
        group_exponent_if_exact=int(group_exponent) if group_exponent else None,
        primary_source_order=int(primary_source_order),
        primary_source_order_source=source,
    )


def run_has_branch_rows(run_rows: pd.DataFrame, run_id: str) -> bool:
    methods = set(run_rows[run_rows["run_id"].astype(str).eq(str(run_id))]["method"].astype(str))
    return bool({"twisted_rank_lift_2", "random_branch_ensemble_2", "validation_branch_ensemble_2"} & methods)


def choose_settings(
    maps: pd.DataFrame,
    runs: pd.DataFrame,
    datasets: list[str],
    prefer_n_models: int,
    settings_per_dataset: int,
    allow_underconstrained: bool,
) -> dict[str, pd.DataFrame]:
    selected: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        group = maps[maps["dataset"].astype(str).eq(str(dataset))].copy()
        if group.empty:
            continue
        candidates = []
        for run_id, run_group in group.groupby("run_id", sort=True):
            first = run_group.iloc[0]
            n_models = int(first["n_models"])
            relation_count = int(len(relations_from_group(run_group)))
            status = relation_count_status(relation_count, 4)
            if status == "underconstrained" and not allow_underconstrained and n_models != int(prefer_n_models):
                continue
            width = int(first["width"])
            has_branch = run_has_branch_rows(runs, str(run_id))
            score = (
                0 if n_models == int(prefer_n_models) else 1,
                0 if has_branch else 1,
                0 if width in {64, 128} else 1,
                width,
                str(first.get("domain_shift", "")),
                str(first.get("matching", "")),
                str(run_id),
            )
            candidates.append((score, str(run_id), run_group))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        for _, run_id, run_group in candidates[: int(settings_per_dataset)]:
            selected[str(dataset)] = run_group.copy()
            break
    return selected


def best_method(run_rows: pd.DataFrame, method_names: Iterable[str], contains: bool = False) -> dict | None:
    if run_rows.empty:
        return None
    methods = [str(method) for method in method_names]
    if contains:
        mask = run_rows["method"].astype(str).map(lambda value: any(token in value for token in methods))
    else:
        mask = run_rows["method"].astype(str).isin(methods)
    subset = run_rows[mask].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["val_accuracy", "val_loss"], ascending=[False, True])
    return subset.iloc[0].to_dict()


def best_fallback_row(run_rows: pd.DataFrame) -> dict | None:
    rows = run_rows.copy()
    if "is_ensemble_or_extra_capacity" in rows:
        rows = rows[~rows["is_ensemble_or_extra_capacity"].map(safe_bool)].copy()
    rows = rows[~rows["method"].astype(str).str.contains("rank_lift|branch_ensemble|control", regex=True)].copy()
    if rows.empty:
        return None
    rows = rows.sort_values(["val_accuracy", "val_loss"], ascending=[False, True])
    return rows.iloc[0].to_dict()


def metric(row: dict | None, column: str) -> float:
    if row is None:
        return np.nan
    return safe_float(row.get(column))


def implemented_lift_method_for_prime(prime: int, run_rows: pd.DataFrame) -> tuple[str, dict | None]:
    p = int(prime)
    candidates = []
    if p == 2:
        candidates.append("twisted_rank_lift_2")
    candidates.extend([f"twisted_rank_lift_{p}", f"prime_C{p}_branch_lift", f"rank_lift_{p}"])
    for method in dict.fromkeys(candidates):
        row = best_method(run_rows, [method])
        if row is not None:
            return method, row
    return f"prime_C{p}_branch_lift", None


def selection_decision(row: dict) -> tuple[bool, str]:
    if bool(row.get("uses_test_for_selection", False)):
        return False, "blocked_test_metric_selection_forbidden"
    if not bool(row.get("eligible", False)):
        return False, "ineligible_prime_not_selected"
    if str(row.get("relation_count_status", "")) == "underconstrained":
        return False, "underconstrained_diagnostic_only"
    if not bool(row.get("implemented_real_lift", False)):
        return False, "diagnostic_only_no_real_prediction"
    val = safe_float(row.get("validation_accuracy"))
    fallback = safe_float(row.get("best_fallback_validation_accuracy"))
    random_control = safe_float(row.get("random_same_branch_control_validation_accuracy"))
    wrong_control = safe_float(row.get("wrong_prime_control_validation_accuracy"))
    if not np.isfinite(val):
        return False, "missing_lift_validation_metric"
    if not np.isfinite(fallback) or val <= fallback:
        return False, "not_selected_fails_best_fallback_gate"
    if not np.isfinite(random_control):
        return False, "not_selected_missing_random_same_branch_control"
    if val <= random_control:
        return False, "not_selected_fails_random_same_branch_control"
    if not np.isfinite(wrong_control):
        return False, "not_selected_missing_wrong_prime_control"
    if val <= wrong_control:
        return False, "not_selected_fails_wrong_prime_control"
    return True, "validation_selected_smoke_hypothesis_generating"


def build_prime_rows(setting: SelectedSetting, run_rows: pd.DataFrame, primes: list[int]) -> list[dict]:
    fallback = best_fallback_row(run_rows)
    greedy = best_method(run_rows, ["greedy_soup"])
    c2m3 = best_method(run_rows, ["c2m3_synchronized", "c2m3_permutation"])
    monomial = best_method(run_rows, ["monomial"], contains=True)
    rows = []
    for peel in prime_peeling_plan(setting.primary_source_order, primes):
        prime = int(peel["prime"])
        lift_method, lift = implemented_lift_method_for_prime(prime, run_rows)
        random_control = best_method(run_rows, [f"random_branch_ensemble_{prime}", f"random_same_branch_count_control_{prime}"])
        wrong_control = best_method(run_rows, [f"validation_branch_ensemble_{prime}", f"wrong_prime_control_{prime}"])
        implemented = bool(peel["eligible"] and lift is not None)
        accuracy_status = "real_prediction_metric_found" if implemented else "diagnostic_only_no_real_prediction"
        row = {
            "dataset": setting.dataset,
            "run_id": setting.run_id,
            "setting_id": setting.setting_id,
            "architecture": setting.architecture,
            "n_models": setting.n_models,
            "width": setting.width,
            "matching": setting.matching,
            "domain_shift": setting.domain_shift,
            "relation_count": setting.relation_count,
            "relation_count_status": setting.relation_count_status,
            "observed_holonomy_order_lcm": setting.observed_holonomy_order_lcm,
            "group_closure_status": setting.group_closure_status,
            "group_exponent_if_exact": setting.group_exponent_if_exact if setting.group_exponent_if_exact else np.nan,
            "primary_source_order": setting.primary_source_order,
            "primary_source_order_source": setting.primary_source_order_source,
            "candidate_method": lift_method,
            "implemented_real_lift": implemented,
            "accuracy_status": accuracy_status,
            "validation_accuracy": metric(lift, "val_accuracy") if implemented else np.nan,
            "test_accuracy": metric(lift, "test_accuracy") if implemented else np.nan,
            "best_fallback_method": fallback.get("method") if fallback else "",
            "best_fallback_validation_accuracy": metric(fallback, "val_accuracy"),
            "best_fallback_test_accuracy": metric(fallback, "test_accuracy"),
            "greedy_soup_validation_accuracy": metric(greedy, "val_accuracy"),
            "greedy_soup_test_accuracy": metric(greedy, "test_accuracy"),
            "c2m3_validation_accuracy": metric(c2m3, "val_accuracy"),
            "c2m3_test_accuracy": metric(c2m3, "test_accuracy"),
            "monomial_validation_accuracy": metric(monomial, "val_accuracy"),
            "monomial_test_accuracy": metric(monomial, "test_accuracy"),
            "random_same_branch_control_validation_accuracy": metric(random_control, "val_accuracy"),
            "random_same_branch_control_test_accuracy": metric(random_control, "test_accuracy"),
            "wrong_prime_control_validation_accuracy": metric(wrong_control, "val_accuracy"),
            "wrong_prime_control_test_accuracy": metric(wrong_control, "test_accuracy"),
            "capacity_multiplier": float(prime),
            "inference_multiplier": float(prime),
            "uses_test_for_selection": False,
            **peel,
        }
        row["validation_delta_vs_best_fallback"] = row["validation_accuracy"] - row["best_fallback_validation_accuracy"]
        row["validation_delta_vs_random_same_branch_control"] = (
            row["validation_accuracy"] - row["random_same_branch_control_validation_accuracy"]
        )
        row["validation_delta_vs_wrong_prime_control"] = row["validation_accuracy"] - row["wrong_prime_control_validation_accuracy"]
        row["test_delta_vs_best_fallback"] = row["test_accuracy"] - row["best_fallback_test_accuracy"]
        selected, status = selection_decision(row)
        row["selected_by_validation"] = bool(selected)
        if not row["eligible"]:
            status = "ineligible_prime_not_selected"
        row["claim_status"] = status
        rows.append(row)
    return rows


def paired_stats(rows: pd.DataFrame) -> pd.DataFrame:
    stats = []
    if rows.empty:
        return pd.DataFrame()
    comparisons = [
        ("validation_delta_vs_best_fallback", "validation_delta_vs_best_fallback"),
        ("validation_delta_vs_random_same_branch_control", "validation_delta_vs_random_same_branch_control"),
        ("validation_delta_vs_wrong_prime_control", "validation_delta_vs_wrong_prime_control"),
        ("test_delta_vs_best_fallback", "test_delta_vs_best_fallback"),
    ]
    for prime, group in rows.groupby("prime", sort=True):
        for label, column in comparisons:
            vals = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            stats.append(
                {
                    "prime": int(prime),
                    "comparison": label,
                    "n_rows": int(len(group)),
                    "n_finite": int(vals.size),
                    "mean_delta": float(vals.mean()) if vals.size else np.nan,
                    "min_delta": float(vals.min()) if vals.size else np.nan,
                    "max_delta": float(vals.max()) if vals.size else np.nan,
                    "implemented_real_lift_rows": int(group["implemented_real_lift"].fillna(False).sum()),
                    "selected_by_validation_rows": int(group["selected_by_validation"].fillna(False).sum()),
                    "claim_status": "smoke_descriptive_only",
                }
            )
    return pd.DataFrame(stats)


def selected_settings_frame(settings: list[SelectedSetting], primes: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **setting.__dict__,
                "prime_list_used": ",".join(str(p) for p in primes),
            }
            for setting in settings
        ]
    )


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if not np.isfinite(value) else f"{value:.6g}"
            vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(args: argparse.Namespace, settings: list[SelectedSetting], rows: pd.DataFrame, stats: pd.DataFrame) -> None:
    report_path = args.reports_dir / "small_prime_peeling_smoke_report.md"
    selected = selected_settings_frame(settings, parse_csv(args.prime_list, int))
    implemented = rows[rows["implemented_real_lift"].fillna(False)].copy()
    selected_rows = rows[rows["selected_by_validation"].fillna(False)].copy()
    eligible = (
        rows[rows["eligible"].fillna(False)]
        .groupby("dataset")["prime"]
        .apply(lambda vals: ",".join(str(int(v)) for v in vals))
        .to_dict()
    )
    text = f"""# Small-Prime Peeling Smoke Test

Generated by `experiments/small_prime_peeling_smoke.py`.

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
- Large-prime branch lifts are extra-capacity unless controlled by same-branch-count baselines.
- Do not claim real Brauer/projective or period-index structure from this test.
- Do not claim broad real model-merging improvement from this test.
- Prime convention: a prime `p` is eligible only when `p` divides the observed order/exponent source; this report never says the group order divides `p`.

## Prime List

`{args.prime_list}`

## Selected Settings

{md_table(selected, ["dataset", "run_id", "setting_id", "n_models", "width", "matching", "relation_count", "relation_count_status", "observed_holonomy_order_lcm", "group_closure_status", "primary_source_order"])}

## Prime Peeling Rows

{md_table(rows, ["dataset", "run_id", "prime", "p_adic_multiplicity", "eligible", "remaining_order_before", "remaining_order_after", "candidate_method", "implemented_real_lift", "selected_by_validation", "claim_status"], max_rows=40)}

## Accuracy Deltas

{md_table(rows, ["dataset", "prime", "validation_accuracy", "test_accuracy", "best_fallback_validation_accuracy", "random_same_branch_control_validation_accuracy", "wrong_prime_control_validation_accuracy", "validation_delta_vs_best_fallback", "test_delta_vs_best_fallback", "capacity_multiplier"], max_rows=40)}

## Paired Stats

{md_table(stats, ["prime", "comparison", "n_rows", "n_finite", "mean_delta", "implemented_real_lift_rows", "selected_by_validation_rows", "claim_status"], max_rows=80)}

## Final Interpretation

- Selected MNIST run IDs: `{", ".join(selected[selected["dataset"].eq("mnist")]["run_id"].astype(str).tolist()) or "none"}`
- Selected Fashion-MNIST run IDs: `{", ".join(selected[selected["dataset"].eq("fashion_mnist")]["run_id"].astype(str).tolist()) or "none"}`
- Eligible primes by dataset: `{json.dumps(eligible, sort_keys=True)}`
- Implemented real lift rows: `{len(implemented)}`
- Validation-selected prime lift rows: `{len(selected_rows)}`
- p=2 implemented real lift: `{"yes" if bool((rows["prime"].eq(2) & rows["implemented_real_lift"]).any()) else "no"}`
- p=3 implemented real lift: `{"yes" if bool((rows["prime"].eq(3) & rows["implemented_real_lift"]).any()) else "no"}`

This smoke test only checks whether small prime factors of the observed holonomy order/exponent line up with existing prediction-level branch rows. Unimplemented prime candidates are diagnostic-only and cannot be selected.
"""
    report_path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="4,3")
    parser.add_argument("--settings-per-dataset", type=int, default=1)
    parser.add_argument("--prime-list", default="2,3,5,7,11,13,17,23,29,31")
    parser.add_argument("--prefer-n-models", type=int, default=4)
    parser.add_argument("--allow-underconstrained", action="store_true")
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
    model_counts = set(parse_csv(args.model_counts, int))
    primes = parse_csv(args.prime_list, int)
    maps = load_triangle_maps(args.reports_dir, set(datasets), model_counts)
    runs = load_run_metrics(args.reports_dir)
    chosen = choose_settings(
        maps,
        runs,
        datasets=datasets,
        prefer_n_models=args.prefer_n_models,
        settings_per_dataset=args.settings_per_dataset,
        allow_underconstrained=args.allow_underconstrained,
    )
    if len(chosen) < len(datasets):
        missing = sorted(set(datasets) - set(chosen))
        raise RuntimeError(f"could not select one setting for datasets: {missing}")

    settings = []
    row_dicts = []
    for dataset in datasets:
        group = chosen[dataset]
        setting = summarize_relation_set(group, args.max_group_order, args.max_generators, args.max_exact_order)
        settings.append(setting)
        run_rows = runs[runs["run_id"].astype(str).eq(setting.run_id)].copy()
        row_dicts.extend(build_prime_rows(setting, run_rows, primes))

    rows = pd.DataFrame(row_dicts)
    for column in MAIN_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
    rows = rows[MAIN_COLUMNS + [c for c in rows.columns if c not in MAIN_COLUMNS]].copy()
    stats = paired_stats(rows)
    selected_settings = selected_settings_frame(settings, primes)

    rows.to_csv(args.reports_dir / "csv" / "small_prime_peeling_smoke.csv", index=False, lineterminator="\n")
    stats.to_csv(args.reports_dir / "csv" / "small_prime_peeling_smoke_paired_stats.csv", index=False, lineterminator="\n")
    selected_settings.to_csv(
        args.reports_dir / "csv" / "small_prime_peeling_smoke_selected_settings.csv",
        index=False,
        lineterminator="\n",
    )
    write_report(args, settings, rows, stats)

    eligible = rows[rows["eligible"].fillna(False)].groupby("dataset")["prime"].apply(lambda vals: list(map(int, vals))).to_dict()
    print("Selected settings:")
    for setting in settings:
        print(f"- {setting.dataset}: {setting.run_id} (N={setting.n_models}, W={setting.width}, {setting.matching})")
    print()
    print("Eligible primes:")
    for dataset in datasets:
        print(f"- {dataset}: {eligible.get(dataset, [])}")
    print()
    print("Implemented real lifts:")
    for prime in [2, 3]:
        ok = bool((rows["prime"].eq(prime) & rows["implemented_real_lift"]).any())
        print(f"- p={prime}: {'yes' if ok else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
