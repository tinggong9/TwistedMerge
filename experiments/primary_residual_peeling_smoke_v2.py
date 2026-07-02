#!/usr/bin/env python
"""Real no-lift p-primary residual peeling smoke test.

This v2 smoke test closes the gap left by the diagnostic-only v1 script.  It
turns certified quotient edge assignments into valid permutation corrections,
reruns same-capacity C2M3-style synchronization on the corrected maps, and
evaluates validation/test accuracy on the fixed-setting loaders.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_merging_benchmark import (  # noqa: E402
    DomainShiftDataset,
    average_models,
    compose_perm,
    device_from_arg,
    evaluate_model,
    invert_perm,
    load_dataset,
    make_loader,
    make_model,
    permutation_disagreement,
    permute_model_to_reference,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.nonabelian_holonomy import infer_holonomy_group  # noqa: E402
from src.primary_holonomy import (  # noqa: E402
    fit_primary_quotient,
    observed_holonomy_order_lcm,
    p_adic_valuation,
    relation_count_status,
    triangle_relation_from_perms,
)


PREFERRED_RUN_IDS = {
    "mnist": "mnist_mlp_N4_W64_input_noise_monomial_activation_seed4200",
    "fashion_mnist": "fashion_mnist_mlp_N4_W64_input_noise_monomial_activation_seed4200",
}

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
    "epochs",
    "max_train_samples",
    "max_test_samples",
    "batch_size",
    "lr",
    "optimizer",
    "weight_decay",
    "scheduler",
    "step_size",
    "gamma",
    "augmentation",
    "dataset_seed",
    "val_fraction",
    "method",
    "val_accuracy",
    "test_accuracy",
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
    "representative_correction_status",
    "quotient_residual_before",
    "quotient_residual_after",
    "quotient_residual_reduction",
    "permutation_cycle_residual_before",
    "permutation_cycle_residual_after",
    "permutation_cycle_residual_reduction",
    "edge_cochain_solve_status",
    "representative_selection_status",
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

CORRECTED_MAP_COLUMNS = [
    "dataset",
    "run_id",
    "prime",
    "peel_mode",
    "edge",
    "quotient_edge_label",
    "representative_label",
    "representative_generator_label",
    "representative_disagreement_from_identity",
    "edge_cochain_solve_status",
    "representative_selection_status",
    "original_map",
    "correction_map",
    "corrected_map",
    "map_valid",
    "map_type",
    "quotient_residual_before",
    "quotient_residual_after",
    "permutation_cycle_residual_before",
    "permutation_cycle_residual_after",
    "residual_reduction",
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
    model_source: str


@dataclass(frozen=True)
class EdgeCochainSolution:
    solved_exact: bool
    edge_labels: dict[tuple[int, int], int]
    quotient_residual_before: float
    quotient_residual_after: float
    n_equations: int
    n_variables: int
    rank: int
    solve_status: str
    sign: int


@dataclass(frozen=True)
class RepresentativeChoice:
    label: int
    representative: np.ndarray | None
    representative_label: int | None
    generator_label: int | None
    disagreement_from_identity: float
    status: str


@dataclass(frozen=True)
class CorrectionResult:
    corrections: dict[tuple[int, int], np.ndarray]
    corrected: dict[tuple[int, int], np.ndarray]
    edge_labels: dict[tuple[int, int], int]
    representative_choices: dict[tuple[int, int], RepresentativeChoice]
    solution: EdgeCochainSolution
    edge_cochain_solve_status: str
    representative_selection_status: str
    implemented: bool
    inverse_consistency_ok: bool


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def safe_float(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def is_finite(value) -> bool:
    return math.isfinite(safe_float(value))


def safe_perm(value) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = tuple(int(item) for item in value)
    elif isinstance(value, str) and value.strip() and value != "nan":
        try:
            arr = tuple(int(item) for item in json.loads(value))
        except Exception:
            return None
    else:
        return None
    return arr if is_valid_permutation(arr) else None


def is_valid_permutation(perm: Iterable[int]) -> bool:
    values = [int(item) for item in perm]
    return bool(values) and sorted(values) == list(range(len(values)))


def permutation_json(perm: np.ndarray) -> str:
    return json.dumps(np.asarray(perm, dtype=int).tolist(), separators=(",", ":"))


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
            cumulative.append(int(prime))
        row["cumulative_primes"] = ",".join(str(item) for item in cumulative)
        rows.append(row)
        remaining = int(row["remaining_order_after"])
    return rows


def split_indices(n_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    torch, _, _ = require_torch()
    n_val = max(1, int(round(n_items * float(val_fraction))))
    n_train = max(1, int(n_items) - n_val)
    if n_train + n_val > int(n_items):
        n_val = int(n_items) - n_train
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(int(n_items), generator=generator).tolist()
    return indices[:n_train], indices[n_train : n_train + n_val]


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def command_text(argv: list[str]) -> str:
    return " ".join([".venv/bin/python", "experiments/primary_residual_peeling_smoke_v2.py", *argv])


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
    numeric_cols = [
        "n_models",
        "width",
        "seed",
        "epochs",
        "max_train_samples",
        "max_test_samples",
        "batch_size",
        "lr",
        "weight_decay",
        "step_size",
        "gamma",
        "dataset_seed",
        "val_fraction",
        "val_accuracy",
        "test_accuracy",
    ]
    for col in numeric_cols:
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


def checkpoint_paths(reports_dir: Path, setting_id: str, seed: int, n_models: int) -> list[Path]:
    base = reports_dir / "checkpoints" / "fixed_setting_verification" / setting_id
    return [base / f"seed{int(seed)}_model{idx}.pt" for idx in range(int(n_models))]


def checkpoints_available(reports_dir: Path, setting_id: str, seed: int, n_models: int) -> bool:
    return all(path.exists() for path in checkpoint_paths(reports_dir, setting_id, seed, n_models))


def summarize_relation_set(group: pd.DataFrame, reports_dir: Path, max_group_order: int, max_generators: int, max_exact_order: int) -> SelectedSetting:
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
    setting_id = str(first["setting_id"])
    seed = int(first["seed"])
    n_models = int(first["n_models"])
    source = (
        "checkpoint"
        if checkpoints_available(reports_dir, setting_id, seed, n_models)
        else "deterministic_retrain_from_fixed_setting_metadata"
    )
    return SelectedSetting(
        dataset=str(first["dataset"]),
        run_id=str(first["run_id"]),
        setting_id=setting_id,
        architecture=str(first.get("architecture", "")),
        n_models=n_models,
        width=int(first["width"]),
        domain_shift=str(first.get("domain_shift", "")),
        matching=str(first.get("matching", "")),
        seed=seed,
        relation_count=int(len(relations)),
        relation_count_status=relation_count_status(len(relations), 4),
        observed_holonomy_order_lcm=int(observed_lcm),
        group_closure_status=str(summary.group_status),
        group_exponent_if_exact=int(group_exponent) if group_exponent else None,
        primary_source_order=int(group_exponent) if group_exponent else int(observed_lcm),
        primary_source_order_source="group_exponent_if_exact" if group_exponent else "observed_holonomy_order_lcm",
        model_source=source,
    )


def choose_settings(
    maps: pd.DataFrame,
    runs: pd.DataFrame,
    datasets: list[str],
    prefer_n_models: int,
    settings_per_dataset: int,
) -> dict[str, pd.DataFrame]:
    selected = {}
    for dataset in datasets:
        group = maps[maps["dataset"].astype(str).eq(str(dataset))].copy()
        if group.empty:
            continue
        preferred = PREFERRED_RUN_IDS.get(str(dataset))
        if preferred and preferred in set(group["run_id"].astype(str)) and preferred in set(runs["run_id"].astype(str)):
            selected[str(dataset)] = group[group["run_id"].astype(str).eq(preferred)].copy()
            continue
        candidates = []
        for run_id, run_group in group.groupby("run_id", sort=True):
            if str(run_id) not in set(runs["run_id"].astype(str)):
                continue
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


def run_hyperparams(runs: pd.DataFrame, run_id: str) -> dict:
    rows = runs[runs["run_id"].astype(str).eq(str(run_id))].copy()
    if rows.empty:
        raise RuntimeError(f"missing fixed-setting run metadata for {run_id}")
    first = rows.iloc[0].to_dict()
    return {
        "epochs": int(first.get("epochs", 10)),
        "max_train_samples": int(first.get("max_train_samples", 10000)),
        "max_test_samples": int(first.get("max_test_samples", 2000)),
        "batch_size": int(first.get("batch_size", 128)),
        "lr": safe_float(first.get("lr"), 0.001),
        "optimizer": str(first.get("optimizer", "adamw")),
        "weight_decay": safe_float(first.get("weight_decay"), 0.0001),
        "scheduler": str(first.get("scheduler", "cosine")),
        "step_size": int(first.get("step_size", 3)),
        "gamma": safe_float(first.get("gamma"), 0.5),
        "augmentation": str(first.get("augmentation", "none")),
        "dataset_seed": int(first.get("dataset_seed", 314159)),
        "val_fraction": safe_float(first.get("val_fraction"), 0.2),
    }


def build_loaders_and_models(setting: SelectedSetting, runs: pd.DataFrame, reports_dir: Path, data_dir: Path, device_arg: str):
    torch, _, _ = require_torch()
    hparams = run_hyperparams(runs, setting.run_id)
    device = device_from_arg(device_arg)
    spec, train_base, test_base = load_dataset(
        setting.dataset,
        data_dir,
        hparams["max_train_samples"],
        hparams["max_test_samples"],
        hparams["dataset_seed"],
        augmentation=hparams["augmentation"],
    )
    train_indices, val_indices = split_indices(len(train_base), hparams["val_fraction"], hparams["dataset_seed"] + 17)
    val_subset = torch.utils.data.Subset(train_base, val_indices)
    val_loader = make_loader(val_subset, hparams["batch_size"], shuffle=False, seed=hparams["dataset_seed"] + 100)
    test_loader = make_loader(test_base, hparams["batch_size"], shuffle=False, seed=hparams["dataset_seed"] + 200)

    models = []
    paths = checkpoint_paths(reports_dir, setting.setting_id, setting.seed, setting.n_models)
    if all(path.exists() for path in paths):
        for path in paths:
            payload = torch.load(path, map_location="cpu")
            model = make_model(setting.architecture, spec, setting.width)
            model.load_state_dict(payload["state_dict"])
            model.to("cpu")
            models.append(model)
        return SimpleNamespace(
            spec=spec,
            val_loader=val_loader,
            test_loader=test_loader,
            models=models,
            hparams=hparams,
            model_source="checkpoint",
            device=device,
        )

    for model_idx in range(setting.n_models):
        local_seed = int(setting.seed) + 1009 * model_idx + 37 * int(setting.width) + 101 * int(setting.n_models)
        set_seed(local_seed)
        shifted_train = DomainShiftDataset(train_base, setting.domain_shift, model_idx, setting.n_models)
        train_subset = torch.utils.data.Subset(shifted_train, train_indices)
        train_loader = make_loader(train_subset, hparams["batch_size"], shuffle=True, seed=local_seed + 1)
        model = make_model(setting.architecture, spec, setting.width)
        train_model(
            model,
            train_loader,
            hparams["epochs"],
            hparams["lr"],
            device,
            optimizer=hparams["optimizer"],
            weight_decay=hparams["weight_decay"],
            scheduler=hparams["scheduler"],
            step_size=hparams["step_size"],
            gamma=hparams["gamma"],
        )
        model.to("cpu")
        models.append(model)
    return SimpleNamespace(
        spec=spec,
        val_loader=val_loader,
        test_loader=test_loader,
        models=models,
        hparams=hparams,
        model_source="deterministic_retrain_from_fixed_setting_metadata",
        device=device,
    )


def reconstruct_pairwise_perms(group: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    n_models = int(group.iloc[0]["n_models"])
    sample = safe_perm(group.iloc[0]["p_ij"])
    if sample is None:
        raise RuntimeError("triangle artifact is missing valid p_ij permutation")
    width = len(sample)
    pairwise: dict[tuple[int, int], np.ndarray] = {(idx, idx): np.arange(width, dtype=int) for idx in range(n_models)}
    for _, row in group.iterrows():
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        for a, b, col in ((i, j, "p_ij"), (j, k, "p_jk"), (k, i, "p_ki")):
            perm = safe_perm(row.get(col))
            if perm is None:
                continue
            pairwise[(a, b)] = np.asarray(perm, dtype=int)
    for i, j in product(range(n_models), repeat=2):
        if (i, j) not in pairwise and (j, i) in pairwise:
            pairwise[(i, j)] = invert_perm(pairwise[(j, i)])
    missing = [(i, j) for i, j in product(range(n_models), repeat=2) if (i, j) not in pairwise]
    if missing:
        raise RuntimeError(f"missing pairwise permutation artifacts for edges {missing}")
    return pairwise


def cycle_residual(pairwise: dict[tuple[int, int], np.ndarray], n_models: int) -> float:
    if not pairwise:
        return float("nan")
    width = len(next(iter(pairwise.values())))
    identity = np.arange(width, dtype=int)
    residuals = []
    for i, j, k in combinations(range(int(n_models)), 3):
        defect = compose_perm(compose_perm(pairwise[(i, j)], pairwise[(j, k)]), pairwise[(k, i)])
        residuals.append(permutation_disagreement(defect, identity))
    return float(np.mean(residuals)) if residuals else 0.0


def permutation_power(perm: Iterable[int], exponent: int) -> np.ndarray:
    base = np.asarray(tuple(int(item) for item in perm), dtype=int)
    width = len(base)
    identity = np.arange(width, dtype=int)
    exp = int(exponent)
    if exp == 0:
        return identity.copy()
    if exp < 0:
        base = invert_perm(base)
        exp = -exp
    out = identity.copy()
    for _ in range(exp):
        out = compose_perm(out, base)
    return out


def triangle_defects_from_pairwise(pairwise: dict[tuple[int, int], np.ndarray], n_models: int) -> dict[tuple[int, int, int], np.ndarray]:
    defects: dict[tuple[int, int, int], np.ndarray] = {}
    for i, j, k in combinations(range(int(n_models)), 3):
        defects[(i, j, k)] = compose_perm(compose_perm(pairwise[(i, j)], pairwise[(j, k)]), pairwise[(k, i)])
    return defects


def relations_from_pairwise(pairwise: dict[tuple[int, int], np.ndarray], n_models: int) -> tuple:
    relations = []
    for (i, j, k), hol in triangle_defects_from_pairwise(pairwise, n_models).items():
        relations.append(triangle_relation_from_perms(pairwise[(i, j)], pairwise[(j, k)], pairwise[(k, i)], hol))
    return tuple(relations)


def quotient_label_from_fit(perm: Iterable[int], fit, p: int) -> int:
    if fit is None:
        return 0
    key = tuple(int(item) for item in perm)
    return int(fit.assignment.get(key, 0)) % int(p)


def quotient_defect_labels_from_pairwise(
    pairwise: dict[tuple[int, int], np.ndarray],
    fit,
    n_models: int,
    prime: int,
) -> dict[tuple[int, int, int], int]:
    defects = triangle_defects_from_pairwise(pairwise, n_models)
    return {tri: quotient_label_from_fit(defect, fit, prime) for tri, defect in defects.items()}


def quotient_residual_from_labels(labels: dict[tuple[int, int, int], int], p: int) -> float:
    if not labels:
        return 0.0
    return float(np.mean([int(value) % int(p) != 0 for value in labels.values()]))


def _edge_variable_index(n_models: int) -> dict[tuple[int, int], int]:
    return {edge: idx for idx, edge in enumerate(combinations(range(int(n_models)), 2))}


def _oriented_edge_value(edge_labels: dict[tuple[int, int], int], i: int, j: int, p: int) -> int:
    if i == j:
        return 0
    if i < j:
        return int(edge_labels.get((i, j), 0)) % int(p)
    return (-int(edge_labels.get((j, i), 0))) % int(p)


def _rref_solve_mod_p(matrix: list[list[int]], rhs: list[int], p: int) -> tuple[bool, list[int], int]:
    p = int(p)
    if not matrix:
        return True, [], 0
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix[0] else 0
    aug = [[int(value) % p for value in row] + [int(rhs[idx]) % p] for idx, row in enumerate(matrix)]
    row = 0
    pivots: list[int] = []
    for col in range(n_cols):
        pivot = next((candidate for candidate in range(row, n_rows) if aug[candidate][col] % p), None)
        if pivot is None:
            continue
        aug[row], aug[pivot] = aug[pivot], aug[row]
        inv = pow(int(aug[row][col]) % p, -1, p)
        aug[row] = [(value * inv) % p for value in aug[row]]
        for other in range(n_rows):
            if other == row:
                continue
            factor = aug[other][col] % p
            if factor:
                aug[other] = [(aug[other][idx] - factor * aug[row][idx]) % p for idx in range(n_cols + 1)]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    inconsistent = any(all(aug[r][c] % p == 0 for c in range(n_cols)) and aug[r][-1] % p != 0 for r in range(n_rows))
    solution = [0 for _ in range(n_cols)]
    if not inconsistent:
        for pivot_row, pivot_col in enumerate(pivots):
            solution[pivot_col] = int(aug[pivot_row][-1]) % p
    return not inconsistent, solution, len(pivots)


def solve_edge_cochain_mod_p(
    triangle_defect_labels: dict[tuple[int, int, int], int],
    n_models: int,
    p: int,
    sign: int = 1,
) -> EdgeCochainSolution:
    """Solve d(edge_label) = sign * triangle_defect_label over F_p."""

    p = int(p)
    variables = _edge_variable_index(n_models)
    rows: list[list[int]] = []
    rhs: list[int] = []
    for i, j, k in sorted(triangle_defect_labels):
        row = [0 for _ in variables]
        for a, b in ((i, j), (j, k), (k, i)):
            if a == b:
                continue
            if a < b:
                row[variables[(a, b)]] = (row[variables[(a, b)]] + 1) % p
            else:
                row[variables[(b, a)]] = (row[variables[(b, a)]] - 1) % p
        rows.append(row)
        rhs.append((int(sign) * int(triangle_defect_labels[(i, j, k)])) % p)
    exact, vector, rank = _rref_solve_mod_p(rows, rhs, p)
    edge_labels = {edge: int(vector[idx]) % p for edge, idx in variables.items()} if vector else {edge: 0 for edge in variables}
    residuals = []
    for (i, j, k), label in sorted(triangle_defect_labels.items()):
        lhs = (
            _oriented_edge_value(edge_labels, i, j, p)
            + _oriented_edge_value(edge_labels, j, k, p)
            + _oriented_edge_value(edge_labels, k, i, p)
        ) % p
        residuals.append((lhs - int(sign) * int(label)) % p)
    residual_after = float(np.mean([value % p != 0 for value in residuals])) if residuals else 0.0
    residual_before = quotient_residual_from_labels(triangle_defect_labels, p)
    solved_exact = bool(exact and residual_after <= 0.0)
    status = "exact_quotient_cochain_solve" if solved_exact else "quotient_cochain_inconsistent"
    return EdgeCochainSolution(
        solved_exact=solved_exact,
        edge_labels=edge_labels,
        quotient_residual_before=float(residual_before),
        quotient_residual_after=float(residual_after),
        n_equations=int(len(rows)),
        n_variables=int(len(variables)),
        rank=int(rank),
        solve_status=f"{status}_sign_{int(sign)}",
        sign=int(sign),
    )


def solve_best_edge_cochain_mod_p(
    triangle_defect_labels: dict[tuple[int, int, int], int],
    n_models: int,
    p: int,
) -> EdgeCochainSolution:
    candidates = [
        solve_edge_cochain_mod_p(triangle_defect_labels, n_models, p, sign=1),
        solve_edge_cochain_mod_p(triangle_defect_labels, n_models, p, sign=-1),
    ]
    candidates.sort(key=lambda item: (not item.solved_exact, item.quotient_residual_after, 0 if item.sign == 1 else 1))
    return candidates[0]


def representative_for_cp_label(
    label: int,
    fit,
    observed_holonomies: Iterable[np.ndarray],
    width: int,
    p: int,
) -> RepresentativeChoice:
    p = int(p)
    label = int(label) % p
    identity = np.arange(width, dtype=int)
    if label == 0:
        return RepresentativeChoice(
            label=0,
            representative=identity.copy(),
            representative_label=0,
            generator_label=0,
            disagreement_from_identity=0.0,
            status="identity_representative",
        )
    best: RepresentativeChoice | None = None
    for holonomy in observed_holonomies:
        generator = np.asarray(holonomy, dtype=int)
        if not is_valid_permutation(generator):
            continue
        generator_label = quotient_label_from_fit(generator, fit, p)
        if generator_label % p == 0:
            continue
        try:
            inv_label = pow(generator_label, -1, p)
        except ValueError:
            continue
        exponent = (inv_label * label) % p
        representative = permutation_power(generator, exponent)
        rep_label = (generator_label * exponent) % p
        if rep_label != label:
            continue
        disagreement = permutation_disagreement(representative, identity)
        choice = RepresentativeChoice(
            label=label,
            representative=representative,
            representative_label=int(rep_label),
            generator_label=int(generator_label),
            disagreement_from_identity=float(disagreement),
            status="observed_holonomy_power_representative",
        )
        if best is None or choice.disagreement_from_identity < best.disagreement_from_identity:
            best = choice
    if best is None:
        return RepresentativeChoice(
            label=label,
            representative=None,
            representative_label=None,
            generator_label=None,
            disagreement_from_identity=float("nan"),
            status="no_representative_correction_available",
        )
    return best


def apply_edge_label_corrections(
    pairwise: dict[tuple[int, int], np.ndarray],
    representatives: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    return {
        edge: compose_perm(invert_perm(representatives[edge]), np.asarray(pairwise[edge], dtype=int))
        for edge in pairwise
    }


def _inverse_consistency_ok(pairwise: dict[tuple[int, int], np.ndarray], n_models: int) -> bool:
    for i, j in product(range(int(n_models)), repeat=2):
        if not np.array_equal(pairwise[(j, i)], invert_perm(pairwise[(i, j)])):
            return False
    return True


def solve_and_correct_pairwise(
    pairwise: dict[tuple[int, int], np.ndarray],
    fit,
    n_models: int,
    prime: int,
) -> CorrectionResult:
    width = len(next(iter(pairwise.values())))
    identity = np.arange(width, dtype=int)
    defects = triangle_defects_from_pairwise(pairwise, n_models)
    labels = {tri: quotient_label_from_fit(defect, fit, prime) for tri, defect in defects.items()}
    solution = solve_best_edge_cochain_mod_p(labels, n_models, prime)
    edge_labels: dict[tuple[int, int], int] = {}
    corrections = {(idx, idx): identity.copy() for idx in range(int(n_models))}
    representative_choices: dict[tuple[int, int], RepresentativeChoice] = {}
    for i, j in product(range(int(n_models)), repeat=2):
        if i == j:
            edge_labels[(i, j)] = 0
            representative_choices[(i, j)] = RepresentativeChoice(
                label=0,
                representative=identity.copy(),
                representative_label=0,
                generator_label=0,
                disagreement_from_identity=0.0,
                status="identity_representative",
            )
            continue
        edge_labels[(i, j)] = _oriented_edge_value(solution.edge_labels, i, j, prime)
    if not solution.solved_exact:
        corrected = {edge: value.copy() for edge, value in pairwise.items()}
        for i, j in product(range(int(n_models)), repeat=2):
            corrections[(i, j)] = identity.copy()
            representative_choices.setdefault(
                (i, j),
                RepresentativeChoice(
                    label=edge_labels.get((i, j), 0),
                    representative=identity.copy(),
                    representative_label=0,
                    generator_label=0,
                    disagreement_from_identity=0.0,
                    status=solution.solve_status,
                ),
            )
        return CorrectionResult(
            corrections=corrections,
            corrected=corrected,
            edge_labels=edge_labels,
            representative_choices=representative_choices,
            solution=solution,
            edge_cochain_solve_status=solution.solve_status,
            representative_selection_status="quotient_cochain_inconsistent",
            implemented=False,
            inverse_consistency_ok=_inverse_consistency_ok(corrected, n_models),
        )

    observed_holonomies = list(defects.values())
    representative_statuses = []
    for i, j in product(range(int(n_models)), repeat=2):
        if i == j:
            continue
        choice = representative_for_cp_label(edge_labels[(i, j)], fit, observed_holonomies, width, prime)
        representative_choices[(i, j)] = choice
        representative_statuses.append(choice.status)
        if choice.representative is not None:
            corrections[(i, j)] = choice.representative.copy()
    if any(status == "no_representative_correction_available" for status in representative_statuses):
        corrected = {edge: value.copy() for edge, value in pairwise.items()}
        for i, j in product(range(int(n_models)), repeat=2):
            corrections.setdefault((i, j), identity.copy())
        return CorrectionResult(
            corrections=corrections,
            corrected=corrected,
            edge_labels=edge_labels,
            representative_choices=representative_choices,
            solution=solution,
            edge_cochain_solve_status=solution.solve_status,
            representative_selection_status="no_representative_correction_available",
            implemented=False,
            inverse_consistency_ok=_inverse_consistency_ok(corrected, n_models),
        )
    corrected = apply_edge_label_corrections(pairwise, corrections)
    return CorrectionResult(
        corrections=corrections,
        corrected=corrected,
        edge_labels=edge_labels,
        representative_choices=representative_choices,
        solution=solution,
        edge_cochain_solve_status=solution.solve_status,
        representative_selection_status="representative_correction_available",
        implemented=True,
        inverse_consistency_ok=_inverse_consistency_ok(corrected, n_models),
    )


def shuffled_correction_maps(
    pairwise: dict[tuple[int, int], np.ndarray],
    corrections: dict[tuple[int, int], np.ndarray],
    n_models: int,
    seed: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    width = len(next(iter(pairwise.values())))
    identity = np.arange(width, dtype=int)
    rng = np.random.default_rng(seed)
    directed = [(i, j) for i, j in product(range(int(n_models)), repeat=2) if i != j]
    correction_values = [corrections[edge].copy() for edge in directed]
    rng.shuffle(correction_values)
    shuffled = {(idx, idx): identity.copy() for idx in range(int(n_models))}
    corrected = {(idx, idx): identity.copy() for idx in range(int(n_models))}
    for edge, correction in zip(directed, correction_values):
        shuffled[edge] = correction
        corrected[edge] = compose_perm(invert_perm(correction), pairwise[edge])
    return shuffled, corrected


def random_same_residual_norm_maps(
    pairwise: dict[tuple[int, int], np.ndarray],
    corrections: dict[tuple[int, int], np.ndarray],
    n_models: int,
    seed: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    width = len(next(iter(pairwise.values())))
    identity = np.arange(width, dtype=int)
    rng = np.random.default_rng(seed)
    random_corrections = {(idx, idx): identity.copy() for idx in range(int(n_models))}
    corrected = {(idx, idx): identity.copy() for idx in range(int(n_models))}
    for i, j in product(range(int(n_models)), repeat=2):
        if i == j:
            continue
        moved = int(np.sum(corrections[(i, j)] != identity))
        random_map = identity.copy()
        swaps = max(0, int(round(moved / 2)))
        for _ in range(swaps):
            a, b = rng.choice(width, size=2, replace=False)
            random_map[a], random_map[b] = random_map[b], random_map[a]
        random_corrections[(i, j)] = random_map
        corrected[(i, j)] = compose_perm(invert_perm(random_map), pairwise[(i, j)])
    return random_corrections, corrected


def evaluate_c2m3_from_pairwise(setting: SelectedSetting, bundle, pairwise: dict[tuple[int, int], np.ndarray]) -> dict:
    if setting.architecture != "mlp":
        raise RuntimeError(f"v2 correction adapter currently supports mlp permutation C2M3, got {setting.architecture}")
    ref, synced, sync_disagreement = synchronize_permutations(pairwise, setting.n_models)
    aligned = [
        permute_model_to_reference(
            bundle.models[idx],
            setting.architecture,
            bundle.spec,
            setting.width,
            synced[idx],
        )
        for idx in range(setting.n_models)
    ]
    model = average_models(aligned, setting.architecture, bundle.spec, setting.width)
    val = evaluate_model(model, bundle.val_loader, bundle.device)
    test = evaluate_model(model, bundle.test_loader, bundle.device)
    return {
        "validation_accuracy": float(val["accuracy"]),
        "validation_loss": float(val["loss"]),
        "test_accuracy": float(test["accuracy"]),
        "test_loss": float(test["loss"]),
        "sync_reference": int(ref),
        "sync_disagreement": float(sync_disagreement),
    }


def map_rows_for(
    setting: SelectedSetting,
    prime: int,
    peel_mode: str,
    original: dict[tuple[int, int], np.ndarray],
    corrections: dict[tuple[int, int], np.ndarray],
    corrected: dict[tuple[int, int], np.ndarray],
    quotient_before: float,
    quotient_after: float,
    permutation_before: float,
    permutation_after: float,
    edge_labels: dict[tuple[int, int], int] | None = None,
    representative_choices: dict[tuple[int, int], RepresentativeChoice] | None = None,
    edge_cochain_solve_status: str = "",
    representative_selection_status: str = "",
) -> list[dict]:
    rows = []
    for edge in sorted(original):
        corr = corrections.get(edge)
        out = corrected.get(edge)
        choice = representative_choices.get(edge) if representative_choices else None
        valid = corr is not None and out is not None and is_valid_permutation(corr) and is_valid_permutation(out)
        rows.append(
            {
                "dataset": setting.dataset,
                "run_id": setting.run_id,
                "prime": int(prime),
                "peel_mode": peel_mode,
                "edge": f"{edge[0]}->{edge[1]}",
                "quotient_edge_label": int(edge_labels.get(edge, 0)) if edge_labels else np.nan,
                "representative_label": choice.representative_label if choice else np.nan,
                "representative_generator_label": choice.generator_label if choice else np.nan,
                "representative_disagreement_from_identity": choice.disagreement_from_identity if choice else np.nan,
                "edge_cochain_solve_status": edge_cochain_solve_status,
                "representative_selection_status": choice.status if choice else representative_selection_status,
                "original_map": permutation_json(original[edge]),
                "correction_map": permutation_json(corr if corr is not None else np.asarray([], dtype=int)),
                "corrected_map": permutation_json(out if out is not None else np.asarray([], dtype=int)),
                "map_valid": bool(valid),
                "map_type": "permutation" if valid else "invalid",
                "quotient_residual_before": float(quotient_before),
                "quotient_residual_after": float(quotient_after),
                "permutation_cycle_residual_before": float(permutation_before),
                "permutation_cycle_residual_after": float(permutation_after),
                "residual_reduction": float(quotient_before - quotient_after),
            }
        )
    return rows


def no_lift_capacity_metadata() -> dict:
    return {"capacity_multiplier": 1.0, "inference_multiplier": 1.0}


def make_base_row(
    setting: SelectedSetting,
    peel: dict,
    prime: int,
    peel_mode: str,
    cumulative_primes: str,
    fit,
    before: float,
    after: float,
    reduces: bool,
    quotient_before: float | None = None,
    quotient_after: float | None = None,
    edge_cochain_solve_status: str = "",
    representative_selection_status: str = "",
) -> dict:
    q_before = float(before if quotient_before is None else quotient_before)
    q_after = float(after if quotient_after is None else quotient_after)
    p_before = float(before)
    p_after = float(after)
    return {
        **setting.__dict__,
        **peel,
        "prime": int(prime),
        "peel_mode": peel_mode,
        "cumulative_primes": cumulative_primes,
        "quotient_fit_status": fit.quotient_fit_status if fit is not None else "not_attempted",
        "quotient_relation_violation_rate": float(fit.relation_violation_rate) if fit is not None else float("nan"),
        "edge_correction_status": "edge_correction_found" if reduces else "no_residual_reduction",
        "representative_correction_status": "",
        "quotient_residual_before": q_before,
        "quotient_residual_after": q_after,
        "quotient_residual_reduction": float(q_before - q_after),
        "permutation_cycle_residual_before": p_before,
        "permutation_cycle_residual_after": p_after,
        "permutation_cycle_residual_reduction": float(p_before - p_after),
        "edge_cochain_solve_status": edge_cochain_solve_status,
        "representative_selection_status": representative_selection_status,
        "corrected_cycle_residual_before": p_before,
        "corrected_cycle_residual_after": p_after,
        "correction_reduces_residual": bool(reduces),
        **no_lift_capacity_metadata(),
        "uses_test_for_selection": False,
    }


def row_missing_reason(row: dict, fallback: str = "") -> str:
    if not is_finite(row.get("validation_accuracy")) or not is_finite(row.get("test_accuracy")):
        return fallback or "missing_validation_or_test_metric"
    return ""


def selection_decision(row: dict) -> tuple[bool, str]:
    if bool(row.get("uses_test_for_selection", False)):
        return False, "blocked_test_metric_selection_forbidden"
    if not bool(row.get("eligible", False)):
        return False, "prime_not_eligible"
    if not bool(row.get("implemented_corrected_merge", False)):
        return False, row.get("na_reason") or "corrected_merge_not_implemented"
    if not is_finite(row.get("validation_accuracy")):
        return False, "missing_corrected_validation_metric"
    if not is_finite(row.get("test_accuracy")):
        return False, "missing_corrected_test_metric"
    q_before = safe_float(row.get("quotient_residual_before"))
    q_after = safe_float(row.get("quotient_residual_after"))
    p_before = safe_float(row.get("permutation_cycle_residual_before"))
    p_after = safe_float(row.get("permutation_cycle_residual_after"))
    if not (math.isfinite(q_before) and math.isfinite(q_after) and q_after < q_before):
        return False, "metric_produced_but_not_claimable"
    if math.isfinite(p_before) and math.isfinite(p_after) and p_after > p_before + 1e-12:
        return False, "quotient_peel_not_permutation_safe"
    if safe_float(row.get("capacity_multiplier")) != 1.0 or safe_float(row.get("inference_multiplier")) != 1.0:
        return False, "not_capacity_matched_no_lift"
    val = safe_float(row.get("validation_accuracy"))
    baseline = safe_float(row.get("baseline_validation_accuracy"))
    if not math.isfinite(baseline) or val <= baseline:
        return False, "not_selected_fails_unpeeled_baseline_gate"
    control_cols = [
        ("wrong_prime_control_validation_accuracy", "wrong_prime_control"),
        ("shuffled_control_validation_accuracy", "shuffled_control"),
        ("random_residual_control_validation_accuracy", "random_residual_control"),
    ]
    incomplete = []
    for col, label in control_cols:
        control_val = safe_float(row.get(col))
        if not math.isfinite(control_val):
            incomplete.append(label)
        elif val <= control_val:
            return False, f"not_selected_fails_{label}"
    if incomplete:
        return False, "accuracy_available_but_control_gate_incomplete"
    return True, "smoke_positive_validation_selected"


def accuracy_row(
    setting: SelectedSetting,
    peel: dict,
    method: str,
    baseline_method: str,
    metrics: dict | None,
    baseline: dict | None,
    before: float,
    after: float,
    reduces: bool,
    fit,
    peel_mode: str,
    cumulative_primes: str,
    representative_status: str,
    implemented: bool,
    na_reason: str = "",
    control_metrics: dict | None = None,
    force_claim_status: str | None = None,
    quotient_before: float | None = None,
    quotient_after: float | None = None,
    edge_cochain_solve_status: str = "",
    representative_selection_status: str = "",
) -> dict:
    row = make_base_row(
        setting,
        peel,
        int(peel["prime"]),
        peel_mode,
        cumulative_primes,
        fit,
        before,
        after,
        reduces,
        quotient_before=quotient_before,
        quotient_after=quotient_after,
        edge_cochain_solve_status=edge_cochain_solve_status,
        representative_selection_status=representative_selection_status,
    )
    row.update(
        {
            "representative_correction_status": representative_status,
            "method": method,
            "baseline_method": baseline_method,
            "implemented_corrected_merge": bool(implemented),
            "validation_accuracy": safe_float(metrics.get("validation_accuracy")) if metrics else float("nan"),
            "test_accuracy": safe_float(metrics.get("test_accuracy")) if metrics else float("nan"),
            "baseline_validation_accuracy": safe_float(baseline.get("validation_accuracy")) if baseline else float("nan"),
            "baseline_test_accuracy": safe_float(baseline.get("test_accuracy")) if baseline else float("nan"),
            "wrong_prime_control_validation_accuracy": float("nan"),
            "wrong_prime_control_test_accuracy": float("nan"),
            "shuffled_control_validation_accuracy": float("nan"),
            "shuffled_control_test_accuracy": float("nan"),
            "random_residual_control_validation_accuracy": float("nan"),
            "random_residual_control_test_accuracy": float("nan"),
        }
    )
    if control_metrics:
        for prefix, values in control_metrics.items():
            row[f"{prefix}_validation_accuracy"] = safe_float(values.get("validation_accuracy"))
            row[f"{prefix}_test_accuracy"] = safe_float(values.get("test_accuracy"))
    row["validation_delta_vs_baseline"] = row["validation_accuracy"] - row["baseline_validation_accuracy"]
    row["test_delta_vs_baseline"] = row["test_accuracy"] - row["baseline_test_accuracy"]
    row["validation_delta_vs_wrong_prime_control"] = row["validation_accuracy"] - row["wrong_prime_control_validation_accuracy"]
    row["validation_delta_vs_shuffled_control"] = row["validation_accuracy"] - row["shuffled_control_validation_accuracy"]
    row["validation_delta_vs_random_residual_control"] = row["validation_accuracy"] - row["random_residual_control_validation_accuracy"]
    row["na_reason"] = row_missing_reason(row, na_reason)
    selected, status = selection_decision(row)
    row["selected_by_validation"] = bool(selected)
    row["claim_status"] = force_claim_status or status
    return row


def baseline_row(setting: SelectedSetting, metrics: dict, before: float) -> dict:
    peel = {
        "prime": 0,
        "prime_index": -1,
        "p_adic_multiplicity": 0,
        "eligible": False,
        "remaining_order_before": setting.primary_source_order,
        "remaining_order_after": setting.primary_source_order,
    }
    row = make_base_row(setting, peel, 0, "baseline", "", None, before, before, False)
    row.update(
        {
            "quotient_fit_status": "not_applicable_baseline",
            "edge_correction_status": "not_applicable_baseline",
            "representative_correction_status": "not_applicable_baseline",
            "method": "baseline_c2m3_permutation",
            "baseline_method": "baseline_c2m3_permutation",
            "implemented_corrected_merge": False,
            "validation_accuracy": safe_float(metrics.get("validation_accuracy")),
            "test_accuracy": safe_float(metrics.get("test_accuracy")),
            "baseline_validation_accuracy": safe_float(metrics.get("validation_accuracy")),
            "baseline_test_accuracy": safe_float(metrics.get("test_accuracy")),
            "validation_delta_vs_baseline": 0.0,
            "test_delta_vs_baseline": 0.0,
            "wrong_prime_control_validation_accuracy": float("nan"),
            "wrong_prime_control_test_accuracy": float("nan"),
            "shuffled_control_validation_accuracy": float("nan"),
            "shuffled_control_test_accuracy": float("nan"),
            "random_residual_control_validation_accuracy": float("nan"),
            "random_residual_control_test_accuracy": float("nan"),
            "validation_delta_vs_wrong_prime_control": float("nan"),
            "validation_delta_vs_shuffled_control": float("nan"),
            "validation_delta_vs_random_residual_control": float("nan"),
            "selected_by_validation": False,
            "claim_status": "baseline_reference_real_evaluation",
            "na_reason": row_missing_reason(metrics),
        }
    )
    return row


def evaluate_prime_family(
    setting: SelectedSetting,
    group: pd.DataFrame,
    bundle,
    primes: list[int],
) -> tuple[list[dict], list[dict]]:
    pairwise = reconstruct_pairwise_perms(group)
    before = cycle_residual(pairwise, setting.n_models)
    rows = []
    corrected_map_rows = []
    baseline = evaluate_c2m3_from_pairwise(setting, bundle, pairwise)
    rows.append(baseline_row(setting, baseline, before))
    plan = prime_peeling_plan(setting.primary_source_order, primes)
    cumulative_pairwise = {edge: value.copy() for edge, value in pairwise.items()}
    cumulative_primes: list[int] = []

    for peel in plan:
        prime = int(peel["prime"])
        if not peel["eligible"]:
            dummy = make_base_row(setting, peel, prime, "peel_p_only", ",".join(str(p) for p in cumulative_primes), None, before, before, False)
            dummy.update(
                {
                    "representative_correction_status": "prime_not_eligible",
                    "method": "peeled_p_c2m3_permutation",
                    "baseline_method": "baseline_c2m3_permutation",
                    "implemented_corrected_merge": False,
                    "validation_accuracy": float("nan"),
                    "test_accuracy": float("nan"),
                    "baseline_validation_accuracy": baseline["validation_accuracy"],
                    "baseline_test_accuracy": baseline["test_accuracy"],
                    "validation_delta_vs_baseline": float("nan"),
                    "test_delta_vs_baseline": float("nan"),
                    "wrong_prime_control_validation_accuracy": float("nan"),
                    "wrong_prime_control_test_accuracy": float("nan"),
                    "shuffled_control_validation_accuracy": float("nan"),
                    "shuffled_control_test_accuracy": float("nan"),
                    "random_residual_control_validation_accuracy": float("nan"),
                    "random_residual_control_test_accuracy": float("nan"),
                    "validation_delta_vs_wrong_prime_control": float("nan"),
                    "validation_delta_vs_shuffled_control": float("nan"),
                    "validation_delta_vs_random_residual_control": float("nan"),
                    "selected_by_validation": False,
                    "claim_status": "prime_not_eligible",
                    "na_reason": "prime_not_eligible",
                }
            )
            rows.append(dummy)
            continue

        relations = relations_from_pairwise(pairwise, setting.n_models)
        fit = fit_primary_quotient(relations, prime, random_restarts=8, seed=setting.seed + prime)
        result = solve_and_correct_pairwise(pairwise, fit, setting.n_models, prime)
        after = cycle_residual(result.corrected, setting.n_models)
        quotient_reduces = bool(result.solution.quotient_residual_after < result.solution.quotient_residual_before)
        permutation_safe = bool(after <= before + 1e-12)
        reduces = bool(result.implemented and quotient_reduces and permutation_safe)
        corrected_map_rows.extend(
            map_rows_for(
                setting,
                prime,
                "peel_p_only",
                pairwise,
                result.corrections,
                result.corrected,
                result.solution.quotient_residual_before,
                result.solution.quotient_residual_after,
                before,
                after,
                edge_labels=result.edge_labels,
                representative_choices=result.representative_choices,
                edge_cochain_solve_status=result.edge_cochain_solve_status,
                representative_selection_status=result.representative_selection_status,
            )
        )

        wrong_prime = next((candidate for candidate in primes if candidate != prime and setting.primary_source_order % int(candidate) != 0), None)
        wrong_metrics = None
        wrong_after = float("nan")
        wrong_result = None
        if wrong_prime is not None:
            wrong_fit = fit_primary_quotient(relations, int(wrong_prime), random_restarts=4, seed=setting.seed + 997 + int(wrong_prime))
            wrong_result = solve_and_correct_pairwise(pairwise, wrong_fit, setting.n_models, int(wrong_prime))
            wrong_after = cycle_residual(wrong_result.corrected, setting.n_models)
            if wrong_result.implemented:
                wrong_metrics = evaluate_c2m3_from_pairwise(setting, bundle, wrong_result.corrected)
            corrected_map_rows.extend(
                map_rows_for(
                    setting,
                    prime,
                    "wrong_prime_peel_control",
                    pairwise,
                    wrong_result.corrections,
                    wrong_result.corrected,
                    wrong_result.solution.quotient_residual_before,
                    wrong_result.solution.quotient_residual_after,
                    before,
                    wrong_after,
                    edge_labels=wrong_result.edge_labels,
                    representative_choices=wrong_result.representative_choices,
                    edge_cochain_solve_status=wrong_result.edge_cochain_solve_status,
                    representative_selection_status=wrong_result.representative_selection_status,
                )
            )

        shuffled_metrics = None
        random_metrics = None
        shuffled_after = float("nan")
        random_after = float("nan")
        shuffled_q_after = float("nan")
        random_q_after = float("nan")
        if result.implemented:
            shuffled_corrections, shuffled = shuffled_correction_maps(pairwise, result.corrections, setting.n_models, setting.seed + 2000 + prime)
            shuffled_metrics = evaluate_c2m3_from_pairwise(setting, bundle, shuffled)
            shuffled_after = cycle_residual(shuffled, setting.n_models)
            shuffled_q_after = quotient_residual_from_labels(
                quotient_defect_labels_from_pairwise(shuffled, fit, setting.n_models, prime),
                prime,
            )
            corrected_map_rows.extend(
                map_rows_for(
                    setting,
                    prime,
                    "shuffled_quotient_peel_control",
                    pairwise,
                    shuffled_corrections,
                    shuffled,
                    result.solution.quotient_residual_before,
                    shuffled_q_after,
                    before,
                    shuffled_after,
                    edge_labels=result.edge_labels,
                    edge_cochain_solve_status=result.edge_cochain_solve_status,
                    representative_selection_status="shuffled_quotient_control",
                )
            )

            random_corrections, random_maps = random_same_residual_norm_maps(pairwise, result.corrections, setting.n_models, setting.seed + 3000 + prime)
            random_metrics = evaluate_c2m3_from_pairwise(setting, bundle, random_maps)
            random_after = cycle_residual(random_maps, setting.n_models)
            random_q_after = quotient_residual_from_labels(
                quotient_defect_labels_from_pairwise(random_maps, fit, setting.n_models, prime),
                prime,
            )
            corrected_map_rows.extend(
                map_rows_for(
                    setting,
                    prime,
                    "random_same_residual_norm_peel_control",
                    pairwise,
                    random_corrections,
                    random_maps,
                    result.solution.quotient_residual_before,
                    random_q_after,
                    before,
                    random_after,
                    edge_labels=result.edge_labels,
                    edge_cochain_solve_status=result.edge_cochain_solve_status,
                    representative_selection_status="random_same_residual_norm_control",
                )
            )

        control_metrics = {
            "shuffled_control": shuffled_metrics or {},
            "random_residual_control": random_metrics or {},
        }
        if wrong_metrics is not None:
            control_metrics["wrong_prime_control"] = wrong_metrics

        metrics = evaluate_c2m3_from_pairwise(setting, bundle, result.corrected) if result.implemented else None
        if not result.solution.solved_exact:
            na_reason = "quotient_cochain_inconsistent"
        elif not result.implemented:
            na_reason = result.representative_selection_status
        elif not quotient_reduces:
            na_reason = "metric_produced_but_not_claimable"
        elif not permutation_safe:
            na_reason = "quotient_peel_not_permutation_safe"
        else:
            na_reason = ""
        rows.append(
            accuracy_row(
                setting,
                peel,
                "peeled_p_c2m3_permutation",
                "baseline_c2m3_permutation",
                metrics,
                baseline,
                before,
                after,
                reduces,
                fit,
                "peel_p_only",
                str(prime),
                result.representative_selection_status,
                implemented=result.implemented,
                na_reason=na_reason,
                control_metrics=control_metrics,
                quotient_before=result.solution.quotient_residual_before,
                quotient_after=result.solution.quotient_residual_after,
                edge_cochain_solve_status=result.edge_cochain_solve_status,
                representative_selection_status=result.representative_selection_status,
            )
        )
        for control_method, control_peel_mode, control_values, control_after, control_status in [
            ("wrong_prime_peel_c2m3_control", "wrong_prime_peel_control", wrong_metrics, wrong_after, "wrong_prime_control"),
            ("shuffled_quotient_peel_c2m3_control", "shuffled_quotient_peel_control", shuffled_metrics, shuffled_after, "shuffled_control"),
            ("random_same_residual_norm_peel_control", "random_same_residual_norm_peel_control", random_metrics, random_after, "random_residual_control"),
        ]:
            control_q_after = {
                "wrong_prime_peel_c2m3_control": wrong_result.solution.quotient_residual_after if wrong_result else float("nan"),
                "shuffled_quotient_peel_c2m3_control": shuffled_q_after,
                "random_same_residual_norm_peel_control": random_q_after,
            }[control_method]
            rows.append(
                accuracy_row(
                    setting,
                    peel,
                    control_method,
                    "baseline_c2m3_permutation",
                    control_values,
                    baseline,
                    before,
                    control_after,
                    bool(math.isfinite(control_after) and control_after <= before + 1e-12 and math.isfinite(control_q_after) and control_q_after < result.solution.quotient_residual_before),
                    fit,
                    control_peel_mode,
                    str(prime),
                    control_status,
                    implemented=control_values is not None,
                    na_reason="control_metric_unavailable" if control_values is None else "",
                    control_metrics=control_metrics,
                    force_claim_status="control_real_evaluation_not_selectable" if control_values is not None else "control_unavailable",
                    quotient_before=result.solution.quotient_residual_before,
                    quotient_after=control_q_after,
                    edge_cochain_solve_status=result.edge_cochain_solve_status,
                    representative_selection_status=control_status,
                )
            )

        cumulative_before = cycle_residual(cumulative_pairwise, setting.n_models)
        cumulative_relations = relations_from_pairwise(cumulative_pairwise, setting.n_models)
        cumulative_fit = fit_primary_quotient(cumulative_relations, prime, random_restarts=8, seed=setting.seed + 5000 + prime)
        cumulative_result = solve_and_correct_pairwise(cumulative_pairwise, cumulative_fit, setting.n_models, prime)
        cumulative_after = cycle_residual(cumulative_result.corrected, setting.n_models)
        cumulative_q_reduces = bool(cumulative_result.solution.quotient_residual_after < cumulative_result.solution.quotient_residual_before)
        cumulative_perm_safe = bool(cumulative_after <= cumulative_before + 1e-12)
        cumulative_reduces = bool(cumulative_result.implemented and cumulative_q_reduces and cumulative_perm_safe)
        cumulative_original = {edge: value.copy() for edge, value in cumulative_pairwise.items()}
        if cumulative_result.implemented and cumulative_q_reduces:
            cumulative_pairwise = {edge: value.copy() for edge, value in cumulative_result.corrected.items()}
            cumulative_primes.append(prime)
        corrected_map_rows.extend(
            map_rows_for(
                setting,
                prime,
                "cumulative_peel",
                cumulative_original,
                cumulative_result.corrections,
                cumulative_result.corrected,
                cumulative_result.solution.quotient_residual_before,
                cumulative_result.solution.quotient_residual_after,
                cumulative_before,
                cumulative_after,
                edge_labels=cumulative_result.edge_labels,
                representative_choices=cumulative_result.representative_choices,
                edge_cochain_solve_status=cumulative_result.edge_cochain_solve_status,
                representative_selection_status=cumulative_result.representative_selection_status,
            )
        )
        cumulative_metrics = evaluate_c2m3_from_pairwise(setting, bundle, cumulative_pairwise) if cumulative_result.implemented else None
        if not cumulative_result.solution.solved_exact:
            cumulative_na_reason = "quotient_cochain_inconsistent"
        elif not cumulative_result.implemented:
            cumulative_na_reason = cumulative_result.representative_selection_status
        elif not cumulative_q_reduces:
            cumulative_na_reason = "metric_produced_but_not_claimable"
        elif not cumulative_perm_safe:
            cumulative_na_reason = "quotient_peel_not_permutation_safe"
        else:
            cumulative_na_reason = ""
        rows.append(
            accuracy_row(
                setting,
                peel,
                "cumulative_peeled_c2m3_permutation",
                "baseline_c2m3_permutation",
                cumulative_metrics,
                baseline,
                cumulative_before,
                cumulative_after,
                cumulative_reduces,
                cumulative_fit,
                "cumulative",
                ",".join(str(item) for item in cumulative_primes),
                cumulative_result.representative_selection_status,
                implemented=cumulative_result.implemented,
                na_reason=cumulative_na_reason,
                control_metrics=control_metrics,
                quotient_before=cumulative_result.solution.quotient_residual_before,
                quotient_after=cumulative_result.solution.quotient_residual_after,
                edge_cochain_solve_status=cumulative_result.edge_cochain_solve_status,
                representative_selection_status=cumulative_result.representative_selection_status,
            )
        )

        mono = accuracy_row(
            setting,
            peel,
            "peeled_p_monomial_scale",
            "baseline_monomial_scale",
            None,
            None,
            before,
            after,
            reduces,
            fit,
            "peel_p_only",
            str(prime),
            "monomial_correction_adapter_not_available",
            implemented=False,
            na_reason="monomial_correction_adapter_not_available",
            control_metrics=None,
            force_claim_status="monomial_correction_adapter_not_available",
            quotient_before=result.solution.quotient_residual_before,
            quotient_after=result.solution.quotient_residual_after,
            edge_cochain_solve_status=result.edge_cochain_solve_status,
            representative_selection_status="monomial_correction_adapter_not_available",
        )
        rows.append(mono)
    return rows, corrected_map_rows


def paired_stats(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    data = rows[
        rows["implemented_corrected_merge"].fillna(False)
        & rows["method"].astype(str).isin(["peeled_p_c2m3_permutation", "cumulative_peeled_c2m3_permutation"])
    ].copy()
    for (method, dataset), group in data.groupby(["method", "dataset"], sort=True):
        vals = pd.to_numeric(group["validation_delta_vs_baseline"], errors="coerce").dropna()
        tests = pd.to_numeric(group["test_delta_vs_baseline"], errors="coerce").dropna()
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "n_rows": int(len(group)),
                "n_finite_validation_delta": int(len(vals)),
                "n_finite_test_delta": int(len(tests)),
                "mean_validation_delta_vs_baseline": float(vals.mean()) if len(vals) else float("nan"),
                "best_validation_delta_vs_baseline": float(vals.max()) if len(vals) else float("nan"),
                "mean_test_delta_vs_baseline": float(tests.mean()) if len(tests) else float("nan"),
                "selected_by_validation_rows": int(group["selected_by_validation"].fillna(False).sum()),
                "claim_status": "real_positive" if group["selected_by_validation"].fillna(False).any() else "real_negative_or_diagnostic_only",
            }
        )
    return pd.DataFrame(out)


def selected_settings_frame(settings: list[SelectedSetting], primes: list[int]) -> pd.DataFrame:
    return pd.DataFrame([{**setting.__dict__, "prime_list_used": ",".join(str(p) for p in primes)} for setting in settings])


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df[[col for col in columns if col in df.columns]].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = "" if not math.isfinite(value) else f"{value:.6g}"
            vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_status(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "implementation_invalid"
    if rows["selected_by_validation"].fillna(False).any():
        return "real_positive"
    evidence = rows[rows["method"].astype(str).isin(["peeled_p_c2m3_permutation", "cumulative_peeled_c2m3_permutation"])].copy()
    solve_status = evidence["edge_cochain_solve_status"].astype(str) if "edge_cochain_solve_status" in evidence else pd.Series(dtype=str)
    if evidence.empty or not solve_status.str.contains("exact_quotient_cochain_solve|quotient_cochain_inconsistent", regex=True).any():
        return "implementation_invalid"
    exact = solve_status.str.contains("exact_quotient_cochain_solve", regex=False)
    if not exact.any():
        return "diagnostic_only"
    if not evidence["implemented_corrected_merge"].fillna(False).any():
        return "diagnostic_only"
    implemented = rows[rows["implemented_corrected_merge"].fillna(False)]
    finite_val = pd.to_numeric(implemented["validation_accuracy"], errors="coerce").notna()
    finite_test = pd.to_numeric(implemented["test_accuracy"], errors="coerce").notna()
    no_lift = pd.to_numeric(implemented["capacity_multiplier"], errors="coerce").eq(1.0) & pd.to_numeric(
        implemented["inference_multiplier"],
        errors="coerce",
    ).eq(1.0)
    if bool((finite_val & finite_test & no_lift).any()):
        return "real_negative"
    return "diagnostic_only"


def write_report(args, settings: list[SelectedSetting], rows: pd.DataFrame, stats: pd.DataFrame, maps: pd.DataFrame) -> None:
    status = run_status(rows)
    selected = selected_settings_frame(settings, parse_csv(args.prime_list, int))
    eligible = (
        rows[rows["eligible"].fillna(False)]
        .groupby("dataset")["prime"]
        .apply(lambda vals: ",".join(str(int(v)) for v in sorted(set(vals))))
        .to_dict()
    )
    implemented = rows[rows["implemented_corrected_merge"].fillna(False)]
    finite_val = pd.to_numeric(implemented["validation_accuracy"], errors="coerce").notna()
    finite_test = pd.to_numeric(implemented["test_accuracy"], errors="coerce").notna()
    q_before = pd.to_numeric(rows.get("quotient_residual_before", pd.Series(dtype=float)), errors="coerce")
    q_after = pd.to_numeric(rows.get("quotient_residual_after", pd.Series(dtype=float)), errors="coerce")
    p_before = pd.to_numeric(rows.get("permutation_cycle_residual_before", pd.Series(dtype=float)), errors="coerce")
    p_after = pd.to_numeric(rows.get("permutation_cycle_residual_after", pd.Series(dtype=float)), errors="coerce")
    quotient_reduction_rate = float((q_after < q_before).mean()) if len(rows) else float("nan")
    permutation_safe_rate = float((p_after <= p_before + 1e-12).mean()) if len(rows) else float("nan")
    selected_rows = rows[rows["selected_by_validation"].fillna(False)]
    exact_solve_count = int(rows["edge_cochain_solve_status"].astype(str).str.contains("exact_quotient_cochain_solve", regex=False).sum())
    representative_available_count = int(rows["representative_selection_status"].astype(str).eq("representative_correction_available").sum())
    best_peeled = rows[rows["method"].astype(str).eq("peeled_p_c2m3_permutation")]["validation_delta_vs_baseline"].dropna()
    best_cum = rows[rows["method"].astype(str).eq("cumulative_peeled_c2m3_permutation")]["validation_delta_vs_baseline"].dropna()
    best_mono = rows[rows["method"].astype(str).str.contains("monomial")]["validation_delta_vs_baseline"].dropna()
    text = f"""# Primary Residual Peeling Smoke V2

audit_classification: `{status}`

run_status: `{status}`

Generated by `experiments/primary_residual_peeling_smoke_v2.py`.

## Exact Command

```bash
{command_text(sys.argv[1:])}
```

## Git State

- Git commit: `{git_output("rev-parse", "--short", "HEAD")}`
- Dirty status (tracked files only): `{git_output("status", "--short", "--untracked-files=no") or "clean"}`

## Scope

- This is a two-setting smoke test.
- This is no-lift primary residual peeling, not a branch/rank lift.
- Positive rows are hypothesis-generating only.
- This does not prove real Brauer/projective or period-index structure.
- This does not prove broad model-merging improvement.

## Selected Settings

{md_table(selected, ["dataset", "run_id", "setting_id", "n_models", "width", "matching", "relation_count", "relation_count_status", "observed_holonomy_order_lcm", "group_closure_status", "primary_source_order", "model_source"])}

## Eligible Primes

`{json.dumps(eligible, sort_keys=True)}`

## Correction Diagnostics

{md_table(rows[rows["method"].astype(str).eq("peeled_p_c2m3_permutation")], ["dataset", "prime", "eligible", "quotient_fit_status", "quotient_relation_violation_rate", "edge_cochain_solve_status", "representative_selection_status", "quotient_residual_before", "quotient_residual_after", "quotient_residual_reduction", "permutation_cycle_residual_before", "permutation_cycle_residual_after", "permutation_cycle_residual_reduction", "correction_reduces_residual", "implemented_corrected_merge", "na_reason"], 40)}

## Corrected Merge Accuracy

{md_table(rows[rows["method"].astype(str).isin(["baseline_c2m3_permutation", "peeled_p_c2m3_permutation", "cumulative_peeled_c2m3_permutation"])], ["dataset", "prime", "method", "validation_accuracy", "test_accuracy", "baseline_validation_accuracy", "baseline_test_accuracy", "validation_delta_vs_baseline", "test_delta_vs_baseline", "quotient_residual_reduction", "permutation_cycle_residual_reduction", "capacity_multiplier", "inference_multiplier", "selected_by_validation", "claim_status", "na_reason"], 80)}

## Controls

{md_table(rows[rows["method"].astype(str).str.contains("control")], ["dataset", "prime", "method", "validation_accuracy", "test_accuracy", "baseline_validation_accuracy", "validation_delta_vs_baseline", "claim_status", "na_reason"], 80)}

## Paired Stats

{md_table(stats, ["dataset", "method", "n_rows", "n_finite_validation_delta", "n_finite_test_delta", "mean_validation_delta_vs_baseline", "best_validation_delta_vs_baseline", "mean_test_delta_vs_baseline", "selected_by_validation_rows", "claim_status"], 40)}

## Final Interpretation

- Audit classification: `{status}`
- Run status: `{status}`
- Corrected map rows: `{len(maps)}`
- Exact quotient cochain solves: `{exact_solve_count}`
- Representative corrections available: `{representative_available_count}`
- Quotient residual reduction rate: `{quotient_reduction_rate:.6g}`
- Permutation residual safe-correction rate: `{permutation_safe_rate:.6g}`
- Corrected merge implemented count: `{len(implemented)}`
- Finite validation metrics count: `{int(finite_val.sum())}`
- Finite test metrics count: `{int(finite_test.sum())}`
- Best peeled C2M3 validation delta vs unpeeled C2M3: `{float(best_peeled.max()) if len(best_peeled) else "not_run"}`
- Best cumulative peeled C2M3 validation delta vs unpeeled C2M3: `{float(best_cum.max()) if len(best_cum) else "not_run"}`
- Best peeled monomial validation delta vs unpeeled monomial: `{float(best_mono.max()) if len(best_mono) else "not_run"}`
- Wrong-prime control available: `{"yes" if rows["wrong_prime_control_validation_accuracy"].notna().any() else "no"}`
- Shuffled control available: `{"yes" if rows["shuffled_control_validation_accuracy"].notna().any() else "no"}`
- Random residual control available: `{"yes" if rows["random_residual_control_validation_accuracy"].notna().any() else "no"}`
- Selected methods count: `{len(selected_rows)}`
- Selected methods details: `{", ".join(selected_rows["method"].astype(str).tolist()) if len(selected_rows) else "none"}`

Final interpretation:
- `{"real_positive" if len(selected_rows) else status}`
- Selection remains validation-gated and test metrics are evaluation-only.
"""
    (args.reports_dir / "primary_residual_peeling_smoke_v2_report.md").write_text(text, encoding="utf-8")


def assert_completed_with_metrics(rows: pd.DataFrame) -> None:
    status = run_status(rows)
    if status == "implementation_invalid":
        raise RuntimeError(
            f"primary_residual_peeling_smoke_v2 implementation_invalid: quotient cochain solve evidence was not produced"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--model-counts", default="4,3")
    parser.add_argument("--settings-per-dataset", type=int, default=1)
    parser.add_argument("--prime-list", default="2,3,5,7,17,19,43")
    parser.add_argument("--prefer-n-models", type=int, default=4)
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
        raise RuntimeError(f"failed_to_run_corrected_merge: missing triangle/run artifacts for datasets: {missing}")

    settings = []
    all_rows = []
    all_map_rows = []
    for dataset in datasets:
        group = chosen[dataset]
        setting = summarize_relation_set(group, args.reports_dir, args.max_group_order, args.max_generators, args.max_exact_order)
        if setting.architecture != "mlp":
            raise RuntimeError(f"failed_to_run_corrected_merge: v2 adapter currently supports mlp, got {setting.architecture}")
        settings.append(setting)
        bundle = build_loaders_and_models(setting, runs, args.reports_dir, args.data_dir, args.device)
        rows, map_rows = evaluate_prime_family(setting, group, bundle, primes)
        all_rows.extend(rows)
        all_map_rows.extend(map_rows)

    rows = pd.DataFrame(all_rows)
    for col in MAIN_COLUMNS:
        if col not in rows:
            rows[col] = np.nan
    rows = rows[MAIN_COLUMNS + [col for col in rows.columns if col not in MAIN_COLUMNS]].copy()
    maps_out = pd.DataFrame(all_map_rows)
    for col in CORRECTED_MAP_COLUMNS:
        if col not in maps_out:
            maps_out[col] = np.nan
    maps_out = maps_out[CORRECTED_MAP_COLUMNS].copy()
    stats = paired_stats(rows)
    selected = selected_settings_frame(settings, primes)

    assert_completed_with_metrics(rows)

    rows.to_csv(args.reports_dir / "csv" / "primary_residual_peeling_smoke_v2.csv", index=False, lineterminator="\n")
    stats.to_csv(args.reports_dir / "csv" / "primary_residual_peeling_smoke_v2_paired_stats.csv", index=False, lineterminator="\n")
    selected.to_csv(args.reports_dir / "csv" / "primary_residual_peeling_smoke_v2_selected_settings.csv", index=False, lineterminator="\n")
    maps_out.to_csv(args.reports_dir / "csv" / "primary_residual_peeling_smoke_v2_corrected_maps.csv", index=False, lineterminator="\n")
    write_report(args, settings, rows, stats, maps_out)

    status = run_status(rows)
    implemented = rows[rows["implemented_corrected_merge"].fillna(False)]
    finite_val = pd.to_numeric(implemented["validation_accuracy"], errors="coerce").notna()
    finite_test = pd.to_numeric(implemented["test_accuracy"], errors="coerce").notna()
    selected_rows = rows[rows["selected_by_validation"].fillna(False)]
    eligible = (
        rows[rows["eligible"].fillna(False)]
        .groupby("dataset")["prime"]
        .apply(lambda vals: sorted(set(map(int, vals))))
        .to_dict()
    )
    best_peeled = rows[rows["method"].astype(str).eq("peeled_p_c2m3_permutation")]["validation_delta_vs_baseline"].dropna()
    best_cum = rows[rows["method"].astype(str).eq("cumulative_peeled_c2m3_permutation")]["validation_delta_vs_baseline"].dropna()
    best_mono = rows[rows["method"].astype(str).str.contains("monomial")]["validation_delta_vs_baseline"].dropna()
    q_before = pd.to_numeric(rows["quotient_residual_before"], errors="coerce")
    q_after = pd.to_numeric(rows["quotient_residual_after"], errors="coerce")
    p_before = pd.to_numeric(rows["permutation_cycle_residual_before"], errors="coerce")
    p_after = pd.to_numeric(rows["permutation_cycle_residual_after"], errors="coerce")
    quotient_reduction_rate = float((q_after < q_before).mean()) if len(rows) else float("nan")
    permutation_safe_rate = float((p_after <= p_before + 1e-12).mean()) if len(rows) else float("nan")
    exact_solve_count = int(rows["edge_cochain_solve_status"].astype(str).str.contains("exact_quotient_cochain_solve", regex=False).sum())
    representative_available_count = int(rows["representative_selection_status"].astype(str).eq("representative_correction_available").sum())
    controls_passed = int(selected_rows["selected_by_validation"].fillna(False).sum())
    if status == "diagnostic_only":
        blockers = sorted(set(rows["na_reason"].dropna().astype(str)) - {""})
        final_interpretation = "diagnostic_only: " + (", ".join(blockers[:5]) if blockers else "no selectable corrected rows")
    elif status == "real_negative":
        final_interpretation = "real_negative: quotient cochain corrections ran but no validation-selected correction beat baseline and controls"
    elif status == "real_positive":
        final_interpretation = "real_positive: at least one validation-selected no-lift correction beat baseline and controls"
    else:
        final_interpretation = "implementation_invalid: quotient cochain solve evidence missing"
    print(f"Audit classification: {status}")
    print(f"Run status: {status}")
    print("\nSelected settings:")
    for setting in settings:
        print(f"- {setting.dataset}: {setting.run_id} (N={setting.n_models}, W={setting.width}, source={setting.model_source})")
    print("\nEligible primes:")
    for dataset in datasets:
        print(f"- {dataset}: {eligible.get(dataset, [])}")
    print("\nCorrected map rows:")
    print(f"- count: {len(maps_out)}")
    print(f"- quotient residual reduction rate: {quotient_reduction_rate:.6g}")
    print(f"- permutation residual safe-correction rate: {permutation_safe_rate:.6g}")
    print(f"- exact quotient cochain solves: {exact_solve_count}")
    print(f"- representative corrections available: {representative_available_count}")
    print("\nCorrected merge rows:")
    print(f"- implemented count: {len(implemented)}")
    print(f"- finite validation metrics count: {int(finite_val.sum())}")
    print(f"- finite test metrics count: {int(finite_test.sum())}")
    print("\nBest validation deltas:")
    print(f"- peeled C2M3 vs unpeeled C2M3: {float(best_peeled.max()) if len(best_peeled) else 'not_run'}")
    print(f"- cumulative peeled C2M3 vs unpeeled C2M3: {float(best_cum.max()) if len(best_cum) else 'not_run'}")
    print(f"- peeled monomial vs unpeeled monomial: {float(best_mono.max()) if len(best_mono) else 'not_run'}")
    print("\nControls:")
    print(f"- wrong-prime control available: {'yes' if rows['wrong_prime_control_validation_accuracy'].notna().any() else 'no'}")
    print(f"- shuffled control available: {'yes' if rows['shuffled_control_validation_accuracy'].notna().any() else 'no'}")
    print(f"- random residual control available: {'yes' if rows['random_residual_control_validation_accuracy'].notna().any() else 'no'}")
    print(f"- controls passed: {controls_passed}")
    print("\nSelected methods:")
    print(f"- count: {len(selected_rows)}")
    print(f"- details: {', '.join(selected_rows['method'].astype(str).tolist()) if len(selected_rows) else 'none'}")
    print("\nFinal interpretation:")
    print(f"- {final_interpretation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
