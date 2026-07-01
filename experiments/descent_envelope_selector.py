#!/usr/bin/env python
"""Validation-descent envelope over generated same-base candidates.

The same-base task-vector benchmark evaluates several candidate generators
separately.  This script treats those generated models as a shared candidate
pool and asks whether validation descent improves when the pool is expanded.
It uses test metrics only after each selector has been frozen by validation
metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.same_base_task_vector_benchmark import (  # noqa: E402
    average_vectors,
    combined_loader,
    dare_vector,
    evaluate_across_tasks,
    make_subset,
    sample_indices,
    slerp,
    sequential_slerp,
    split_indices,
    state_vector,
    subset_by_classes,
    task_arithmetic_vector,
    ties_vector,
    vector_to_model,
)
from src.model_merging_benchmark import (  # noqa: E402
    DatasetSpec,
    average_models,
    clone_model,
    device_from_arg,
    format_markdown_table,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    require_torch,
)


RUN_CSV = "descent_envelope_selector.csv"
SUMMARY_CSV = "descent_envelope_summary.csv"
REPORT_MD = "descent_envelope_selector_report.md"
PLOT_PDF = "descent_envelope_deltas.pdf"
TABLE_TEX = "descent_envelope_table.tex"

TASK_VECTOR_FAMILIES = {"slerp_grid", "slerp_sequential", "task_arithmetic", "ties_merging", "dare"}
BASELINE_SELECTOR = "greedy_soup_original_pool"
SELECTORS = [
    BASELINE_SELECTOR,
    "best_validation_generated_candidate",
    "greedy_soup_over_generated_candidates",
    "conservative_lcb_generated_candidate",
    "worst_task_validation_selector",
]


@dataclass
class Candidate:
    label: str
    family: str
    kind: str
    model: object
    params: dict
    uses_validation_data: bool
    val_average_accuracy: float = float("nan")
    val_worst_task_accuracy: float = float("nan")
    val_average_loss: float = float("nan")
    test_average_accuracy: float = float("nan")
    test_worst_task_accuracy: float = float("nan")
    test_average_loss: float = float("nan")
    validation_lcb: float = float("nan")


@dataclass
class LoadedRun:
    row: pd.Series
    spec: DatasetSpec
    val_loaders: dict[str, object]
    test_loaders: dict[str, object]
    combined_val_loader: object
    combined_test_loader: object
    base_model: object
    task_models: list


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_seed_text(text: str | None) -> set[int] | None:
    if text is None or not str(text).strip():
        return None
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            out.update(range(int(start), int(end)))
        elif "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_mean(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def bootstrap_mean_ci(values, samples: int, seed: int) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(max(1, int(samples))):
        idx = rng.integers(0, len(arr), len(arr))
        draws.append(float(arr[idx].mean()))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def resolve_path(path_text: str | Path) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    text = str(path_text)
    marker = "reports/checkpoints/same_base_task_vector/"
    if marker in text:
        candidate = ROOT / text[text.index(marker) :]
        if candidate.exists():
            return candidate
    return path


def load_checkpoint_model(path: Path, architecture: str, spec: DatasetSpec, width: int):
    torch, _, _ = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(architecture, spec, width)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def task_rows(payload: str) -> list[dict]:
    rows = json.loads(str(payload))
    return [
        {
            "task_name": str(row["task_name"]),
            "classes": tuple(int(item) for item in row["classes"]),
            "val_samples": int(row["val_samples"]),
            "test_samples": int(row["test_samples"]),
        }
        for row in rows
    ]


def loaders_for_run(row: pd.Series, args: argparse.Namespace):
    dataset = str(row["dataset"])
    seed = int(row["seed"])
    max_train = int(row.get("max_train_samples", args.max_train_samples))
    max_test = int(row.get("max_test_samples", args.max_test_samples))
    spec, train_base, test_base = load_dataset(
        dataset,
        args.data_dir,
        max_train,
        max_test,
        int(args.dataset_seed),
        augmentation=str(args.augmentation),
    )
    _train_indices, val_indices = split_indices(len(train_base), float(args.val_fraction), int(args.dataset_seed) + 17 + seed)
    val_loaders: dict[str, object] = {}
    test_loaders: dict[str, object] = {}
    for idx, task in enumerate(task_rows(row["task_definitions_json"])):
        val_indices_task = subset_by_classes(
            train_base,
            val_indices,
            task["classes"],
            task["val_samples"],
            seed + 2000 + idx,
        )
        test_indices_task = subset_by_classes(
            test_base,
            list(range(len(test_base))),
            task["classes"],
            task["test_samples"],
            seed + 3000 + idx,
        )
        val_loaders[task["task_name"]] = make_loader(
            make_subset(train_base, val_indices_task),
            int(args.batch_size),
            shuffle=False,
            seed=seed + 5000 + idx,
        )
        test_loaders[task["task_name"]] = make_loader(
            make_subset(test_base, test_indices_task),
            int(args.batch_size),
            shuffle=False,
            seed=seed + 6000 + idx,
        )
    combined_val = combined_loader(val_loaders, int(args.batch_size), seed + 7000)
    combined_test = combined_loader(test_loaders, int(args.batch_size), seed + 8000)
    return spec, val_loaders, test_loaders, combined_val, combined_test


def load_run(row: pd.Series, args: argparse.Namespace) -> LoadedRun | None:
    spec, val_loaders, test_loaders, combined_val, combined_test = loaders_for_run(row, args)
    architecture = str(row["architecture"])
    width = int(row["width"])
    base_path = resolve_path(row["base_checkpoint"])
    task_paths = [resolve_path(item) for item in json.loads(str(row["task_checkpoints_json"]))]
    if not base_path.exists() or not all(path.exists() for path in task_paths):
        return None
    base_model = load_checkpoint_model(base_path, architecture, spec, width)
    task_models = [load_checkpoint_model(path, architecture, spec, width) for path in task_paths]
    return LoadedRun(
        row=row,
        spec=spec,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        combined_val_loader=combined_val,
        combined_test_loader=combined_test,
        base_model=base_model,
        task_models=task_models,
    )


def eval_candidate(candidate: Candidate, loaded: LoadedRun, device, n_val_examples: int, lcb_z: float) -> Candidate:
    val_metrics = evaluate_across_tasks(candidate.model, loaded.val_loaders, device)
    candidate.val_average_accuracy = float(val_metrics["average_accuracy"])
    candidate.val_worst_task_accuracy = float(val_metrics["worst_accuracy"])
    candidate.val_average_loss = float(val_metrics["average_loss"])
    p = min(max(candidate.val_average_accuracy, 0.0), 1.0)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / max(int(n_val_examples), 1))
    candidate.validation_lcb = float(p - float(lcb_z) * se)
    return candidate


def ensure_test_metrics(candidate: Candidate, loaded: LoadedRun, device) -> Candidate:
    if math.isfinite(candidate.test_average_accuracy):
        return candidate
    test_metrics = evaluate_across_tasks(candidate.model, loaded.test_loaders, device)
    candidate.test_average_accuracy = float(test_metrics["average_accuracy"])
    candidate.test_worst_task_accuracy = float(test_metrics["worst_accuracy"])
    candidate.test_average_loss = float(test_metrics["average_loss"])
    return candidate


def add_candidate(candidates: list[Candidate], label: str, family: str, kind: str, model, params: dict | None = None, uses_validation_data: bool = False) -> None:
    candidates.append(
        Candidate(
            label=label,
            family=family,
            kind=kind,
            model=model,
            params=params or {},
            uses_validation_data=uses_validation_data,
        )
    )


def generate_candidates(loaded: LoadedRun, args: argparse.Namespace, device) -> list[Candidate]:
    row = loaded.row
    architecture = str(row["architecture"])
    width = int(row["width"])
    candidates: list[Candidate] = []
    add_candidate(candidates, "base_model", "base_model", "common_base_reference", loaded.base_model)
    for idx, model in enumerate(loaded.task_models):
        add_candidate(candidates, f"fine_tuned_task_{idx}", "fine_tuned_task_model", "task_model", model, {"task_index": idx})
    weight_model = average_models(loaded.task_models, architecture, loaded.spec, width)
    add_candidate(candidates, "weight_average", "weight_average", "ordinary_weight_average", weight_model)

    greedy_model, greedy_indices, _test_metrics, _trajectory = greedy_soup(
        loaded.task_models,
        loaded.combined_val_loader,
        loaded.combined_val_loader,
        device,
        architecture,
        loaded.spec,
        width,
        return_trajectory=True,
    )
    add_candidate(
        candidates,
        "greedy_soup_original_pool_final",
        "greedy_soup_original_pool",
        "validation_soup_over_fine_tuned_tasks",
        greedy_model,
        {"selection_indices": greedy_indices},
        uses_validation_data=True,
    )

    base_vector, meta = state_vector(loaded.base_model)
    task_vectors = [state_vector(model)[0] for model in loaded.task_models]
    mean_task_vector = average_vectors(task_vectors)
    seq_model = vector_to_model(sequential_slerp(task_vectors), meta, architecture, loaded.spec, width)
    add_candidate(candidates, "slerp_sequential", "slerp_sequential", "same_base_slerp", seq_model)
    for t in parse_csv(args.slerp_grid, float):
        model = vector_to_model(slerp(base_vector, mean_task_vector, float(t)), meta, architecture, loaded.spec, width)
        add_candidate(candidates, f"slerp_grid_t={float(t):g}", "slerp_grid", "base_to_mean_task_slerp", model, {"t": float(t)})

    deltas = [vec - base_vector for vec in task_vectors]
    for scale in parse_csv(args.task_arithmetic_scales, float):
        model = vector_to_model(task_arithmetic_vector(base_vector, deltas, float(scale)), meta, architecture, loaded.spec, width)
        add_candidate(candidates, f"task_arithmetic_scale={float(scale):g}", "task_arithmetic", "task_arithmetic_grid", model, {"scale": float(scale)})
    for density in parse_csv(args.ties_densities, float):
        for scale in parse_csv(args.ties_scales, float):
            model = vector_to_model(ties_vector(base_vector, deltas, float(density), float(scale)), meta, architecture, loaded.spec, width)
            add_candidate(
                candidates,
                f"ties_density={float(density):g}_scale={float(scale):g}",
                "ties_merging",
                "ties_grid",
                model,
                {"density": float(density), "scale": float(scale)},
            )
    for drop_rate in parse_csv(args.dare_drop_rates, float):
        for scale in parse_csv(args.dare_scales, float):
            model = vector_to_model(
                dare_vector(base_vector, deltas, float(drop_rate), float(scale), int(row["seed"]) + int(10000 * float(drop_rate)) + int(100 * float(scale))),
                meta,
                architecture,
                loaded.spec,
                width,
            )
            add_candidate(
                candidates,
                f"dare_drop={float(drop_rate):g}_scale={float(scale):g}",
                "dare",
                "dare_grid",
                model,
                {"drop_rate": float(drop_rate), "scale": float(scale)},
            )
    return candidates


def candidate_sort_key(candidate: Candidate) -> tuple[float, float, float]:
    return (
        safe_float(candidate.val_average_accuracy),
        safe_float(candidate.val_worst_task_accuracy),
        -safe_float(candidate.val_average_loss),
    )


def best_by_validation(candidates: list[Candidate]) -> Candidate:
    return max(candidates, key=candidate_sort_key)


def best_by_worst_task_validation(candidates: list[Candidate]) -> Candidate:
    return max(
        candidates,
        key=lambda c: (
            safe_float(c.val_worst_task_accuracy),
            safe_float(c.val_average_accuracy),
            -safe_float(c.val_average_loss),
        ),
    )


def best_task_vector_validation(candidates: list[Candidate]) -> Candidate:
    task_candidates = [candidate for candidate in candidates if candidate.family in TASK_VECTOR_FAMILIES]
    return best_by_validation(task_candidates if task_candidates else candidates)


def conservative_lcb_candidate(candidates: list[Candidate], original: Candidate, margin: float) -> Candidate:
    best = max(candidates, key=lambda c: (safe_float(c.validation_lcb), safe_float(c.val_average_accuracy), -safe_float(c.val_average_loss)))
    if safe_float(best.validation_lcb) > safe_float(original.validation_lcb) + float(margin):
        return best
    return original


def greedy_soup_generated(candidates: list[Candidate], loaded: LoadedRun, args: argparse.Namespace, device) -> tuple[object, list[int], dict, dict]:
    architecture = str(loaded.row["architecture"])
    width = int(loaded.row["width"])
    order = sorted(range(len(candidates)), key=lambda idx: candidate_sort_key(candidates[idx]), reverse=True)
    selected = [order[0]]
    soup = clone_model(candidates[order[0]].model, architecture, loaded.spec, width)
    best_metrics = evaluate_across_tasks(soup, loaded.val_loaders, device)
    best_acc = float(best_metrics["average_accuracy"])
    best_loss = float(best_metrics["average_loss"])
    trajectory = [
        {
            "candidate_rank": 1,
            "candidate_index": int(order[0]),
            "candidate_label": candidates[order[0]].label,
            "candidate_family": candidates[order[0]].family,
            "soup_indices_before": [],
            "soup_indices_after": list(selected),
            "validation_accuracy_before": float("nan"),
            "validation_loss_before": float("nan"),
            "candidate_soup_validation_accuracy": best_acc,
            "candidate_soup_validation_loss": best_loss,
            "accepted": True,
            "decision_reason": "accepted_initial_best_validation_candidate",
        }
    ]
    for rank, idx in enumerate(order[1:], start=2):
        before = list(selected)
        candidate_indices = selected + [idx]
        candidate_soup = average_models([candidates[item].model for item in candidate_indices], architecture, loaded.spec, width)
        metrics = evaluate_across_tasks(candidate_soup, loaded.val_loaders, device)
        acc = float(metrics["average_accuracy"])
        loss = float(metrics["average_loss"])
        accepted = bool(acc >= best_acc)
        if accepted:
            soup = candidate_soup
            selected = candidate_indices
            best_acc = acc
            best_loss = loss
        trajectory.append(
            {
                "candidate_rank": int(rank),
                "candidate_index": int(idx),
                "candidate_label": candidates[idx].label,
                "candidate_family": candidates[idx].family,
                "soup_indices_before": before,
                "soup_indices_after": list(selected),
                "validation_accuracy_before": float(best_acc if accepted else metrics["average_accuracy"] - (acc - best_acc)),
                "validation_loss_before": float(best_loss if accepted else best_loss),
                "candidate_soup_validation_accuracy": acc,
                "candidate_soup_validation_loss": loss,
                "accepted": accepted,
                "decision_reason": "accepted_validation_average_accuracy_non_decrease" if accepted else "rejected_validation_average_accuracy_decrease",
            }
        )
    val_metrics = evaluate_across_tasks(soup, loaded.val_loaders, device)
    test_metrics = evaluate_across_tasks(soup, loaded.test_loaders, device)
    payload = {
        "val_average_accuracy": float(val_metrics["average_accuracy"]),
        "val_worst_task_accuracy": float(val_metrics["worst_accuracy"]),
        "val_average_loss": float(val_metrics["average_loss"]),
        "test_average_accuracy": float(test_metrics["average_accuracy"]),
        "test_worst_task_accuracy": float(test_metrics["worst_accuracy"]),
        "test_average_loss": float(test_metrics["average_loss"]),
        "selected_indices": selected,
        "selected_labels": [candidates[item].label for item in selected],
        "selected_families": [candidates[item].family for item in selected],
        "selected_count": len(selected),
        "selected_family": "generated_candidate_soup",
        "selected_label": "greedy_soup_over_generated_candidates",
        "selected_kind": "validation_soup_over_generated_candidates",
        "selected_params": {"selection_indices": selected},
    }
    return soup, selected, payload, {"trajectory": trajectory}


def base_metadata(row: pd.Series) -> dict:
    return {
        "setting_id": str(row["setting_id"]),
        "run_id": str(row["run_id"]),
        "dataset": str(row["dataset"]),
        "task_preset": str(row["task_preset"]),
        "architecture": str(row["architecture"]),
        "width": int(row["width"]),
        "n_tasks": int(row["n_tasks"]),
        "seed": int(row["seed"]),
        "base_epochs": int(row["base_epochs"]),
        "finetune_epochs": int(row["finetune_epochs"]),
        "max_train_samples": int(row["max_train_samples"]),
        "max_test_samples": int(row["max_test_samples"]),
        "task_vector_sign_conflict_fraction": safe_float(row.get("task_vector_sign_conflict_fraction")),
        "task_vector_mean_pairwise_cosine": safe_float(row.get("task_vector_mean_pairwise_cosine")),
        "task_vector_min_pairwise_cosine": safe_float(row.get("task_vector_min_pairwise_cosine")),
        "triangle_cycle_score": safe_float(row.get("triangle_cycle_score")),
        "sync_disagreement": safe_float(row.get("sync_disagreement")),
    }


def selector_row(
    selector: str,
    selected_payload: dict,
    original: Candidate,
    best_task_vector: Candidate,
    candidates: list[Candidate],
    row: pd.Series,
) -> dict:
    selected_families = selected_payload.get("selected_families", [selected_payload["selected_family"]])
    delta_greedy = safe_float(selected_payload["test_average_accuracy"]) - safe_float(original.test_average_accuracy)
    delta_greedy_worst = safe_float(selected_payload["test_worst_task_accuracy"]) - safe_float(original.test_worst_task_accuracy)
    delta_task_vector = safe_float(selected_payload["test_average_accuracy"]) - safe_float(best_task_vector.test_average_accuracy)
    return {
        **base_metadata(row),
        "selector": selector,
        "validation_average_accuracy": selected_payload["val_average_accuracy"],
        "validation_worst_task_accuracy": selected_payload["val_worst_task_accuracy"],
        "validation_average_loss": selected_payload["val_average_loss"],
        "test_average_accuracy": selected_payload["test_average_accuracy"],
        "test_worst_task_accuracy": selected_payload["test_worst_task_accuracy"],
        "test_average_loss": selected_payload["test_average_loss"],
        "delta_test_average_accuracy_vs_original_greedy": delta_greedy,
        "delta_test_worst_task_accuracy_vs_original_greedy": delta_greedy_worst,
        "delta_test_average_accuracy_vs_best_single_task_vector_method": delta_task_vector,
        "original_greedy_test_average_accuracy": original.test_average_accuracy,
        "original_greedy_test_worst_task_accuracy": original.test_worst_task_accuracy,
        "best_single_task_vector_label": best_task_vector.label,
        "best_single_task_vector_family": best_task_vector.family,
        "best_single_task_vector_test_average_accuracy": best_task_vector.test_average_accuracy,
        "selected_candidate_label": selected_payload["selected_label"],
        "selected_candidate_family": selected_payload["selected_family"],
        "selected_candidate_kind": selected_payload["selected_kind"],
        "selected_candidate_params_json": compact_json(selected_payload.get("selected_params", {})),
        "selected_candidate_families_json": compact_json(selected_families),
        "selected_candidate_count": int(selected_payload.get("selected_count", 1)),
        "generated_candidate_count": int(len(candidates)),
        "candidate_families_json": compact_json(sorted(set(candidate.family for candidate in candidates))),
        "uses_validation_data": selector != BASELINE_SELECTOR,
        "test_used_for_selection": False,
        "selection_rule": selector,
    }


def candidate_payload(candidate: Candidate) -> dict:
    return {
        "val_average_accuracy": candidate.val_average_accuracy,
        "val_worst_task_accuracy": candidate.val_worst_task_accuracy,
        "val_average_loss": candidate.val_average_loss,
        "test_average_accuracy": candidate.test_average_accuracy,
        "test_worst_task_accuracy": candidate.test_worst_task_accuracy,
        "test_average_loss": candidate.test_average_loss,
        "selected_label": candidate.label,
        "selected_family": candidate.family,
        "selected_kind": candidate.kind,
        "selected_params": candidate.params,
        "selected_families": [candidate.family],
        "selected_count": 1,
    }


def evaluate_run(row: pd.Series, args: argparse.Namespace, device) -> list[dict]:
    loaded = load_run(row, args)
    if loaded is None:
        skipped = {
            **base_metadata(row),
            "selector": "all",
            "status": "skipped",
            "skip_reason": "missing_same_base_checkpoint",
            "test_used_for_selection": False,
        }
        return [skipped]
    n_val_examples = sum(len(loader.dataset) for loader in loaded.val_loaders.values())
    candidates = generate_candidates(loaded, args, device)
    evaluated = [
        eval_candidate(candidate, loaded, device, n_val_examples, float(args.lcb_z))
        for candidate in candidates
    ]
    original = next(candidate for candidate in evaluated if candidate.family == "greedy_soup_original_pool")
    best_task_vector = best_task_vector_validation(evaluated)
    ensure_test_metrics(original, loaded, device)
    ensure_test_metrics(best_task_vector, loaded, device)
    rows = []
    selector_candidates = {
        BASELINE_SELECTOR: original,
        "best_validation_generated_candidate": best_by_validation(evaluated),
        "conservative_lcb_generated_candidate": conservative_lcb_candidate(evaluated, original, float(args.lcb_margin)),
        "worst_task_validation_selector": best_by_worst_task_validation(evaluated),
    }
    for selector, candidate in selector_candidates.items():
        ensure_test_metrics(candidate, loaded, device)
        row_out = selector_row(selector, candidate_payload(candidate), original, best_task_vector, evaluated, row)
        row_out["status"] = "ok"
        row_out["skip_reason"] = ""
        rows.append(row_out)
    _soup, _selected, soup_payload, _audit = greedy_soup_generated(evaluated, loaded, args, device)
    row_out = selector_row("greedy_soup_over_generated_candidates", soup_payload, original, best_task_vector, evaluated, row)
    row_out["status"] = "ok"
    row_out["skip_reason"] = ""
    rows.append(row_out)
    for candidate in evaluated:
        candidate.model.to("cpu")
    loaded.base_model.to("cpu")
    for model in loaded.task_models:
        model.to("cpu")
    return rows


def run_rows(args: argparse.Namespace) -> pd.DataFrame:
    device = device_from_arg(str(args.device))
    benchmark = pd.read_csv(args.benchmark_csv)
    base_rows = benchmark[
        (benchmark["method"].astype(str).eq("base_model"))
        & (benchmark["status"].astype(str).eq("ok"))
    ].drop_duplicates("run_id")
    datasets = set(parse_csv(args.datasets, str)) if str(args.datasets).strip() else None
    task_presets = set(parse_csv(args.task_presets, str)) if str(args.task_presets).strip() else None
    settings = set(parse_csv(args.settings, str)) if str(args.settings).strip() else None
    seeds = parse_seed_text(args.seeds)
    if datasets is not None:
        base_rows = base_rows[base_rows["dataset"].astype(str).isin(datasets)]
    if task_presets is not None:
        base_rows = base_rows[base_rows["task_preset"].astype(str).isin(task_presets)]
    if settings is not None:
        base_rows = base_rows[base_rows["setting_id"].astype(str).isin(settings)]
    if seeds is not None:
        base_rows = base_rows[base_rows["seed"].astype(int).isin(seeds)]
    selected_rows = []
    for _setting, group in base_rows.sort_values(["setting_id", "seed"]).groupby("setting_id", sort=True):
        if int(args.max_runs_per_setting) > 0:
            group = group.head(int(args.max_runs_per_setting))
        selected_rows.append(group)
    if selected_rows:
        base_rows = pd.concat(selected_rows, ignore_index=True)
    else:
        base_rows = base_rows.head(0)
    out = []
    for idx, row in enumerate(base_rows.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        print(f"[{idx}/{len(base_rows)}] {row_series['run_id']}", flush=True)
        out.extend(evaluate_run(row_series, args, device))
    return pd.DataFrame(out)


def summarize(rows: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    ok = rows[rows["status"].astype(str).eq("ok")].copy()
    records = []
    group_cols = ["dataset", "task_preset", "architecture", "width", "n_tasks", "selector"]
    for key, group in ok.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, key))
        delta = pd.to_numeric(group["delta_test_average_accuracy_vs_original_greedy"], errors="coerce")
        delta_worst = pd.to_numeric(group["delta_test_worst_task_accuracy_vs_original_greedy"], errors="coerce")
        delta_task = pd.to_numeric(group["delta_test_average_accuracy_vs_best_single_task_vector_method"], errors="coerce")
        low, high = bootstrap_mean_ci(delta, bootstrap_samples, seed=2003 + len(records) * 17)
        worst_low, worst_high = bootstrap_mean_ci(delta_worst, bootstrap_samples, seed=3001 + len(records) * 19)
        family_counts = group["selected_candidate_family"].value_counts().to_dict()
        n = int(len(group))
        if meta["selector"] == BASELINE_SELECTOR:
            claim = "baseline_original_greedy_soup"
        elif n >= 20 and safe_float(low) > 0.0:
            claim = "supported_exact_setting_enriched_pool_improves_original_greedy"
        elif n >= 20 and safe_float(high) < 0.0:
            claim = "negative_exact_setting_enriched_pool_underperforms_original_greedy"
        elif n < 20:
            claim = "descriptive_below_20_seed_gate"
        else:
            claim = "descriptive_ci_overlaps_original_greedy"
        records.append(
            {
                **meta,
                "n_rows": n,
                "n_unique_seeds": int(group["seed"].nunique()),
                "mean_validation_average_accuracy": safe_mean(group["validation_average_accuracy"]),
                "mean_validation_worst_task_accuracy": safe_mean(group["validation_worst_task_accuracy"]),
                "mean_test_average_accuracy": safe_mean(group["test_average_accuracy"]),
                "mean_test_worst_task_accuracy": safe_mean(group["test_worst_task_accuracy"]),
                "mean_delta_test_average_accuracy_vs_original_greedy": safe_mean(delta),
                "delta_vs_original_greedy_ci_low": low,
                "delta_vs_original_greedy_ci_high": high,
                "mean_delta_test_worst_task_accuracy_vs_original_greedy": safe_mean(delta_worst),
                "worst_task_delta_ci_low": worst_low,
                "worst_task_delta_ci_high": worst_high,
                "mean_delta_test_average_accuracy_vs_best_single_task_vector_method": safe_mean(delta_task),
                "wins_vs_original_greedy": int((delta > 1e-12).sum()),
                "ties_vs_original_greedy": int((delta.abs() <= 1e-12).sum()),
                "losses_vs_original_greedy": int((delta < -1e-12).sum()),
                "selected_family_frequency_json": compact_json(family_counts),
                "test_leakage_violations": int(group["test_used_for_selection"].astype(bool).sum()),
                "claim_status": claim,
                "claim_supported": bool(claim == "supported_exact_setting_enriched_pool_improves_original_greedy"),
                "claim_boundary": "exact same-base setting only; generated-candidate validation selector, not broad model-merging superiority",
            }
        )
    fixed = pd.DataFrame(records)
    overall_rows = []
    for selector, group in ok.groupby("selector", dropna=False, sort=True):
        delta = pd.to_numeric(group["delta_test_average_accuracy_vs_original_greedy"], errors="coerce")
        low, high = bootstrap_mean_ci(delta, bootstrap_samples, seed=9901 + len(overall_rows) * 23)
        if selector == BASELINE_SELECTOR:
            claim = "baseline_original_greedy_soup"
        elif int(group["task_preset"].nunique()) >= 2 and int(group["seed"].nunique()) >= 20 and safe_float(low) > 0.0:
            claim = "supported_multi_setting_enriched_pool_improves_original_greedy"
        else:
            claim = "descriptive_overall_not_broad_superiority"
        overall_rows.append(
            {
                "dataset": "ALL",
                "task_preset": "ALL",
                "architecture": "mlp2",
                "width": -1,
                "n_tasks": -1,
                "selector": selector,
                "n_rows": int(len(group)),
                "n_unique_seeds": int(group["seed"].nunique()),
                "mean_validation_average_accuracy": safe_mean(group["validation_average_accuracy"]),
                "mean_validation_worst_task_accuracy": safe_mean(group["validation_worst_task_accuracy"]),
                "mean_test_average_accuracy": safe_mean(group["test_average_accuracy"]),
                "mean_test_worst_task_accuracy": safe_mean(group["test_worst_task_accuracy"]),
                "mean_delta_test_average_accuracy_vs_original_greedy": safe_mean(delta),
                "delta_vs_original_greedy_ci_low": low,
                "delta_vs_original_greedy_ci_high": high,
                "mean_delta_test_worst_task_accuracy_vs_original_greedy": safe_mean(group["delta_test_worst_task_accuracy_vs_original_greedy"]),
                "worst_task_delta_ci_low": float("nan"),
                "worst_task_delta_ci_high": float("nan"),
                "mean_delta_test_average_accuracy_vs_best_single_task_vector_method": safe_mean(group["delta_test_average_accuracy_vs_best_single_task_vector_method"]),
                "wins_vs_original_greedy": int((delta > 1e-12).sum()),
                "ties_vs_original_greedy": int((delta.abs() <= 1e-12).sum()),
                "losses_vs_original_greedy": int((delta < -1e-12).sum()),
                "selected_family_frequency_json": compact_json(group["selected_candidate_family"].value_counts().to_dict()),
                "test_leakage_violations": int(group["test_used_for_selection"].astype(bool).sum()),
                "claim_status": claim,
                "claim_supported": bool(claim == "supported_multi_setting_enriched_pool_improves_original_greedy"),
                "claim_boundary": "overall diagnostic across same-base settings; not broad model-merging superiority",
            }
        )
    return pd.concat([fixed, pd.DataFrame(overall_rows)], ignore_index=True)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in columns:
        if col not in view.columns:
            view[col] = ""
    return format_markdown_table(view[columns].to_dict("records"), columns)


def plot_deltas(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary[
        (summary["dataset"].astype(str) != "ALL")
        & (summary["selector"].astype(str) != BASELINE_SELECTOR)
    ].copy()
    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.28 * max(len(plot_df), 1))))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No selector delta rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot_df = plot_df.sort_values(["dataset", "task_preset", "width", "mean_delta_test_average_accuracy_vs_original_greedy"])
        y = np.arange(len(plot_df))
        x = pd.to_numeric(plot_df["mean_delta_test_average_accuracy_vs_original_greedy"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(plot_df["delta_vs_original_greedy_ci_low"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(plot_df["delta_vs_original_greedy_ci_high"], errors="coerce").to_numpy(dtype=float)
        err = np.vstack([np.maximum(x - lo, 0.0), np.maximum(hi - x, 0.0)])
        labels = [
            f"{row.dataset}/{row.task_preset}/W{int(row.width)}\n{row.selector}"
            for row in plot_df.itertuples(index=False)
        ]
        colors = ["tab:green" if low > 0.0 else "tab:blue" for low in lo]
        ax.barh(y, x, color=colors, alpha=0.78)
        ax.errorbar(x, y, xerr=err, fmt="none", ecolor="black", capsize=2, linewidth=0.7)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("mean test average accuracy delta vs original greedy soup")
        ax.set_title("Descent envelope selectors over generated same-base candidates")
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_latex(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary[
        (summary["dataset"].astype(str) != "ALL")
        & (summary["selector"].isin(["best_validation_generated_candidate", "greedy_soup_over_generated_candidates", "conservative_lcb_generated_candidate", "worst_task_validation_selector"]))
    ].copy()
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Setting & Selector & Seeds & Test acc. & Worst acc. & $\Delta$ greedy & CI low \\",
        r"\midrule",
    ]
    for row in rows.sort_values(["dataset", "task_preset", "width", "selector"]).itertuples(index=False):
        setting = f"{row.dataset}/{row.task_preset}/W{int(row.width)}".replace("_", "\\_")
        selector = str(row.selector).replace("_", "\\_")
        lines.append(
            f"{setting} & {selector} & {int(row.n_unique_seeds)} & "
            f"{row.mean_test_average_accuracy:.4f} & {row.mean_test_worst_task_accuracy:.4f} & "
            f"{row.mean_delta_test_average_accuracy_vs_original_greedy:.4f} & "
            f"{row.delta_vs_original_greedy_ci_low:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(args: argparse.Namespace, rows: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    overall = summary[(summary["dataset"].astype(str) == "ALL") & (summary["selector"].astype(str) != BASELINE_SELECTOR)].copy()
    supported = summary[summary["claim_supported"] == True]  # noqa: E712
    if overall.empty:
        headline = "No completed selector rows."
    else:
        best = overall.sort_values("mean_delta_test_average_accuracy_vs_original_greedy", ascending=False).iloc[0]
        headline = (
            f"Best overall selector by mean delta is `{best['selector']}` with "
            f"delta `{best['mean_delta_test_average_accuracy_vs_original_greedy']:.4f}` "
            f"and CI [`{best['delta_vs_original_greedy_ci_low']:.4f}`, `{best['delta_vs_original_greedy_ci_high']:.4f}`]."
        )
        if supported.empty:
            headline += " No broad model-merging superiority claim is supported."
        else:
            headline += f" `{len(supported)}` exact-setting selector rows pass the positive CI gate."
    leakage = int(rows.get("test_used_for_selection", pd.Series(dtype=bool)).astype(bool).sum()) if not rows.empty else 0
    report = f"""# Descent Envelope Selector

Generated by `experiments/descent_envelope_selector.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

This is a same-base task-vector validation-descent envelope. It treats SLERP, Task Arithmetic, TIES, DARE, weight averaging, base/fine-tuned models, and the original greedy soup as generated candidates. Selectors use validation metrics only; test metrics are evaluation-only after each selector is frozen.

This is not an independent-seed rebasin benchmark and does not write paper prose.

## Headline

{headline}

## Outputs

- `reports/csv/{RUN_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/{REPORT_MD}`
- `reports/plots/{PLOT_PDF}`
- `reports/tables/{TABLE_TEX}`

## Selector Summary

{md_table(summary, ["dataset", "task_preset", "width", "selector", "n_rows", "n_unique_seeds", "mean_validation_average_accuracy", "mean_test_average_accuracy", "mean_test_worst_task_accuracy", "mean_delta_test_average_accuracy_vs_original_greedy", "delta_vs_original_greedy_ci_low", "delta_vs_original_greedy_ci_high", "mean_delta_test_average_accuracy_vs_best_single_task_vector_method", "claim_status"], 100)}

## Selection Frequencies

{md_table(summary[summary["dataset"].astype(str) != "ALL"], ["dataset", "task_preset", "width", "selector", "selected_family_frequency_json", "wins_vs_original_greedy", "ties_vs_original_greedy", "losses_vs_original_greedy"], 80)}

## Test Leakage Audit

- Selector rows with `test_used_for_selection=True`: `{leakage}`
- Candidate selection rules use validation average accuracy, validation worst-task accuracy, or validation lower confidence bounds.
- Delta columns compare held-out test metrics after selection; they are not selector inputs.

## Claim Boundary

- This report may support exact-setting statements about whether an enriched generated-candidate pool improves validation descent in the tested same-base settings.
- It must not be cited as broad model-merging superiority unless paired CIs and multiple task families support the exact statement.
- Git-ReBasin/C2M3 independent-seed claims are out of scope for this same-base task-vector envelope.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def update_claims_audit(summary: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        return
    exact_supported = summary[(summary["dataset"].astype(str) != "ALL") & (summary["claim_supported"] == True)]  # noqa: E712
    overall_supported = summary[(summary["dataset"].astype(str) == "ALL") & (summary["claim_supported"] == True)]  # noqa: E712
    if exact_supported.empty:
        status = "Supported descriptive"
        evidence = "`reports/descent_envelope_selector_report.md` records the same-base descent-envelope selector; no exact-setting positive CI gate passed."
    else:
        status = "Supported exact-setting"
        best = exact_supported.sort_values("mean_delta_test_average_accuracy_vs_original_greedy", ascending=False).iloc[0]
        evidence = (
            "`reports/descent_envelope_selector_report.md` records validation-only generated-candidate selectors; "
            f"{len(exact_supported)} exact-setting selector rows pass positive paired CIs; strongest row "
            f"`{best['selector']}` on `{best['dataset']}/{best['task_preset']}/W{int(best['width'])}` "
            f"has mean delta `{best['mean_delta_test_average_accuracy_vs_original_greedy']:.4f}`. "
            "No broad superiority claim is made."
        )
    if not overall_supported.empty:
        evidence += " Overall multi-setting support is also flagged descriptively in the selector summary."
    audit_row = f"| Enriched generated-candidate descent envelopes are validation-only same-base selectors. | {status} | {evidence} |"
    text = path.read_text(encoding="utf-8")
    marker = "Enriched generated-candidate descent envelopes are validation-only same-base selectors."
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if marker in line:
            lines[idx] = audit_row
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    insert_marker = "<!-- prompt10-claim-audit:start -->"
    if insert_marker in text:
        text = text.replace(insert_marker, audit_row + "\n\n" + insert_marker)
    else:
        text = text.rstrip() + "\n\n" + audit_row + "\n"
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", type=Path, default=ROOT / "reports" / "csv" / "same_base_task_vector_benchmark.csv")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--task-presets", default="")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-runs-per-setting", type=int, default=0)
    parser.add_argument("--slerp-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--task-arithmetic-scales", default="0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--ties-densities", default="0.2,0.5,1.0")
    parser.add_argument("--ties-scales", default="0.5,1.0,1.25")
    parser.add_argument("--dare-drop-rates", default="0.1,0.3,0.5")
    parser.add_argument("--dare-scales", default="0.5,1.0,1.25")
    parser.add_argument("--lcb-z", type=float, default=1.96)
    parser.add_argument("--lcb-margin", type=float, default=0.0)
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--augmentation", default="none")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--update-claims-audit", action="store_true", default=True)
    parser.add_argument("--no-update-claims-audit", action="store_false", dest="update_claims_audit")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    return args


def main() -> None:
    args = parse_args()
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    table_dir = args.reports_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = run_rows(args)
    summary = summarize(rows, int(args.bootstrap_samples))
    rows.to_csv(csv_dir / RUN_CSV, index=False, lineterminator="\n")
    summary.to_csv(csv_dir / SUMMARY_CSV, index=False, lineterminator="\n")
    plot_deltas(summary, plot_dir / PLOT_PDF)
    write_latex(summary, table_dir / TABLE_TEX)
    write_report(args, rows, summary, args.reports_dir / REPORT_MD)
    if args.update_claims_audit:
        update_claims_audit(summary, args.reports_dir / "claims_audit.md")
    print(f"wrote {csv_dir / RUN_CSV}")
    print(f"wrote {csv_dir / SUMMARY_CSV}")
    print(f"wrote {args.reports_dir / REPORT_MD}")
    print(f"wrote {plot_dir / PLOT_PDF}")
    print(f"wrote {table_dir / TABLE_TEX}")


if __name__ == "__main__":
    main()
