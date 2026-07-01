#!/usr/bin/env python
"""Greedy soup over an expanded family of gauge-corrected candidates.

The experiment treats TwistedMerge-style gauges as candidate-family enrichment:
the selector is still greedy validation descent, but the candidate pool includes
ordinary local sections plus permutation- and monomial-gauge corrected sections.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.model_merging_fixed_setting_verification import (  # noqa: E402
    layer_reference_perms,
    permutation_arg_for_architecture,
    split_indices,
    synchronize_alignment_bundle,
    synced_layer_perms,
)
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    clone_model,
    compute_layerwise_pairwise_permutations,
    device_from_arg,
    evaluate_model,
    format_markdown_table,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
)
from src.monomial_gauge_alignment import (  # noqa: E402
    apply_monomial_alignment_to_reference,
    average_monomial_defect_score,
    estimate_pairwise_monomial_alignments,
    monomial_scaling_statistics,
)


RUNS_CSV = "expanded_candidate_soup.csv"
SUMMARY_CSV = "expanded_candidate_soup_summary.csv"
TRAJECTORY_CSV = "expanded_candidate_soup_trajectory.csv"
REPORT_MD = "expanded_candidate_soup_report.md"
PLOT_PDF = "expanded_candidate_soup_deltas.pdf"
TOL = 1e-12


@dataclass
class Candidate:
    model: object
    label: str
    candidate_type: str
    candidate_family: str
    source_model_index: int | None
    exact_relu_symmetry: bool
    is_extra_capacity: bool = False
    parameter_multiplier: float = 1.0
    inference_multiplier: float = 1.0
    capacity_matched_to_weight_average: bool = True
    note: str = ""


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


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_checkpoint_name(path: Path) -> tuple[int, int] | None:
    match = re.fullmatch(r"seed(\d+)_model(\d+)\.pt", path.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def checkpoint_groups(
    checkpoint_root: Path,
    datasets: set[str] | None,
    settings: set[str] | None,
    seeds: set[int] | None,
    max_seeds_per_setting: int,
) -> tuple[list[dict], list[dict]]:
    torch, _, _ = require_torch()
    groups: list[dict] = []
    skipped: list[dict] = []
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint root not found: {checkpoint_root}")
    for setting_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        if settings is not None and setting_dir.name not in settings:
            continue
        by_seed: dict[int, dict[int, Path]] = {}
        for path in sorted(setting_dir.glob("seed*_model*.pt")):
            parsed = parse_checkpoint_name(path)
            if parsed is None:
                continue
            seed, model_index = parsed
            if seeds is not None and seed not in seeds:
                continue
            by_seed.setdefault(seed, {})[model_index] = path
        selected_seeds = sorted(by_seed)
        if int(max_seeds_per_setting) > 0:
            selected_seeds = selected_seeds[: int(max_seeds_per_setting)]
        for seed in selected_seeds:
            model_paths = by_seed[seed]
            first_path = model_paths[min(model_paths)]
            payload = torch.load(first_path, map_location="cpu", weights_only=False)
            metadata = dict(payload.get("metadata", {}))
            if datasets is not None and str(metadata.get("dataset")) not in datasets:
                continue
            n_models = int(metadata["n_models"])
            missing = [idx for idx in range(n_models) if idx not in model_paths]
            record = {
                "setting_id": setting_dir.name,
                "seed": int(seed),
                "model_paths": model_paths,
                "metadata": metadata,
                "complete": not missing,
                "missing_model_indices": missing,
            }
            if missing:
                skipped.append(
                    {
                        "setting_id": setting_dir.name,
                        "seed": int(seed),
                        "skip_reason": "missing_checkpoint_file",
                        "missing_model_indices": json.dumps(missing, separators=(",", ":")),
                    }
                )
            else:
                groups.append(record)
    return groups, skipped


def checkpoint_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    return path


def load_eval_loaders(metadata: dict, args: argparse.Namespace, cache: dict):
    torch, _, _ = require_torch()
    key = (
        metadata["dataset"],
        int(metadata.get("max_train_samples", args.max_train_samples)),
        int(metadata.get("max_test_samples", args.max_test_samples)),
        int(metadata.get("dataset_seed", args.dataset_seed)),
        str(metadata.get("augmentation", args.augmentation)),
        int(metadata.get("train_split_seed", int(metadata.get("dataset_seed", args.dataset_seed)) + 17)),
        float(args.val_fraction),
        int(args.batch_size),
    )
    if key in cache:
        return cache[key]
    spec, train_base, test_base = load_dataset(
        key[0],
        args.data_dir,
        key[1],
        key[2],
        key[3],
        augmentation=key[4],
    )
    _train_indices, val_indices = split_indices(len(train_base), key[6], key[5])
    val_subset = torch.utils.data.Subset(train_base, val_indices)
    val_loader = make_loader(val_subset, key[7], shuffle=False, seed=key[3] + 100)
    test_loader = make_loader(test_base, key[7], shuffle=False, seed=key[3] + 200)
    match_loader = make_loader(val_subset, key[7], shuffle=False, seed=key[3] + 300)
    cache[key] = (spec, val_loader, test_loader, match_loader, len(val_subset), len(test_base))
    return cache[key]


def load_models(group: dict, spec, device) -> list:
    torch, _, _ = require_torch()
    metadata = group["metadata"]
    architecture = str(metadata["architecture"])
    width = int(metadata["width"])
    models = []
    for model_index in range(int(metadata["n_models"])):
        payload = torch.load(group["model_paths"][model_index], map_location="cpu", weights_only=False)
        model = make_model(architecture, spec, width)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def json_compact(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def add_candidate(
    candidates: list[Candidate],
    model,
    *,
    label: str,
    candidate_type: str,
    candidate_family: str,
    source_model_index: int | None,
    exact_relu_symmetry: bool,
    note: str,
) -> None:
    candidates.append(
        Candidate(
            model=model,
            label=label,
            candidate_type=candidate_type,
            candidate_family=candidate_family,
            source_model_index=source_model_index,
            exact_relu_symmetry=exact_relu_symmetry,
            note=note,
        )
    )


def build_monomial_candidates(
    args: argparse.Namespace,
    models: list,
    spec,
    width: int,
    match_loader,
    device,
    *,
    matching: str,
    scale_method: str,
    family: str,
    note: str,
) -> tuple[list[Candidate], dict]:
    alignments = estimate_pairwise_monomial_alignments(
        models,
        match_loader,
        device,
        matching=matching,
        max_batches=int(args.feature_batches),
        scale_method=scale_method,
        log_scale_clip=float(args.monomial_log_scale_clip),
        shrinkage=float(args.monomial_shrinkage),
        activation_similarity_threshold=float(args.monomial_activation_similarity_threshold),
    )
    stats = monomial_scaling_statistics(alignments)
    score, _triangle_rows = average_monomial_defect_score(alignments, len(models))
    stats["monomial_average_defect_score"] = score
    out: list[Candidate] = []
    aligned = [models[0]]
    for idx in range(1, len(models)):
        aligned_model = apply_monomial_alignment_to_reference(models[idx], spec, width, alignments[(0, idx)])
        aligned.append(aligned_model)
        add_candidate(
            out,
            aligned_model,
            label=f"{family}:section:{idx}",
            candidate_type="gauge_corrected_local_section",
            candidate_family=family,
            source_model_index=idx,
            exact_relu_symmetry=True,
            note=note,
        )
    merged = average_models(aligned, "mlp2", spec, width)
    add_candidate(
        out,
        merged,
        label=f"{family}:average",
        candidate_type="gauge_corrected_average",
        candidate_family=family,
        source_model_index=None,
        exact_relu_symmetry=True,
        note=f"{note}; averaged gauge-corrected sections",
    )
    return out, stats


def build_candidates(group: dict, args: argparse.Namespace, spec, match_loader, device) -> tuple[list[Candidate], list[dict]]:
    metadata = group["metadata"]
    architecture = str(metadata["architecture"])
    width = int(metadata["width"])
    if architecture != "mlp2":
        raise ValueError(f"expanded candidate soup currently expects mlp2 checkpoints, got {architecture}")
    models = load_models(group, spec, device)
    candidates: list[Candidate] = []
    diagnostics: list[dict] = []
    for idx, model in enumerate(models):
        add_candidate(
            candidates,
            model,
            label=f"local:{idx}",
            candidate_type="local_individual_checkpoint",
            candidate_family="local",
            source_model_index=idx,
            exact_relu_symmetry=True,
            note="ordinary local section checkpoint",
        )

    weight_average = average_models(models, architecture, spec, width)
    add_candidate(
        candidates,
        weight_average,
        label="weight_average",
        candidate_type="ordinary_weight_average",
        candidate_family="weight_average",
        source_model_index=None,
        exact_relu_symmetry=False,
        note="ordinary parameter average candidate",
    )

    pairwise_by_layer = compute_layerwise_pairwise_permutations(
        models,
        architecture,
        match_loader,
        device,
        method="activation",
    )
    git_aligned = [models[0]]
    for idx in range(1, len(models)):
        aligned = permute_model_to_reference(
            models[idx],
            architecture,
            spec,
            width,
            permutation_arg_for_architecture(
                architecture,
                layer_reference_perms(pairwise_by_layer, 0, idx),
            ),
        )
        git_aligned.append(aligned)
        add_candidate(
            candidates,
            aligned,
            label=f"git_rebasin_ref0:section:{idx}",
            candidate_type="gauge_corrected_local_section",
            candidate_family="git_rebasin_ref0",
            source_model_index=idx,
            exact_relu_symmetry=True,
            note="activation-matched Git-ReBasin-style section aligned to model 0",
        )
    git_average = average_models(git_aligned, architecture, spec, width)
    add_candidate(
        candidates,
        git_average,
        label="git_rebasin_ref0:average",
        candidate_type="gauge_corrected_average",
        candidate_family="git_rebasin_ref0",
        source_model_index=None,
        exact_relu_symmetry=True,
        note="average after pairwise alignment to model 0",
    )

    sync_ref, synced_by_layer, sync_disagreement = synchronize_alignment_bundle(pairwise_by_layer, len(models))
    c2m3_aligned = []
    for idx in range(len(models)):
        aligned = permute_model_to_reference(
            models[idx],
            architecture,
            spec,
            width,
            permutation_arg_for_architecture(architecture, synced_layer_perms(synced_by_layer, idx)),
        )
        c2m3_aligned.append(aligned)
        add_candidate(
            candidates,
            aligned,
            label=f"c2m3_synchronized:section:{idx}",
            candidate_type="gauge_corrected_local_section",
            candidate_family="c2m3_synchronized",
            source_model_index=idx,
            exact_relu_symmetry=True,
            note=f"cycle-synchronized permutation section; sync_ref={sync_ref}",
        )
    c2m3_average = average_models(c2m3_aligned, architecture, spec, width)
    add_candidate(
        candidates,
        c2m3_average,
        label="c2m3_synchronized:average",
        candidate_type="gauge_corrected_average",
        candidate_family="c2m3_synchronized",
        source_model_index=None,
        exact_relu_symmetry=True,
        note=f"average after C2M3-style synchronization; sync_disagreement={sync_disagreement:.6g}",
    )
    diagnostics.append(
        {
            "diagnostic": "c2m3_sync_disagreement",
            "value": float(sync_disagreement),
            "note": sync_ref,
        }
    )

    monomial_specs = [
        (
            "monomial_activation_raw",
            "monomial_activation_mlp2",
            "raw",
            "activation-matched positive monomial gauge aligned to model 0",
        ),
        (
            "monomial_activation_shrinkage",
            "monomial_activation_mlp2",
            "shrinkage",
            "activation-matched positive monomial gauge with shrinkage regularization",
        ),
        (
            "monomial_activation_global",
            "monomial_activation_mlp2",
            "global_synchronized",
            "activation-matched positive monomial gauge with global log-scale synchronization",
        ),
        (
            "monomial_weight_raw",
            "monomial_weight_mlp2",
            "raw",
            "weight-matched positive monomial gauge aligned to model 0",
        ),
    ]
    for family, matching, scale_method, note in monomial_specs:
        try:
            mono_candidates, mono_stats = build_monomial_candidates(
                args,
                models,
                spec,
                width,
                match_loader,
                device,
                matching=matching,
                scale_method=scale_method,
                family=family,
                note=note,
            )
            candidates.extend(mono_candidates)
            for key, value in mono_stats.items():
                diagnostics.append(
                    {
                        "diagnostic": f"{family}_{key}",
                        "value": float(value) if isinstance(value, (int, float, np.floating)) else value,
                        "note": "",
                    }
                )
        except Exception as exc:
            diagnostics.append({"diagnostic": f"{family}_build_failed", "value": float("nan"), "note": repr(exc)})
    return candidates, diagnostics


def candidate_metadata(candidate: Candidate, idx: int) -> dict:
    return {
        "candidate_index": idx,
        "candidate_label": candidate.label,
        "candidate_type": candidate.candidate_type,
        "candidate_family": candidate.candidate_family,
        "candidate_source_model_index": "" if candidate.source_model_index is None else int(candidate.source_model_index),
        "candidate_exact_relu_symmetry": bool(candidate.exact_relu_symmetry),
        "candidate_is_extra_capacity": bool(candidate.is_extra_capacity),
        "candidate_parameter_multiplier": float(candidate.parameter_multiplier),
        "candidate_inference_multiplier": float(candidate.inference_multiplier),
        "candidate_capacity_matched_to_weight_average": bool(candidate.capacity_matched_to_weight_average),
        "candidate_note": candidate.note,
    }


def run_candidate_soup(
    candidates: list[Candidate],
    val_loader,
    test_loader,
    device,
    architecture: str,
    spec,
    width: int,
    *,
    run_id: str,
    candidate_set_name: str,
) -> tuple[object, list[int], dict, dict, list[dict]]:
    val_metrics: list[dict] = []
    for idx, candidate in enumerate(candidates):
        metrics = evaluate_model(candidate.model, val_loader, device)
        val_metrics.append(metrics)
    scored = [(float(metrics["accuracy"]), idx) for idx, metrics in enumerate(val_metrics)]
    order = [idx for _acc, idx in sorted(scored, reverse=True)]
    selected = [order[0]]
    soup = clone_model(candidates[order[0]].model, architecture, spec, width)
    best_acc = float(val_metrics[order[0]]["accuracy"])
    best_loss = float(val_metrics[order[0]]["loss"])
    trajectory: list[dict] = []
    last_accepted_row = 0
    first = candidates[order[0]]
    trajectory.append(
        {
            "run_id": run_id,
            "candidate_set": candidate_set_name,
            "candidate_rank": 1,
            "candidate_order": json_compact(order),
            "soup_indices_before": json_compact([]),
            "soup_indices_after": json_compact(selected),
            "validation_accuracy_before": float("nan"),
            "validation_loss_before": float("nan"),
            "candidate_soup_validation_accuracy": best_acc,
            "candidate_soup_validation_loss": best_loss,
            "validation_accuracy_margin_after_minus_before": float("nan"),
            "validation_loss_margin_before_minus_after": float("nan"),
            "accepted": True,
            "decision_reason": "accepted_initial_best_validation_candidate",
            "decision_metric": "validation_accuracy",
            "decision_metric_source": "direct_candidate_soup_validation_metric",
            "test_used_for_selection": False,
            "is_final_selection": False,
            **candidate_metadata(first, order[0]),
        }
    )
    for candidate_rank, idx in enumerate(order[1:], start=2):
        before_selected = list(selected)
        before_acc = best_acc
        before_loss = best_loss
        candidate_indices = selected + [idx]
        candidate_soup = average_models([candidates[item].model for item in candidate_indices], architecture, spec, width)
        candidate_metrics = evaluate_model(candidate_soup, val_loader, device)
        candidate_acc = float(candidate_metrics["accuracy"])
        candidate_loss = float(candidate_metrics["loss"])
        accepted = bool(candidate_acc >= best_acc)
        if accepted:
            soup = candidate_soup
            selected = candidate_indices
            best_acc = candidate_acc
            best_loss = candidate_loss
            last_accepted_row = len(trajectory)
        trajectory.append(
            {
                "run_id": run_id,
                "candidate_set": candidate_set_name,
                "candidate_rank": int(candidate_rank),
                "candidate_order": json_compact(order),
                "soup_indices_before": json_compact(before_selected),
                "soup_indices_after": json_compact(selected),
                "validation_accuracy_before": before_acc,
                "validation_loss_before": before_loss,
                "candidate_soup_validation_accuracy": candidate_acc,
                "candidate_soup_validation_loss": candidate_loss,
                "validation_accuracy_margin_after_minus_before": float(candidate_acc - before_acc),
                "validation_loss_margin_before_minus_after": float(before_loss - candidate_loss),
                "accepted": accepted,
                "decision_reason": (
                    "accepted_validation_accuracy_non_decrease"
                    if accepted
                    else "rejected_validation_accuracy_decrease"
                ),
                "decision_metric": "validation_accuracy",
                "decision_metric_source": "direct_candidate_soup_validation_metric",
                "test_used_for_selection": False,
                "is_final_selection": False,
                **candidate_metadata(candidates[idx], idx),
            }
        )
    final_val = {"accuracy": best_acc, "loss": best_loss}
    final_test = evaluate_model(soup, test_loader, device)
    trajectory[last_accepted_row]["is_final_selection"] = True
    trajectory[last_accepted_row]["final_test_accuracy"] = float(final_test["accuracy"])
    trajectory[last_accepted_row]["final_test_loss"] = float(final_test["loss"])
    trajectory[last_accepted_row]["test_metric_role"] = "evaluation_only_final_selection"
    return soup, selected, final_val, final_test, trajectory


def safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or samples <= 0:
        mean = float(arr.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(int(samples), arr.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def final_row_from_trajectory(traj: list[dict]) -> dict:
    finals = [row for row in traj if bool(row.get("is_final_selection", False))]
    return finals[-1] if finals else traj[-1]


def run_group(group: dict, args: argparse.Namespace, loader_cache: dict) -> tuple[list[dict], list[dict], list[dict]]:
    metadata = group["metadata"]
    device = device_from_arg(args.device)
    spec, val_loader, test_loader, match_loader, n_val, n_test = load_eval_loaders(metadata, args, loader_cache)
    architecture = str(metadata["architecture"])
    width = int(metadata["width"])
    setting_id = str(group["setting_id"])
    seed = int(group["seed"])
    run_id = f"{setting_id}_seed{seed}"
    candidates, diagnostics = build_candidates(group, args, spec, match_loader, device)
    local_count = int(metadata["n_models"])
    local_candidates = candidates[:local_count]
    _ordinary_model, ordinary_selected, ordinary_val, ordinary_test, ordinary_traj = run_candidate_soup(
        local_candidates,
        val_loader,
        test_loader,
        device,
        architecture,
        spec,
        width,
        run_id=run_id,
        candidate_set_name="ordinary_local_greedy_soup",
    )
    _expanded_model, expanded_selected, expanded_val, expanded_test, expanded_traj = run_candidate_soup(
        candidates,
        val_loader,
        test_loader,
        device,
        architecture,
        spec,
        width,
        run_id=run_id,
        candidate_set_name="expanded_gauge_candidate_soup",
    )
    base = {
        "setting_id": setting_id,
        "run_id": run_id,
        "dataset": str(metadata["dataset"]),
        "architecture": architecture,
        "n_models": int(metadata["n_models"]),
        "width": width,
        "domain_shift": str(metadata["domain_shift"]),
        "matching": str(metadata.get("matching", "activation")),
        "seed": seed,
        "max_train_samples": int(metadata.get("max_train_samples", args.max_train_samples)),
        "max_test_samples": int(metadata.get("max_test_samples", args.max_test_samples)),
        "dataset_seed": int(metadata.get("dataset_seed", args.dataset_seed)),
        "val_fraction": float(args.val_fraction),
        "n_validation_examples": int(n_val),
        "n_test_examples": int(n_test),
        "checkpoint_source": str(group["model_paths"][0].parent),
    }
    ordinary_final = final_row_from_trajectory(ordinary_traj)
    expanded_final = final_row_from_trajectory(expanded_traj)
    final_rows = []
    for method, selected, val, test, final, n_candidates in [
        ("ordinary_greedy_soup", ordinary_selected, ordinary_val, ordinary_test, ordinary_final, len(local_candidates)),
        ("expanded_candidate_soup", expanded_selected, expanded_val, expanded_test, expanded_final, len(candidates)),
    ]:
        selected_labels = [
            (local_candidates if method == "ordinary_greedy_soup" else candidates)[idx].label for idx in selected
        ]
        selected_families = [
            (local_candidates if method == "ordinary_greedy_soup" else candidates)[idx].candidate_family for idx in selected
        ]
        final_rows.append(
            {
                **base,
                "method": method,
                "candidate_set": "ordinary_local_only" if method == "ordinary_greedy_soup" else "expanded_gauge_candidates",
                "n_candidates": int(n_candidates),
                "selected_candidate_indices_json": json_compact(selected),
                "selected_candidate_labels_json": json_compact(selected_labels),
                "selected_candidate_families_json": json_compact(selected_families),
                "selected_candidate_count": int(len(selected)),
                "selected_extra_capacity_count": int(
                    sum((local_candidates if method == "ordinary_greedy_soup" else candidates)[idx].is_extra_capacity for idx in selected)
                ),
                "selected_nonlocal_candidate_count": int(sum(family != "local" for family in selected_families)),
                "final_selected_candidate_label": str(final.get("candidate_label", "")),
                "final_selected_candidate_family": str(final.get("candidate_family", "")),
                "final_selected_candidate_type": str(final.get("candidate_type", "")),
                "val_accuracy": float(val["accuracy"]),
                "val_loss": float(val["loss"]),
                "test_accuracy": float(test["accuracy"]),
                "test_loss": float(test["loss"]),
                "test_used_for_selection": False,
                "is_single_model": True,
                "is_ensemble_or_extra_capacity": False,
                "capacity_matched_to_weight_average": True,
                "parameter_multiplier": 1.0,
                "inference_multiplier": 1.0,
                "delta_test_accuracy_vs_ordinary_greedy": float(test["accuracy"] - ordinary_test["accuracy"]),
                "delta_val_accuracy_vs_ordinary_greedy": float(val["accuracy"] - ordinary_val["accuracy"]),
                "ordinary_greedy_test_accuracy": float(ordinary_test["accuracy"]),
                "ordinary_greedy_val_accuracy": float(ordinary_val["accuracy"]),
            }
        )
    traj_rows = []
    for item in ordinary_traj + expanded_traj:
        row = {**base, **item}
        if "final_test_accuracy" not in row:
            row["final_test_accuracy"] = float("nan")
            row["final_test_loss"] = float("nan")
            row["test_metric_role"] = "not_evaluated_for_candidate_decision"
        traj_rows.append(row)
    diag_rows = [{**base, **item} for item in diagnostics]
    for candidate in candidates:
        candidate.model.to("cpu")
    return final_rows, traj_rows, diag_rows


def paired_summary(finals: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows: list[dict] = []
    pivot = finals.pivot_table(
        index=["setting_id", "run_id", "dataset", "architecture", "n_models", "width", "domain_shift", "matching", "seed"],
        columns="method",
        values=["test_accuracy", "val_accuracy", "selected_nonlocal_candidate_count"],
        aggfunc="first",
    )
    if pivot.empty:
        return pd.DataFrame()
    flat = pivot.copy()
    flat.columns = [f"{a}__{b}" for a, b in flat.columns]
    flat = flat.reset_index()
    flat["test_delta"] = flat["test_accuracy__expanded_candidate_soup"] - flat["test_accuracy__ordinary_greedy_soup"]
    flat["val_delta"] = flat["val_accuracy__expanded_candidate_soup"] - flat["val_accuracy__ordinary_greedy_soup"]
    flat["expanded_selected_nonlocal"] = pd.to_numeric(
        flat.get("selected_nonlocal_candidate_count__expanded_candidate_soup"), errors="coerce"
    ).fillna(0)
    group_cols = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
    grouped_scopes: list[tuple[str, pd.DataFrame, tuple | None]] = [("overall", flat, None)]
    grouped_scopes.extend(
        ("fixed_setting", group, key if isinstance(key, tuple) else (key,))
        for key, group in flat.groupby(group_cols, dropna=False)
    )
    for scope, group, key_tuple in grouped_scopes:
        if scope == "overall":
            meta = {
                "summary_type": "paired_delta",
                "scope": "overall",
                "dataset": "ALL",
                "architecture": "mlp2",
                "n_models": "",
                "width": 128,
                "domain_shift": "ALL",
                "matching": "activation",
            }
        else:
            meta = dict(zip(group_cols, key_tuple, strict=False))
            meta.update({"summary_type": "paired_delta", "scope": "fixed_setting"})
        deltas = pd.to_numeric(group["test_delta"], errors="coerce").to_numpy(dtype=float)
        val_deltas = pd.to_numeric(group["val_delta"], errors="coerce").to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(deltas, bootstrap_samples, seed=27027 + len(rows) * 101)
        mean_delta = float(np.nanmean(deltas)) if len(deltas) else float("nan")
        if math.isfinite(low) and low > 0.0:
            claim_status = "supported_positive_expanded_candidate_soup_over_ordinary_greedy"
        elif mean_delta > 0.0:
            claim_status = "descriptive_positive_ci_crosses_zero"
        else:
            claim_status = "no_improvement_empirical_descent_rejects_or_neutralizes_extra_candidates"
        rows.append(
            {
                **meta,
                "n_runs": int(len(group)),
                "n_unique_seeds": int(group["seed"].nunique()),
                "mean_ordinary_greedy_test_accuracy": float(group["test_accuracy__ordinary_greedy_soup"].mean()),
                "mean_expanded_soup_test_accuracy": float(group["test_accuracy__expanded_candidate_soup"].mean()),
                "mean_test_delta_expanded_minus_ordinary": mean_delta,
                "test_delta_ci_low": low,
                "test_delta_ci_high": high,
                "mean_val_delta_expanded_minus_ordinary": float(np.nanmean(val_deltas)) if len(val_deltas) else float("nan"),
                "wins": int(np.sum(deltas > TOL)),
                "ties": int(np.sum(np.abs(deltas) <= TOL)),
                "losses": int(np.sum(deltas < -TOL)),
                "expanded_selected_nonlocal_fraction": float((group["expanded_selected_nonlocal"] > 0).mean()),
                "claim_status": claim_status,
                "claim_supported": bool(math.isfinite(low) and low > 0.0),
            }
        )
    return pd.DataFrame(rows)


def selected_frequency_summary(finals: pd.DataFrame) -> pd.DataFrame:
    expanded = finals[finals["method"] == "expanded_candidate_soup"].copy()
    rows = []
    group_cols = ["dataset", "architecture", "n_models", "width", "domain_shift", "matching"]
    for key, group in expanded.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key, strict=False))
        total_runs = int(len(group))
        counts: dict[str, int] = {}
        run_counts: dict[str, int] = {}
        selected_total = 0
        for payload in group["selected_candidate_families_json"]:
            families = json.loads(payload)
            selected_total += len(families)
            seen_families = set(families)
            for family in seen_families:
                run_counts[family] = run_counts.get(family, 0) + 1
            for family in families:
                counts[family] = counts.get(family, 0) + 1
        for family, count in sorted(counts.items()):
            rows.append(
                {
                    "summary_type": "selected_candidate_family_frequency",
                    "scope": "fixed_setting",
                    **meta,
                    "candidate_family": family,
                    "selected_count": int(count),
                    "runs_with_selected_family": int(run_counts.get(family, 0)),
                    "selected_fraction_of_runs": float(run_counts.get(family, 0) / max(total_runs, 1)),
                    "selected_count_per_run": float(count / max(total_runs, 1)),
                    "selected_fraction_of_selected_candidates": float(count / max(selected_total, 1)),
                    "n_runs": total_runs,
                }
            )
    return pd.DataFrame(rows)


def candidate_margin_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expanded = trajectory[trajectory["candidate_set"] == "expanded_gauge_candidate_soup"].copy()
    expanded["accepted_bool"] = expanded["accepted"].astype(bool)
    expanded["margin"] = pd.to_numeric(expanded["validation_accuracy_margin_after_minus_before"], errors="coerce")
    expanded["loss_margin"] = pd.to_numeric(expanded["validation_loss_margin_before_minus_after"], errors="coerce")
    group_cols = ["candidate_family", "candidate_type"]
    for key, group in expanded.groupby(group_cols, dropna=False):
        family, ctype = key
        rows.append(
            {
                "summary_type": "candidate_family_margin",
                "scope": "overall",
                "dataset": "ALL",
                "architecture": "mlp2",
                "n_models": "",
                "width": 128,
                "domain_shift": "ALL",
                "matching": "activation",
                "candidate_family": family,
                "candidate_type": ctype,
                "candidate_rows": int(len(group)),
                "accepted_rows": int(group["accepted_bool"].sum()),
                "rejected_rows": int((~group["accepted_bool"]).sum()),
                "acceptance_fraction": float(group["accepted_bool"].mean()),
                "mean_validation_accuracy_margin": float(np.nanmean(group["margin"])),
                "mean_validation_loss_margin": float(np.nanmean(group["loss_margin"])),
            }
        )
    return pd.DataFrame(rows)


def build_summary(finals: pd.DataFrame, trajectory: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    pieces = [
        paired_summary(finals, bootstrap_samples),
        selected_frequency_summary(finals),
        candidate_margin_summary(trajectory),
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def write_plot(finals: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    expanded = finals[finals["method"] == "expanded_candidate_soup"].copy()
    expanded["delta"] = pd.to_numeric(expanded["delta_test_accuracy_vs_ordinary_greedy"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    if expanded.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No expanded soup rows", ha="center", va="center")
            ax.set_axis_off()
    else:
        labels = []
        values = []
        for key, group in expanded.groupby(["dataset", "n_models", "domain_shift"], dropna=False):
            labels.append(f"{key[0]} N={int(key[1])} {key[2]}")
            values.append(group["delta"].dropna().to_numpy())
        axes[0].boxplot(values, tick_labels=labels, showmeans=True)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].tick_params(axis="x", rotation=35, labelsize=8)
        axes[0].set_ylabel("test accuracy delta vs ordinary greedy")
        axes[0].set_title("Expanded candidate soup deltas")
        axes[0].grid(True, axis="y", alpha=0.25)

        nonlocal_flag = pd.to_numeric(expanded["selected_nonlocal_candidate_count"], errors="coerce") > 0
        axes[1].hist(expanded.loc[~nonlocal_flag, "delta"].dropna(), bins=24, alpha=0.7, label="selected local only")
        axes[1].hist(expanded.loc[nonlocal_flag, "delta"].dropna(), bins=24, alpha=0.7, label="selected gauge candidate")
        axes[1].axvline(0.0, color="black", linewidth=0.8)
        axes[1].set_xlabel("test accuracy delta")
        axes[1].set_ylabel("runs")
        axes[1].set_title("Selection outcome")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy().head(max_rows)
    rows = []
    for row in view.to_dict("records"):
        cleaned = {}
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                cleaned[col] = "" if not math.isfinite(float(value)) else f"{float(value):.4f}"
            else:
                cleaned[col] = value
        rows.append(cleaned)
    out = format_markdown_table(rows, columns)
    if len(df) > max_rows:
        out += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return out


def write_report(args: argparse.Namespace, finals: pd.DataFrame, summary: pd.DataFrame, trajectory: pd.DataFrame, skipped: list[dict], path: Path) -> None:
    paired = summary[summary["summary_type"] == "paired_delta"].copy() if not summary.empty else pd.DataFrame()
    freq = summary[summary["summary_type"] == "selected_candidate_family_frequency"].copy() if not summary.empty else pd.DataFrame()
    margins = summary[summary["summary_type"] == "candidate_family_margin"].copy() if not summary.empty else pd.DataFrame()
    overall = paired[paired["scope"] == "overall"].iloc[0].to_dict() if not paired[paired["scope"] == "overall"].empty else {}
    supported = bool(overall.get("claim_supported", False))
    if supported:
        decision = "A positive expanded-candidate-soup claim passes the paired CI gate."
    else:
        decision = "No positive expanded-candidate-soup improvement claim passes the paired CI gate; the supported interpretation is empirical descent over the expanded family."
    report = f"""# Expanded Candidate Soup Report

Generated by `experiments/expanded_candidate_soup.py`.

## Exact Command

```bash
{args.command_string}
```

## Scope

- This report runs greedy validation descent over ordinary local checkpoints plus gauge-corrected candidate sections.
- Checkpoints come from `reports/checkpoints/fixed_setting_verification` and cover the quality-gated activation-matching `mlp2` width-128 MNIST/Fashion-MNIST settings.
- Candidate families include local checkpoints, ordinary weight average, Git-ReBasin-style ref0 sections/average, C2M3-synchronized sections/average, raw activation monomial sections/average, shrinkage monomial sections/average, global synchronized monomial sections/average, and weight-matched monomial sections/average.
- Branch/rank-lift candidates are not included in the greedy soup because they are extra-capacity or ensemble-style objects rather than same-architecture single-model sections. They remain excluded from single-model claims.
- The selector uses validation accuracy only. Test accuracy is evaluated only for the final selected ordinary and expanded soups.

## Outputs

- `reports/csv/{RUNS_CSV}`
- `reports/csv/{SUMMARY_CSV}`
- `reports/csv/{TRAJECTORY_CSV}`
- `reports/{REPORT_MD}`
- `reports/plots/{PLOT_PDF}`

## Claim Decision

{decision}

Overall mean test delta expanded minus ordinary greedy: `{overall.get("mean_test_delta_expanded_minus_ordinary", float("nan")):.4f}`.
Overall paired CI: `[{overall.get("test_delta_ci_low", float("nan")):.4f}, {overall.get("test_delta_ci_high", float("nan")):.4f}]`.
Claim status: `{overall.get("claim_status", "not_run")}`.

## Paired Delta Summary

{md_table(paired, ["scope", "dataset", "n_models", "domain_shift", "n_runs", "mean_ordinary_greedy_test_accuracy", "mean_expanded_soup_test_accuracy", "mean_test_delta_expanded_minus_ordinary", "test_delta_ci_low", "test_delta_ci_high", "wins", "ties", "losses", "expanded_selected_nonlocal_fraction", "claim_status"], 40)}

## Selected Candidate Family Frequencies

{md_table(freq, ["dataset", "n_models", "domain_shift", "candidate_family", "selected_count", "runs_with_selected_family", "selected_fraction_of_runs", "selected_count_per_run", "selected_fraction_of_selected_candidates", "n_runs"], 80)}

## Validation Margins By Candidate Type

{md_table(margins, ["candidate_family", "candidate_type", "candidate_rows", "accepted_rows", "rejected_rows", "acceptance_fraction", "mean_validation_accuracy_margin", "mean_validation_loss_margin"], 80)}

## Trajectory Audit

Every trajectory row records the validation metric available before the accept/reject decision. Rows with `candidate_set = expanded_gauge_candidate_soup` are the expanded-family decisions; rows with `ordinary_local_greedy_soup` are the local-only baseline. `test_used_for_selection` is `False` for every row.

## Skipped Groups

{md_table(pd.DataFrame(skipped), ["setting_id", "seed", "skip_reason", "missing_model_indices"], 40)}

## Claim Boundary

- A performance improvement claim requires paired CI lower bound greater than zero versus ordinary greedy soup.
- If the CI gate fails, the supported claim is only that greedy validation descent can reject or neutralize extra gauge candidates in this checkpointed candidate family.
- This report does not compare against external Git Re-Basin, external C2M3, Model Soups repos, RegMean, TIES, or natural CIFAR results.
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "reports" / "checkpoints" / "fixed_setting_verification")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--datasets", default="mnist,fashion_mnist")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-seeds-per-setting", type=int, default=0, help="0 means all checkpointed seeds.")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all checkpointed runs after filters.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--dataset-seed", type=int, default=314159)
    parser.add_argument("--augmentation", default="none")
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--monomial-log-scale-clip", type=float, default=2.0)
    parser.add_argument("--monomial-shrinkage", type=float, default=0.5)
    parser.add_argument("--monomial-activation-similarity-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.command_string = " ".join([sys.executable, *sys.argv])
    return args


def main() -> None:
    args = parse_args()
    datasets = set(parse_csv(args.datasets)) if parse_csv(args.datasets) else None
    settings = set(parse_csv(args.settings)) if parse_csv(args.settings) else None
    seeds = parse_seed_text(args.seeds)
    groups, skipped = checkpoint_groups(
        args.checkpoint_root,
        datasets,
        settings,
        seeds,
        args.max_seeds_per_setting,
    )
    if int(args.max_runs) > 0:
        groups = groups[: int(args.max_runs)]

    final_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    loader_cache: dict = {}
    for idx, group in enumerate(groups, start=1):
        print(f"[{idx}/{len(groups)}] {group['setting_id']} seed {group['seed']}", flush=True)
        finals, trajectory, diagnostics = run_group(group, args, loader_cache)
        final_rows.extend(finals)
        trajectory_rows.extend(trajectory)
        diagnostic_rows.extend(diagnostics)

    finals = pd.DataFrame(final_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    summary = build_summary(finals, trajectory, args.bootstrap_samples)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if not diagnostics.empty:
        diagnostics["value"] = pd.to_numeric(diagnostics.get("value"), errors="coerce")
        diag_summary = diagnostics.groupby("diagnostic", dropna=False).agg(
            summary_type=("diagnostic", lambda _: "diagnostic"),
            scope=("diagnostic", lambda _: "overall"),
            value_mean=("value", "mean"),
            value_min=("value", "min"),
            value_max=("value", "max"),
            n_rows=("value", "size"),
        ).reset_index()
        summary = pd.concat([summary, diag_summary], ignore_index=True, sort=False)

    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    finals.to_csv(csv_dir / RUNS_CSV, index=False, lineterminator="\n")
    summary.to_csv(csv_dir / SUMMARY_CSV, index=False, lineterminator="\n")
    trajectory.to_csv(csv_dir / TRAJECTORY_CSV, index=False, lineterminator="\n")
    write_plot(finals, plot_dir / PLOT_PDF)
    write_report(args, finals, summary, trajectory, skipped, args.reports_dir / REPORT_MD)
    print(f"wrote {csv_dir / RUNS_CSV}")
    print(f"wrote {csv_dir / SUMMARY_CSV}")
    print(f"wrote {csv_dir / TRAJECTORY_CSV}")
    print(f"wrote {args.reports_dir / REPORT_MD}")
    print(f"wrote {plot_dir / PLOT_PDF}")
    print(f"commit {git_commit()}")


if __name__ == "__main__":
    main()
