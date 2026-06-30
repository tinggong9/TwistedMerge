#!/usr/bin/env python
"""Validated monomial ladder selector benchmark for MNIST MLP merging."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ladder_merge_methods import (  # noqa: E402
    METHOD_METADATA,
    MethodMetadata,
    estimate_signs_and_positive_scales,
    transform_mlp_positive_scale,
)
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    average_models,
    collect_features,
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    greedy_soup,
    load_dataset,
    make_loader,
    make_model,
    permute_model_to_reference,
    require_torch,
    set_seed,
    synchronize_permutations,
    train_model,
)
from src.structure_group_ladder import (  # noqa: E402
    StructureGroupLadderMerge,
    estimate_pairwise_permutations_from_activations,
)


EXTRA_METADATA: dict[str, MethodMetadata] = {
    "validated_ladder_selector": MethodMetadata(
        method="validated_ladder_selector",
        symmetry_status="validation_selected_exact_relu_single_model",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Validation-only selector choosing between C2M3 permutation average and positive monomial scaling average.",
    ),
    "c2m3_greedy_soup": MethodMetadata(
        method="c2m3_greedy_soup",
        symmetry_status="exact_relu_permutation_soup",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Greedy soup over C2M3-aligned models using validation accuracy.",
    ),
    "monomial_scaled_greedy_soup": MethodMetadata(
        method="monomial_scaled_greedy_soup",
        symmetry_status="exact_relu_positive_scale_soup",
        is_single_model=True,
        capacity_matched_to_weight_average=True,
        notes="Greedy soup over positive-monomial-aligned models using validation accuracy.",
    ),
}

METHODS = {**METHOD_METADATA, **EXTRA_METADATA}
INT_TABLE_COLUMNS = {
    "n_models",
    "width",
    "n_rows",
    "n_seeds",
    "n_pairs",
    "accuracy_wins",
    "accuracy_ties",
    "accuracy_losses",
    "fixed_settings_positive",
    "fixed_settings_total",
    "selector_chose_c2m3",
    "selector_chose_monomial",
    "selector_chosen_test_better",
    "selector_chosen_test_tied",
    "selector_chosen_test_worse",
}


@dataclass(frozen=True)
class PairComparison:
    name: str
    method: str
    baseline: str
    claim_label: str


PAIR_COMPARISONS = (
    PairComparison(
        name="validated_ladder_selector_vs_c2m3_permutation",
        method="validated_ladder_selector",
        baseline="c2m3_permutation",
        claim_label="validated selector over C2M3",
    ),
    PairComparison(
        name="monomial_scale_vs_c2m3_permutation",
        method="monomial_scale",
        baseline="c2m3_permutation",
        claim_label="monomial scaling over C2M3",
    ),
    PairComparison(
        name="monomial_scaled_greedy_soup_vs_greedy_soup",
        method="monomial_scaled_greedy_soup",
        baseline="greedy_soup",
        claim_label="monomial-scaled greedy soup over greedy soup",
    ),
    PairComparison(
        name="validated_ladder_selector_vs_greedy_soup",
        method="validated_ladder_selector",
        baseline="greedy_soup",
        claim_label="validated selector over greedy soup",
    ),
)


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_worktree_dirty() -> bool | str:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return bool(status.strip())
    except Exception:
        return "unknown"


def split_train_val(dataset, val_fraction: float, seed: int):
    torch, _, _ = require_torch()
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)


def bootstrap_mean_ci(values, n_bootstrap: int = 2000, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_bootstrap <= 0:
        value = float(arr.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def standard_error(values) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return float("nan")
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def safe_corr(x, y, method: str = "pearson") -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce")
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x_arr.notna() & y_arr.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(x_arr[mask].corr(y_arr[mask], method=method))


def diag_by_level(ladder_result) -> dict[str, object]:
    out: dict[str, object] = {
        "ladder_final_decision": ladder_result.final_decision,
        "ladder_selected_level": ladder_result.selected_level,
        "supports_brauer_projective_interpretation": any(
            diag.supports_brauer_projective_interpretation
            for diag in ladder_result.diagnostics
        ),
        "has_finite_index_candidate": any(
            diag.is_finite_index_candidate
            for diag in ladder_result.diagnostics
        ),
    }
    for diag in ladder_result.diagnostics:
        prefix = diag.level
        out[f"{prefix}_centrality"] = diag.centrality_score
        out[f"{prefix}_cycle_score"] = diag.cycle_score
        out[f"{prefix}_residual_type"] = diag.residual_type
        out[f"{prefix}_phase_residual"] = diag.phase_residual
        out[f"{prefix}_detected_order_d"] = diag.detected_order_d
    perm = out.get("permutation_centrality", float("nan"))
    mono = out.get("monomial_phase_or_scale_centrality", float("nan"))
    out["monomial_centrality_improvement_from_permutation"] = (
        float(perm - mono) if np.isfinite(perm) and np.isfinite(mono) else float("nan")
    )
    return out


def add_method_row(
    rows: list[dict],
    *,
    base: dict,
    method: str,
    test_metrics: dict[str, float],
    val_metrics: dict[str, float],
    single_best_accuracy: float,
    extra: dict | None = None,
) -> None:
    meta = METHODS[method]
    accuracy = float(test_metrics["accuracy"])
    loss = float(test_metrics["loss"])
    row = {
        **base,
        "method": method,
        "loss": loss,
        "accuracy": accuracy,
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "single_best_accuracy": single_best_accuracy,
        "merge_degradation": single_best_accuracy - accuracy,
        "symmetry_status": meta.symmetry_status,
        "is_single_model": meta.is_single_model,
        "capacity_matched_to_weight_average": meta.capacity_matched_to_weight_average,
        "method_notes": meta.notes,
        "evaluation_status": "evaluated",
    }
    if extra:
        row.update(extra)
    rows.append(row)


def choose_by_validation(c2m3_val: dict[str, float], monomial_val: dict[str, float]) -> str:
    acc_delta = monomial_val["accuracy"] - c2m3_val["accuracy"]
    if acc_delta > 0:
        return "monomial_scale"
    if acc_delta < 0:
        return "c2m3_permutation"
    loss_delta = monomial_val["loss"] - c2m3_val["loss"]
    if loss_delta < 0:
        return "monomial_scale"
    return "c2m3_permutation"


def evaluate_on_val_and_test(model, val_loader, test_loader, device) -> tuple[dict[str, float], dict[str, float]]:
    return evaluate_model(model, val_loader, device), evaluate_model(model, test_loader, device)


def run_setting(args, spec, train_data, test_data, seed: int, n_models: int, width: int) -> list[dict]:
    device = device_from_arg(args.device)
    train_subset, val_subset = split_train_val(train_data, args.val_fraction, seed + 77)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False, seed=seed + 700)
    test_loader = make_loader(test_data, args.batch_size, shuffle=False, seed=seed + 999)
    match_loader = make_loader(train_subset, args.batch_size, shuffle=False, seed=seed + 501)

    models = []
    individual_accuracies = []
    individual_losses = []
    for model_idx in range(n_models):
        model_seed = seed + 1000 * model_idx + 17 * width + n_models
        set_seed(model_seed)
        model = make_model("mlp", spec, width)
        train_loader = make_loader(train_subset, args.batch_size, shuffle=True, seed=model_seed + 11)
        train_model(model, train_loader, args.epochs, args.lr, device)
        test_metrics = evaluate_model(model, test_loader, device)
        individual_accuracies.append(float(test_metrics["accuracy"]))
        individual_losses.append(float(test_metrics["loss"]))
        model.to("cpu")
        models.append(model)

    features = {
        idx: collect_features(model, match_loader, device)
        for idx, model in enumerate(models)
    }
    pairwise = estimate_pairwise_permutations_from_activations(features, n_models, width)
    ref, synced, sync_disagreement = synchronize_permutations(pairwise, n_models)
    ladder_result = StructureGroupLadderMerge(max_order=args.max_order).run(
        {"permutation": pairwise},
        n_models=n_models,
        width=width,
        activations=features,
        candidate_lift_rank=width,
    )
    diagnostics = diag_by_level(ladder_result)
    setting_id = f"mnist_mlp_N{n_models}_W{width}_S{seed}"
    single_best_accuracy = float(max(individual_accuracies))
    base = {
        "setting_id": setting_id,
        "dataset": "mnist",
        "architecture": "mlp_relu",
        "n_models": n_models,
        "width": width,
        "seed": seed,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "val_fraction": args.val_fraction,
        "matching": "activation",
        "sync_reference": ref,
        "sync_disagreement": sync_disagreement,
        "individual_accuracy_mean": float(np.mean(individual_accuracies)),
        "individual_accuracy_min": float(np.min(individual_accuracies)),
        "individual_accuracy_max": single_best_accuracy,
        "individual_loss_mean": float(np.mean(individual_losses)),
        **diagnostics,
    }
    rows: list[dict] = []

    weight_avg = average_models(models, "mlp", spec, width)
    weight_val, weight_test = evaluate_on_val_and_test(weight_avg, val_loader, test_loader, device)
    add_method_row(
        rows,
        base=base,
        method="weight_average",
        test_metrics=weight_test,
        val_metrics=weight_val,
        single_best_accuracy=single_best_accuracy,
    )

    aligned_c2m3 = [
        permute_model_to_reference(model, "mlp", spec, width, synced[idx])
        for idx, model in enumerate(models)
    ]
    c2m3_model = average_models(aligned_c2m3, "mlp", spec, width)
    c2m3_val, c2m3_test = evaluate_on_val_and_test(c2m3_model, val_loader, test_loader, device)
    add_method_row(
        rows,
        base=base,
        method="c2m3_permutation",
        test_metrics=c2m3_test,
        val_metrics=c2m3_val,
        single_best_accuracy=single_best_accuracy,
    )

    scaled_models = []
    scale_stats = []
    for idx, model in enumerate(models):
        perm = synced[idx]
        if idx == ref:
            scales = np.ones(width, dtype=float)
        else:
            _signs, scales = estimate_signs_and_positive_scales(features[ref], features[idx], perm)
        scale_stats.append(float(np.mean(scales)))
        scaled_models.append(transform_mlp_positive_scale(model, spec, width, perm, scales))

    monomial_model = average_models(scaled_models, "mlp", spec, width)
    monomial_val, monomial_test = evaluate_on_val_and_test(monomial_model, val_loader, test_loader, device)
    add_method_row(
        rows,
        base=base,
        method="monomial_scale",
        test_metrics=monomial_test,
        val_metrics=monomial_val,
        single_best_accuracy=single_best_accuracy,
        extra={
            "mean_positive_scale": float(np.mean(scale_stats)),
            "symmetry_warning": "positive ReLU scaling is exact before averaging",
        },
    )

    selected = choose_by_validation(c2m3_val, monomial_val)
    selected_test = monomial_test if selected == "monomial_scale" else c2m3_test
    selected_val = monomial_val if selected == "monomial_scale" else c2m3_val
    alternative = "c2m3_permutation" if selected == "monomial_scale" else "monomial_scale"
    alternative_test = c2m3_test if selected == "monomial_scale" else monomial_test
    chosen_better = selected_test["accuracy"] > alternative_test["accuracy"]
    chosen_tied = selected_test["accuracy"] == alternative_test["accuracy"]
    add_method_row(
        rows,
        base=base,
        method="validated_ladder_selector",
        test_metrics=selected_test,
        val_metrics=selected_val,
        single_best_accuracy=single_best_accuracy,
        extra={
            "selector_chose": selected,
            "selector_alternative": alternative,
            "selector_val_accuracy_delta_monomial_minus_c2m3": monomial_val["accuracy"] - c2m3_val["accuracy"],
            "selector_val_loss_delta_monomial_minus_c2m3": monomial_val["loss"] - c2m3_val["loss"],
            "selector_test_accuracy_delta_monomial_minus_c2m3": monomial_test["accuracy"] - c2m3_test["accuracy"],
            "selector_chosen_test_better": bool(chosen_better),
            "selector_chosen_test_tied": bool(chosen_tied),
            "selector_no_test_leakage": True,
        },
    )

    soup, soup_indices, soup_test = greedy_soup(models, val_loader, test_loader, device, "mlp", spec, width)
    soup_val = evaluate_model(soup, val_loader, device)
    add_method_row(
        rows,
        base=base,
        method="greedy_soup",
        test_metrics=soup_test,
        val_metrics=soup_val,
        single_best_accuracy=single_best_accuracy,
        extra={"soup_indices": json.dumps(soup_indices)},
    )

    c2m3_soup, c2m3_soup_indices, c2m3_soup_test = greedy_soup(aligned_c2m3, val_loader, test_loader, device, "mlp", spec, width)
    c2m3_soup_val = evaluate_model(c2m3_soup, val_loader, device)
    add_method_row(
        rows,
        base=base,
        method="c2m3_greedy_soup",
        test_metrics=c2m3_soup_test,
        val_metrics=c2m3_soup_val,
        single_best_accuracy=single_best_accuracy,
        extra={"soup_indices": json.dumps(c2m3_soup_indices)},
    )

    scaled_soup, scaled_soup_indices, scaled_soup_test = greedy_soup(scaled_models, val_loader, test_loader, device, "mlp", spec, width)
    scaled_soup_val = evaluate_model(scaled_soup, val_loader, device)
    add_method_row(
        rows,
        base=base,
        method="monomial_scaled_greedy_soup",
        test_metrics=scaled_soup_test,
        val_metrics=scaled_soup_val,
        single_best_accuracy=single_best_accuracy,
        extra={"soup_indices": json.dumps(scaled_soup_indices)},
    )

    ensemble_test = evaluate_ensemble(models, test_loader, device)
    ensemble_val = evaluate_ensemble(models, val_loader, device)
    add_method_row(
        rows,
        base=base,
        method="ensemble_upper_bound",
        test_metrics=ensemble_test,
        val_metrics=ensemble_val,
        single_best_accuracy=single_best_accuracy,
    )

    by_method = {row["method"]: row for row in rows}
    c2m3_acc = by_method["c2m3_permutation"]["accuracy"]
    c2m3_loss = by_method["c2m3_permutation"]["loss"]
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    greedy_loss = by_method["greedy_soup"]["loss"]
    for row in rows:
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3_acc
        row["loss_delta_vs_c2m3"] = row["loss"] - c2m3_loss
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy_acc
        row["loss_delta_vs_greedy_soup"] = row["loss"] - greedy_loss
    return rows


def summarize_methods(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    group_specs = [
        ("overall", ["method"]),
        ("fixed_setting", ["n_models", "width", "method"]),
    ]
    for scope, keys in group_specs:
        for key_values, group in df.groupby(keys, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            key_map = dict(zip(keys, key_values, strict=True))
            accuracy = pd.to_numeric(group["accuracy"], errors="coerce")
            loss = pd.to_numeric(group["loss"], errors="coerce")
            degradation = pd.to_numeric(group["merge_degradation"], errors="coerce")
            delta_c2m3 = pd.to_numeric(group["accuracy_delta_vs_c2m3"], errors="coerce")
            delta_greedy = pd.to_numeric(group["accuracy_delta_vs_greedy_soup"], errors="coerce")
            rows.append(
                {
                    "summary_type": "method_summary",
                    "scope": scope,
                    "n_models": key_map.get("n_models", "all"),
                    "width": key_map.get("width", "all"),
                    "method": key_map["method"],
                    "baseline": "",
                    "n_rows": int(len(group)),
                    "n_seeds": int(group["seed"].nunique()),
                    "mean_accuracy": float(accuracy.mean()),
                    "accuracy_standard_error": standard_error(accuracy),
                    "mean_loss": float(loss.mean()),
                    "mean_merge_degradation": float(degradation.mean()),
                    "mean_accuracy_delta_vs_c2m3": float(delta_c2m3.mean()),
                    "mean_accuracy_delta_vs_greedy_soup": float(delta_greedy.mean()),
                    "symmetry_status": str(group["symmetry_status"].iloc[0]),
                    "is_single_model": bool(group["is_single_model"].iloc[0]),
                    "capacity_matched_to_weight_average": bool(group["capacity_matched_to_weight_average"].iloc[0]),
                }
            )
    return rows


def paired_rows(df: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    rows: list[dict] = []
    fixed_summary = df.pivot_table(
        index=["n_models", "width", "seed", "setting_id"],
        columns="method",
        values=["accuracy", "loss"],
        aggfunc="first",
    )
    fixed_summary.columns = [f"{metric}__{method}" for metric, method in fixed_summary.columns]
    fixed_summary = fixed_summary.reset_index()
    for comparison in PAIR_COMPARISONS:
        acc_method = f"accuracy__{comparison.method}"
        acc_base = f"accuracy__{comparison.baseline}"
        loss_method = f"loss__{comparison.method}"
        loss_base = f"loss__{comparison.baseline}"
        clean = fixed_summary[[acc_method, acc_base, loss_method, loss_base, "n_models", "width"]].dropna()
        accuracy_delta = clean[acc_method] - clean[acc_base]
        loss_delta = clean[loss_method] - clean[loss_base]
        wins = int((accuracy_delta > 0).sum())
        ties = int((accuracy_delta == 0).sum())
        losses = int((accuracy_delta < 0).sum())
        ci_low, ci_high = bootstrap_mean_ci(accuracy_delta, n_bootstrap=n_bootstrap, seed=8800 + len(rows))
        fixed_positive = 0
        fixed_total = 0
        for (_n_models, _width), group in clean.groupby(["n_models", "width"], dropna=False):
            fixed_total += 1
            if float((group[acc_method] - group[acc_base]).mean()) > 0:
                fixed_positive += 1
        rows.append(
            {
                "summary_type": "paired_comparison",
                "scope": "overall_paired",
                "comparison": comparison.name,
                "claim_label": comparison.claim_label,
                "method": comparison.method,
                "baseline": comparison.baseline,
                "n_pairs": int(len(clean)),
                "paired_mean_accuracy_delta": float(accuracy_delta.mean()) if len(clean) else float("nan"),
                "paired_accuracy_delta_ci_low": ci_low,
                "paired_accuracy_delta_ci_high": ci_high,
                "paired_mean_loss_delta": float(loss_delta.mean()) if len(clean) else float("nan"),
                "accuracy_wins": wins,
                "accuracy_ties": ties,
                "accuracy_losses": losses,
                "sign_test_two_sided_p": sign_test_two_sided(wins, losses),
                "fixed_settings_positive": fixed_positive,
                "fixed_settings_total": fixed_total,
            }
        )
    return rows


def selector_rows(df: pd.DataFrame) -> list[dict]:
    selector = df[df["method"] == "validated_ladder_selector"].copy()
    rows: list[dict] = []
    for scope, group in [("overall", selector), *[(f"N{n}_W{w}", g) for (n, w), g in selector.groupby(["n_models", "width"])]]:
        choices = group["selector_chose"].value_counts(dropna=False)
        rows.append(
            {
                "summary_type": "selector_behavior",
                "scope": scope,
                "n_rows": int(len(group)),
                "selector_chose_c2m3": int(choices.get("c2m3_permutation", 0)),
                "selector_chose_monomial": int(choices.get("monomial_scale", 0)),
                "selector_chosen_test_better": int(group["selector_chosen_test_better"].fillna(False).astype(bool).sum()),
                "selector_chosen_test_tied": int(group["selector_chosen_test_tied"].fillna(False).astype(bool).sum()),
                "selector_chosen_test_worse": int((~group["selector_chosen_test_better"].fillna(False).astype(bool) & ~group["selector_chosen_test_tied"].fillna(False).astype(bool)).sum()),
                "selector_no_test_leakage": bool(group["selector_no_test_leakage"].fillna(True).astype(bool).all()),
            }
        )
    return rows


def residual_correlation_rows(df: pd.DataFrame) -> list[dict]:
    mono = df[df["method"] == "monomial_scale"].copy()
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("overall", mono)]
    groups.extend((f"N{n}_W{w}", group) for (n, w), group in mono.groupby(["n_models", "width"]))
    for scope, group in groups:
        rows.append(
            {
                "summary_type": "residual_correlation",
                "scope": scope,
                "n_rows": int(len(group)),
                "pearson_centrality_improvement_vs_monomial_gain": safe_corr(
                    group["monomial_centrality_improvement_from_permutation"],
                    group["accuracy_delta_vs_c2m3"],
                    method="pearson",
                ),
                "spearman_centrality_improvement_vs_monomial_gain": safe_corr(
                    group["monomial_centrality_improvement_from_permutation"],
                    group["accuracy_delta_vs_c2m3"],
                    method="spearman",
                ),
                "mean_monomial_centrality_improvement": float(pd.to_numeric(group["monomial_centrality_improvement_from_permutation"], errors="coerce").mean()),
                "mean_monomial_accuracy_delta_vs_c2m3": float(pd.to_numeric(group["accuracy_delta_vs_c2m3"], errors="coerce").mean()),
            }
        )
    return rows


def claim_decision_rows(summary_rows: list[dict]) -> list[dict]:
    paired = {row["comparison"]: row for row in summary_rows if row.get("summary_type") == "paired_comparison"}
    method_summary = [
        row for row in summary_rows
        if row.get("summary_type") == "method_summary" and row.get("scope") == "overall"
    ]
    n_seeds = int(max((row.get("n_seeds", 0) for row in method_summary), default=0))
    rows = []

    def decision_for(comparison: str, strong_label: str, descriptive_label: str, no_label: str) -> tuple[str, str]:
        row = paired[comparison]
        mean_delta = float(row["paired_mean_accuracy_delta"])
        ci_low = float(row["paired_accuracy_delta_ci_low"])
        p_value = float(row["sign_test_two_sided_p"])
        wins = int(row["accuracy_wins"])
        losses = int(row["accuracy_losses"])
        fixed_positive = int(row["fixed_settings_positive"])
        fixed_total = int(row["fixed_settings_total"])
        stats_positive = (np.isfinite(ci_low) and ci_low > 0.0) or (np.isfinite(p_value) and p_value < 0.05 and wins > losses)
        most_fixed = fixed_total > 0 and fixed_positive > fixed_total / 2
        if mean_delta > 0.0 and stats_positive and most_fixed and n_seeds >= 20:
            return strong_label, "paired statistics support a positive capacity-matched MNIST result"
        if mean_delta > 0.0:
            return descriptive_label, "mean accuracy delta is positive but strong-claim criteria are not all met"
        return no_label, "paired mean accuracy delta is not positive"

    selector_decision, selector_reason = decision_for(
        "validated_ladder_selector_vs_c2m3_permutation",
        "Strong MNIST MLP win over internal C2M3",
        "Descriptive MNIST MLP improvement over internal C2M3",
        "No improvement over C2M3",
    )
    mono_decision, mono_reason = decision_for(
        "monomial_scale_vs_c2m3_permutation",
        "Strong MNIST MLP monomial win over internal C2M3",
        "Descriptive MNIST MLP monomial improvement over internal C2M3",
        "No monomial improvement over C2M3",
    )
    scaled_soup_decision, scaled_soup_reason = decision_for(
        "monomial_scaled_greedy_soup_vs_greedy_soup",
        "Strong win over greedy soup",
        "Descriptive improvement over greedy soup",
        "No win over greedy soup",
    )
    selector_greedy_decision, selector_greedy_reason = decision_for(
        "validated_ladder_selector_vs_greedy_soup",
        "Strong selector win over greedy soup",
        "Descriptive selector improvement over greedy soup",
        "No selector win over greedy soup",
    )
    decisions = [
        ("validated_selector_over_c2m3", selector_decision, selector_reason),
        ("monomial_scale_over_c2m3", mono_decision, mono_reason),
        ("monomial_scaled_greedy_soup_over_greedy_soup", scaled_soup_decision, scaled_soup_reason),
        ("validated_selector_over_greedy_soup", selector_greedy_decision, selector_greedy_reason),
    ]
    for claim, decision, reason in decisions:
        rows.append(
            {
                "summary_type": "claim_decision",
                "scope": "overall",
                "claim": claim,
                "claim_decision": decision,
                "claim_reason": reason,
                "n_seeds": n_seeds,
            }
        )
    return rows


def summarize(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(summarize_methods(df))
    rows.extend(paired_rows(df, n_bootstrap))
    rows.extend(selector_rows(df))
    rows.extend(residual_correlation_rows(df))
    rows.extend(claim_decision_rows(rows))
    return pd.DataFrame(rows)


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if value == "":
                values.append("")
            elif isinstance(value, bool):
                values.append(str(value))
            elif pd.isna(value):
                values.append("nan")
            elif col in INT_TABLE_COLUMNS:
                values.append(str(int(round(float(value)))))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    mono = df[df["method"] == "monomial_scale"].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for (n_models, width), group in mono.groupby(["n_models", "width"]):
        ax.scatter(
            group["monomial_centrality_improvement_from_permutation"],
            group["accuracy_delta_vs_c2m3"],
            label=f"N={n_models}, W={width}",
            alpha=0.72,
            s=28,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Monomial centrality improvement")
    ax.set_ylabel("Monomial accuracy delta vs C2M3")
    ax.set_title("Validated Ladder Merge: Residual Diagnostic vs Accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, plot_path: Path, path: Path) -> None:
    method_rows = []
    for method in [
        "weight_average",
        "c2m3_permutation",
        "monomial_scale",
        "validated_ladder_selector",
        "greedy_soup",
        "c2m3_greedy_soup",
        "monomial_scaled_greedy_soup",
        "ensemble_upper_bound",
    ]:
        meta = METHODS[method]
        method_rows.append(
            {
                "method": method,
                "symmetry_status": meta.symmetry_status,
                "is_single_model": meta.is_single_model,
                "capacity_matched": meta.capacity_matched_to_weight_average,
            }
        )

    overall = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "overall")].to_dict("records")
    fixed = summary[(summary["summary_type"] == "method_summary") & (summary["scope"] == "fixed_setting")].to_dict("records")
    paired = summary[summary["summary_type"] == "paired_comparison"].to_dict("records")
    selector = summary[summary["summary_type"] == "selector_behavior"].to_dict("records")
    residual = summary[summary["summary_type"] == "residual_correlation"].to_dict("records")
    claims = summary[summary["summary_type"] == "claim_decision"].to_dict("records")

    claim_text = "\n".join(
        f"- `{row['claim']}`: {row['claim_decision']} ({row['claim_reason']})."
        for row in claims
    )
    grid_is_strong = len(parse_csv(args.seeds, int)) >= 20 and len(parse_csv(args.widths, int)) >= 3
    grid_note = (
        "This run uses the stronger default grid requested in the prompt."
        if grid_is_strong
        else "This run uses a reduced grid; results should be treated as descriptive."
    )

    report = f"""# Validated Ladder Merge Report

This report is generated by `experiments/validated_ladder_merge_benchmark.py`.

## Exact Command

```bash
{args.command_string}
```

## Git State At Report Generation

- HEAD commit: `{git_commit()}`
- Worktree dirty: `{git_worktree_dirty()}`

## Grid Settings

- Dataset: MNIST
- Architecture: one-hidden-layer ReLU MLP
- Model counts: `{args.model_counts}`
- Widths: `{args.widths}`
- Seeds: `{args.seeds}`
- Epochs: `{args.epochs}`
- Train samples before validation split: `{args.max_train_samples}`
- Validation fraction: `{args.val_fraction}`
- Test samples: `{args.max_test_samples}`
- Batch size: `{args.batch_size}`
- Matching: activation

{grid_note}

## Method Labels

{table(method_rows, ["method", "symmetry_status", "is_single_model", "capacity_matched"])}

## Main Performance Table

{table(overall, ["method", "n_rows", "n_seeds", "mean_accuracy", "accuracy_standard_error", "mean_loss", "mean_merge_degradation", "mean_accuracy_delta_vs_c2m3", "mean_accuracy_delta_vs_greedy_soup", "symmetry_status"])}

## Fixed-Setting Performance

{table(fixed, ["n_models", "width", "method", "n_seeds", "mean_accuracy", "mean_accuracy_delta_vs_c2m3", "mean_accuracy_delta_vs_greedy_soup"])}

## Paired Comparisons

{table(paired, ["comparison", "n_pairs", "paired_mean_accuracy_delta", "paired_accuracy_delta_ci_low", "paired_accuracy_delta_ci_high", "paired_mean_loss_delta", "accuracy_wins", "accuracy_ties", "accuracy_losses", "sign_test_two_sided_p", "fixed_settings_positive", "fixed_settings_total"])}

## Selector Behavior

{table(selector, ["scope", "n_rows", "selector_chose_c2m3", "selector_chose_monomial", "selector_chosen_test_better", "selector_chosen_test_tied", "selector_chosen_test_worse", "selector_no_test_leakage"])}

The selector chooses between C2M3 and monomial scaling using validation
accuracy, with validation loss as a tie-break. Test accuracy is recorded only
after the validation choice.

## Residual Diagnostic Correlations

{table(residual, ["scope", "n_rows", "pearson_centrality_improvement_vs_monomial_gain", "spearman_centrality_improvement_vs_monomial_gain", "mean_monomial_centrality_improvement", "mean_monomial_accuracy_delta_vs_c2m3"])}

Plot: `reports/plots/{plot_path.name}`.

## Can We Claim an ML Win?

{claim_text}

Positive monomial scaling is an exact ReLU reparameterization before averaging,
so `monomial_scale`, `validated_ladder_selector`, and
`monomial_scaled_greedy_soup` are single-model capacity-matched methods in this
benchmark. The ensemble is extra capacity and is not used for any single-model
claim.

## Negative Boundaries

- This is an MNIST MLP benchmark, not a broad natural model-merging result.
- This does not compare against external Git Re-Basin or external C2M3 code.
- This does not make signed or full-GL transforms exact ReLU symmetries.
- This does not make noncentral residuals Brauer/projective classes.
- No method is selected using test accuracy.

## Environment

```json
{json.dumps(capture_environment(), indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(1800, 1820)))
    parser.add_argument("--model-counts", default="3,4")
    parser.add_argument("--widths", default="16,32,64")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-seed", type=int, default=8128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    spec, train_data, test_data = load_dataset(
        "mnist",
        args.data_dir,
        args.max_train_samples,
        args.max_test_samples,
        args.dataset_seed,
    )
    rows = []
    for seed in parse_csv(args.seeds, int):
        for n_models in parse_csv(args.model_counts, int):
            for width in parse_csv(args.widths, int):
                rows.extend(run_setting(args, spec, train_data, test_data, seed, n_models, width))

    df = pd.DataFrame(rows)
    summary = summarize(df, args.bootstrap_samples)
    csv_dir = args.reports_dir / "csv"
    plot_dir = args.reports_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "validated_ladder_merge_benchmark.csv"
    summary_path = csv_dir / "validated_ladder_merge_summary.csv"
    plot_path = plot_dir / "validated_ladder_merge_delta_vs_c2m3.pdf"
    report_path = args.reports_dir / "validated_ladder_merge_report.md"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_plot(df, plot_path)
    write_report(args, df, summary, plot_path, report_path)
    print(f"wrote {results_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
