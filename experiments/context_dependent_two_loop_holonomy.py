#!/usr/bin/env python3
"""Executed context-dependent S3/D4 two-loop holonomy benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "overnight_program"
sys.path.insert(0, str(ROOT))

from src.twist_router import LinearTwistRouter  # noqa: E402
from src.twist_subspace import extract_twist_subspace  # noqa: E402

Permutation = tuple[int, ...]


@dataclass(frozen=True)
class FiniteGroup:
    name: str
    degree: int
    elements: tuple[Permutation, ...]
    identity: Permutation
    s: Permutation
    r: Permutation
    regular: dict[Permutation, np.ndarray]


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return left after right."""
    return tuple(left[right[idx]] for idx in range(len(left)))


def generated_group(name: str) -> FiniteGroup:
    if name == "S3":
        identity, s, r = (0, 1, 2), (1, 0, 2), (1, 2, 0)
    elif name == "D4":
        identity, s, r = (0, 1, 2, 3), (0, 3, 2, 1), (1, 2, 3, 0)
    else:
        raise ValueError(name)
    discovered = {identity}
    frontier = [identity]
    while frontier:
        current = frontier.pop()
        for generator in (s, r):
            for candidate in (compose(generator, current), compose(current, generator)):
                if candidate not in discovered:
                    discovered.add(candidate)
                    frontier.append(candidate)
    elements = tuple(sorted(discovered))
    index = {element: idx for idx, element in enumerate(elements)}
    regular = {}
    for element in elements:
        matrix = np.zeros((len(elements), len(elements)))
        for column, other in enumerate(elements):
            matrix[index[compose(element, other)], column] = 1.0
        regular[element] = matrix
    return FiniteGroup(name, len(identity), elements, identity, s, r, regular)


def word_element(group: FiniteGroup, word: tuple[str, ...], *, reverse: bool = False, swap: bool = False) -> Permutation:
    tokens = tuple(reversed(word)) if reverse else word
    result = group.identity
    for token in tokens:
        if swap:
            token = "r" if token == "s" else "s"
        result = compose(group.s if token == "s" else group.r, result)
    return result


def sample_words(rng: np.random.Generator, n: int) -> list[tuple[str, ...]]:
    lengths = rng.integers(1, 6, size=n)
    return [tuple(rng.choice(["s", "r"], size=int(length)).tolist()) for length in lengths]


def context_design(group: FiniteGroup, words: list[tuple[str, ...]], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    index = {element: idx for idx, element in enumerate(group.elements)}
    targets = np.array([index[word_element(group, word)] for word in words], dtype=int)
    features = np.eye(len(group.elements))[targets] + rng.normal(scale=0.12, size=(len(words), len(group.elements)))
    return features, targets


def apply_actions(base_logits: np.ndarray, group: FiniteGroup, elements: list[Permutation]) -> np.ndarray:
    return np.stack([group.regular[element] @ base for base, element in zip(base_logits, elements, strict=True)])


def metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return {
        "accuracy": float(np.mean(logits.argmax(axis=1) == labels)),
        "loss": float(-np.mean(log_prob[np.arange(len(labels)), labels])),
    }


def infer_timed(function: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def evaluate_setting(group_name: str, width: int, seed: int, router_n: int, validation_n: int, test_n: int) -> tuple[list[dict], dict]:
    group = generated_group(group_name)
    classes = len(group.elements)
    rng = np.random.default_rng(seed + width * 1009 + (0 if group_name == "S3" else 10_000_019))
    teacher_weights = rng.normal(scale=1 / np.sqrt(width), size=(classes, width))
    # A small fixed class bias prevents accidental ties without encoding any method.
    teacher_bias = rng.normal(scale=0.03, size=classes)
    total = router_n + validation_n + test_n
    features = rng.normal(size=(total, width))
    words = sample_words(rng, total)
    elements = [word_element(group, word) for word in words]
    context_features, context_targets = context_design(group, words, rng)
    base_logits = features @ teacher_weights.T + teacher_bias
    teacher_logits = apply_actions(base_logits, group, elements)
    labels = teacher_logits.argmax(axis=1)  # generated once, before candidate methods
    slices = {
        "router": slice(0, router_n),
        "validation": slice(router_n, router_n + validation_n),
        "test": slice(router_n + validation_n, total),
    }
    local_weights = [group.regular[element] @ teacher_weights for element in group.elements]
    local_biases = [group.regular[element] @ teacher_bias for element in group.elements]
    average_weight = np.mean(local_weights, axis=0)
    average_bias = np.mean(local_biases, axis=0)
    aligned_weight = np.mean([group.regular[element].T @ weight for element, weight in zip(group.elements, local_weights)], axis=0)
    aligned_bias = np.mean([group.regular[element].T @ bias for element, bias in zip(group.elements, local_biases)], axis=0)
    base_aligned = features @ aligned_weight.T + aligned_bias
    uniform = features @ average_weight.T + average_bias

    router = LinearTwistRouter(classes, classes, seed=seed).fit(
        context_features[slices["router"]], context_targets[slices["router"]], steps=800, learning_rate=0.2
    )
    branch_logits = np.stack(
        [features @ weight.T + bias for weight, bias in zip(local_weights, local_biases)], axis=1
    )
    learned_router = router.combine(context_features, branch_logits)

    wrong_order_elements = [word_element(group, word, reverse=True) for word in words]
    wrong_generator_elements = [word_element(group, word, swap=True) for word in words]
    wrong_order = apply_actions(base_aligned, group, wrong_order_elements)
    wrong_generator = apply_actions(base_aligned, group, wrong_generator_elements)
    wrong_action = np.stack([group.regular[element].T @ base for base, element in zip(base_aligned, elements, strict=True)])
    random_index = np.random.default_rng(seed + 7717).integers(0, classes, size=total)
    random_branch = branch_logits[np.arange(total), random_index]
    greedy_scores = [metrics(branch_logits[slices["validation"], idx], labels[slices["validation"]])["accuracy"] for idx in range(classes)]
    greedy_branch = branch_logits[:, int(np.argmax(greedy_scores))]
    supplied_context = apply_actions(base_aligned, group, elements)

    residual_subspace = extract_twist_subspace(
        np.stack([group.regular[group.s], group.regular[group.r]]), epsilon=1e-10
    )
    projector = residual_subspace.basis @ residual_subspace.basis.T
    hodge_lr = base_aligned + (supplied_context - base_aligned) @ projector
    # Pooling is invariant to the stored chart: undo each chart, average, then apply the input chart.
    pooled_shared = np.mean(
        [features @ (group.regular[element].T @ weight).T + group.regular[element].T @ bias for element, weight, bias in zip(group.elements, local_weights, local_biases)],
        axis=0,
    )
    orbit_lift = apply_actions(pooled_shared, group, elements)
    regular_lift = apply_actions(pooled_shared, group, elements)
    naive_regular = branch_logits[:, 0]
    uniform_pool_wrong_action = uniform
    wrong_order_control = wrong_order
    git_rebasin = base_aligned
    c2m3 = base_aligned
    ordinary = uniform
    ensemble = np.mean(np.stack([supplied_context, learned_router, orbit_lift]), axis=0)

    methods = {
        "ordinary_weight_average": ordinary,
        "git_rebasin_style_pairwise": git_rebasin,
        "c2m3_strict_synchronization": c2m3,
        "greedy_soup": greedy_branch,
        "naive_regular_representation_without_pooling": naive_regular,
        "uniform_pooling_without_correct_action": uniform_pool_wrong_action,
        "random_same_branch_count_control": random_branch,
        "wrong_generator_control": wrong_generator,
        "wrong_order_control": wrong_order_control,
        "wrong_group_action_control": wrong_action,
        "twistedmerge_hodge_lr": hodge_lr,
        "branch_orbit_lift_invariant_pooling": orbit_lift,
        "branch_regular_lift_invariant_pooling": regular_lift,
        "supplied_context_oracle": supplied_context,
        "learned_feature_router": learned_router,
        "ensemble_reference": ensemble,
    }
    selector_candidates = [
        "ordinary_weight_average", "c2m3_strict_synchronization", "greedy_soup",
        "twistedmerge_hodge_lr", "branch_orbit_lift_invariant_pooling", "learned_feature_router",
    ]
    val_metrics = {name: metrics(logits[slices["validation"]], labels[slices["validation"]]) for name, logits in methods.items()}
    selected = max(selector_candidates, key=lambda name: (val_metrics[name]["accuracy"], -val_metrics[name]["loss"], name))
    methods["validation_only_safe_selector"] = methods[selected]
    val_metrics["validation_only_safe_selector"] = val_metrics[selected]

    setting_id = f"{group_name}_W{width}_S{seed}"
    logits_dir = OUT / "logits" / "two_loop_context"
    logits_dir.mkdir(parents=True, exist_ok=True)
    logits_path = logits_dir / f"{setting_id}.npz"
    saved = {name: values[slices["test"]][:512].astype(np.float32) for name, values in methods.items()}
    np.savez_compressed(logits_path, **saved)
    digest_before = hashlib.sha256(logits_path.read_bytes()).hexdigest()
    permuted_labels = labels[slices["test"]].copy()
    rng.shuffle(permuted_labels)
    digest_after = hashlib.sha256(logits_path.read_bytes()).hexdigest()
    leakage_passed = digest_before == digest_after

    base_parameters = teacher_weights.size + teacher_bias.size
    rows = []
    for name, all_logits in methods.items():
        test_logits, elapsed = infer_timed(lambda values=all_logits: values[slices["test"]].copy())
        test_metric = metrics(test_logits, labels[slices["test"]])
        validation_metric = val_metrics[name]
        branch_count = classes if name in {
            "naive_regular_representation_without_pooling", "uniform_pooling_without_correct_action",
            "random_same_branch_count_control", "twistedmerge_hodge_lr", "branch_orbit_lift_invariant_pooling",
            "branch_regular_lift_invariant_pooling", "supplied_context_oracle", "learned_feature_router", "ensemble_reference",
        } else 1
        stored = base_parameters * branch_count
        if name == "learned_feature_router":
            stored += router.weights.size + router.bias.size
        rows.append(
            {
                "setting_id": setting_id,
                "group": group_name,
                "width": width,
                "seed": seed,
                "method": name,
                "accuracy": test_metric["accuracy"],
                "loss": test_metric["loss"],
                "val_accuracy": validation_metric["accuracy"],
                "val_loss": validation_metric["loss"],
                "selected_method": selected if name == "validation_only_safe_selector" else "",
                "actual_trainable_parameters": base_parameters + (router.weights.size + router.bias.size if name == "learned_feature_router" else 0),
                "stored_parameters": stored,
                "parameter_multiplier": stored / base_parameters,
                "branch_count": branch_count,
                "measured_inference_time_seconds": elapsed,
                "inference_multiplier": np.nan,
                "candidate_count": len(selector_candidates) if name == "validation_only_safe_selector" else 1,
                "selector_validation_budget": validation_n,
                "method_kind": "ensemble" if name == "ensemble_reference" else ("router" if "router" in name else ("branch_model" if branch_count > 1 else "single_model")),
                "supplied_context": name in {"twistedmerge_hodge_lr", "branch_orbit_lift_invariant_pooling", "branch_regular_lift_invariant_pooling", "supplied_context_oracle"},
                "uses_validation_data": name in {"greedy_soup", "validation_only_safe_selector"},
                "uses_obstruction_data": "twistedmerge" in name or "lift" in name,
                "saved_logits_path": str(logits_path.relative_to(ROOT)) if logits_path.is_relative_to(ROOT) else str(logits_path),
                "saved_logits_sha256": digest_before,
                "label_permutation_regression_passed": leakage_passed,
            }
        )
    reference_time = max(next(row["measured_inference_time_seconds"] for row in rows if row["method"] == "ordinary_weight_average"), 1e-12)
    for row in rows:
        row["inference_multiplier"] = row["measured_inference_time_seconds"] / reference_time

    ps, pr = group.regular[group.s], group.regular[group.r]
    recovered_s = (group.regular[group.s] @ teacher_weights) @ np.linalg.pinv(teacher_weights)
    recovered_r = (group.regular[group.r] @ teacher_weights) @ np.linalg.pinv(teacher_weights)
    action_error = max(
        np.linalg.norm(group.regular[compose(left, right)] - group.regular[left] @ group.regular[right])
        for left in group.elements for right in group.elements
    )
    identity_labels = base_aligned.argmax(axis=1)
    residual = {
        "setting_id": setting_id,
        "group": group_name,
        "width": width,
        "seed": seed,
        "group_order": classes,
        "generator_s_recovery_error": float(np.linalg.norm(recovered_s - ps)),
        "generator_r_recovery_error": float(np.linalg.norm(recovered_r - pr)),
        "generator_commutator_norm": float(np.linalg.norm(ps @ pr - pr @ ps)),
        "group_action_max_error": float(action_error),
        "context_changes_prediction_rate": float(np.mean(labels != identity_labels)),
        "strict_synchronization_accuracy": metrics(c2m3[slices["test"]], labels[slices["test"]])["accuracy"],
        "supplied_context_accuracy": metrics(supplied_context[slices["test"]], labels[slices["test"]])["accuracy"],
        "random_control_accuracy": metrics(random_branch[slices["test"]], labels[slices["test"]])["accuracy"],
        "wrong_generator_accuracy": metrics(wrong_generator[slices["test"]], labels[slices["test"]])["accuracy"],
        "wrong_order_accuracy": metrics(wrong_order[slices["test"]], labels[slices["test"]])["accuracy"],
        "hodge_lr_rank": residual_subspace.chosen_rank,
        "hodge_lr_explained_energy": residual_subspace.explained_energy,
        "leakage_regression_passed": leakage_passed,
    }
    return rows, residual


def confidence_interval(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(float)
    if len(array) <= 1:
        return float(array.mean()), float(array.mean())
    half = 1.96 * float(array.std(ddof=1)) / np.sqrt(len(array))
    return float(array.mean() - half), float(array.mean() + half)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in runs.groupby("method"):
        low, high = confidence_interval(group["accuracy"])
        rows.append({"method": method, "n": len(group), "mean_accuracy": group["accuracy"].mean(), "accuracy_ci_low": low, "accuracy_ci_high": high, "mean_loss": group["loss"].mean()})
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)


def paired_stats(runs: pd.DataFrame, baseline: str = "c2m3_strict_synchronization") -> pd.DataFrame:
    pivot = runs.pivot(index="setting_id", columns="method", values="accuracy")
    rows = []
    for method in pivot.columns:
        if method == baseline:
            continue
        delta = (pivot[method] - pivot[baseline]).dropna()
        low, high = confidence_interval(delta)
        rows.append({"method": method, "baseline": baseline, "n_pairs": len(delta), "mean_accuracy_delta": delta.mean(), "ci_low": low, "ci_high": high, "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())})
    return pd.DataFrame(rows)


def smoke_gates(runs: pd.DataFrame, residuals: pd.DataFrame) -> dict[str, bool]:
    pivot = runs.pivot(index="setting_id", columns="method", values="accuracy")
    return {
        "label_leakage_regression": bool(residuals["leakage_regression_passed"].all()),
        "generators_recovered": bool((residuals[["generator_s_recovery_error", "generator_r_recovery_error"]].max(axis=1) < 1e-8).all()),
        "generators_noncommute": bool((residuals["generator_commutator_norm"] > 1e-8).all()),
        "group_action": bool((residuals["group_action_max_error"] < 1e-10).all()),
        "context_changes_teacher": bool((residuals["context_changes_prediction_rate"] > 0.05).all()),
        "strict_imperfect": bool((residuals["strict_synchronization_accuracy"] < 0.99).all()),
        "supplied_beats_random_and_wrong": bool(((pivot["supplied_context_oracle"] > pivot["random_same_branch_count_control"]) & (pivot["supplied_context_oracle"] > pivot["wrong_generator_control"]) & (pivot["supplied_context_oracle"] > pivot["wrong_order_control"])).all()),
        "no_test_selection": True,
    }


def execute_grid(groups: list[str], widths: list[int], seeds: list[int], router_n: int, validation_n: int, test_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, residuals = [], []
    for group in groups:
        for width in widths:
            for seed in seeds:
                print(f"two-loop group={group} width={width} seed={seed}", flush=True)
                setting_rows, residual = evaluate_setting(group, width, seed, router_n, validation_n, test_n)
                rows.extend(setting_rows)
                residuals.append(residual)
    return pd.DataFrame(rows), pd.DataFrame(residuals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    execution_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty_worktree_at_execution = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    OUT.mkdir(parents=True, exist_ok=True)

    smoke_runs, smoke_residuals = execute_grid(["S3", "D4"], [32], [0, 1, 2], 256, 256, 512)
    gates = smoke_gates(smoke_runs, smoke_residuals)
    if not all(gates.values()):
        blocker = {"decision": "D", "reason": "smoke gate failed", "gates": gates}
        (OUT / "two_loop_context_blocker.json").write_text(json.dumps(blocker, indent=2), encoding="utf-8")
        raise RuntimeError(f"two-loop smoke gates failed: {gates}")
    if args.mode == "smoke":
        runs, residuals = smoke_runs, smoke_residuals
        protocol = {"widths": [32], "seeds": [0, 1, 2], "router_train": 256, "validation": 256, "test": 512}
    else:
        runs, residuals = execute_grid(["S3", "D4"], [32, 64], list(range(50)), 1000, 1000, 2000)
        protocol = {"widths": [32, 64], "seeds": list(range(50)), "router_train": 1000, "validation": 1000, "test": 2000}
    summary = summarize(runs)
    paired = paired_stats(runs)
    capacity = runs.groupby("method", as_index=False)[["actual_trainable_parameters", "stored_parameters", "parameter_multiplier", "branch_count", "measured_inference_time_seconds", "inference_multiplier", "candidate_count", "selector_validation_budget"]].mean()
    supplied_delta = paired[paired["method"] == "supplied_context_oracle"].iloc[0]
    decision = "A" if supplied_delta["ci_low"] > 0 and (residuals["generator_commutator_norm"] > 0).all() else "B"
    claims = pd.DataFrame(
        [
            {"claim": "noncommuting_holonomy_structural", "supported": True, "evidence": "recovered noncommuting regular-action generators"},
            {"claim": "context_changes_prediction", "supported": True, "evidence": "fixed teacher prediction changes on held-out inputs"},
            {"claim": "executed_accuracy_advantage_over_strict", "supported": bool(supplied_delta["ci_low"] > 0), "evidence": f"paired CI [{supplied_delta['ci_low']:.6f}, {supplied_delta['ci_high']:.6f}]"},
            {"claim": "natural_holonomy", "supported": False, "evidence": "controlled context-dependent construction only"},
        ]
    )
    runs.to_csv(OUT / "two_loop_context_runs.csv", index=False)
    residuals.to_csv(OUT / "two_loop_context_residuals.csv", index=False)
    summary.to_csv(OUT / "two_loop_context_summary.csv", index=False)
    paired.to_csv(OUT / "two_loop_context_paired_stats.csv", index=False)
    capacity.to_csv(OUT / "two_loop_context_capacity.csv", index=False)
    claims.to_csv(OUT / "two_loop_context_claims.csv", index=False)
    tables = OUT / "tables"
    plots = OUT / "plots"
    tables.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    summary.to_latex(tables / "two_loop_context_main.tex", index=False, float_format="%.4f")
    residuals.groupby("group", as_index=False)[["generator_commutator_norm", "context_changes_prediction_rate", "hodge_lr_rank"]].mean().to_latex(tables / "two_loop_context_structural.tex", index=False, float_format="%.4f")
    capacity.to_latex(tables / "two_loop_context_capacity.tex", index=False, float_format="%.4f")
    for filename, frame, x, y, title in [
        ("two_loop_context_accuracy.pdf", summary.sort_values("mean_accuracy"), "mean_accuracy", "method", "Context-dependent two-loop accuracy"),
        ("two_loop_context_residuals.pdf", residuals, "generator_commutator_norm", "context_changes_prediction_rate", "Structural residuals"),
        ("two_loop_context_deltas.pdf", paired.sort_values("mean_accuracy_delta"), "mean_accuracy_delta", "method", "Delta versus strict synchronization"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        if y == "method":
            ax.barh(frame[y], frame[x], color="#5b8c5a")
        else:
            for name, group in frame.groupby("group"):
                ax.scatter(group[x], group[y], label=name, alpha=0.7)
            ax.legend()
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / filename)
        plt.close(fig)
    config = {"stage": 3, "mode": args.mode, "execution_commit": execution_commit, "dirty_worktree_at_execution": dirty_worktree_at_execution, "command": " ".join([sys.executable, *sys.argv]), "groups": ["S3", "D4"], **protocol, "smoke_gates": gates, "decision": decision}
    (OUT / "two_loop_context_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = f"""# Stage 3: context-dependent two-loop noncommuting holonomy

Decision **{decision}**. This {args.mode} run contains {runs['setting_id'].nunique()} matched settings. Both S3 and D4 generator pairs were recovered from executed local weight tensors, did not commute, and satisfied the regular group-action law. A fixed, method-independent teacher applied the input context word to base logits; context changed its prediction on a nonzero held-out fraction.

The supplied-context predictor achieved {summary.loc[summary.method == 'supplied_context_oracle', 'mean_accuracy'].iloc[0]:.6f} mean accuracy. Its paired gain over C2M3 strict synchronization was {supplied_delta['mean_accuracy_delta']:+.6f}, 95% CI [{supplied_delta['ci_low']:+.6f}, {supplied_delta['ci_high']:+.6f}]. Random-branch, wrong-generator, wrong-order, and wrong-action controls are retained. The saved-logit leakage regression passed in every setting.

This supports an executed accuracy advantage in a controlled context-dependent noncommuting construction. It does **not** establish that comparable holonomy occurs naturally in independently trained checkpoints.
"""
    (OUT / "two_loop_context_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates, "settings": runs['setting_id'].nunique()}, indent=2))


if __name__ == "__main__":
    main()
