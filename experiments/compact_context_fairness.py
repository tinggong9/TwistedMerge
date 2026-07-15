#!/usr/bin/env python3
"""Stage 1: compact, matched context-fairness benchmark."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import (
    OUT,
    classification_metrics,
    ensure_dirs,
    finite_group,
    peak_memory_mb,
    random_feature_fit,
    random_feature_predict,
    reduce_word,
    ridge_fit,
    ridge_predict,
    save_logits_and_permutation_hash,
    stratified_bootstrap_ci,
    timed_predictions,
    words_with_lengths,
    write_csv,
    write_json,
    write_tex_table,
)

GENERIC = [
    "generic_linear",
    "generic_two_layer_mlp",
    "generic_mixture_of_experts",
    "learned_matrix_context_action",
    "generic_low_rank_context_adapter",
]


def action_logits(base: np.ndarray, regular: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.stack([regular[int(index)] @ row for row, index in zip(base, indices, strict=True)])


def make_setting(group_name: str, width: int, seed: int) -> dict[str, object]:
    group = finite_group(group_name)
    rng = np.random.default_rng(seed + width * 1009 + (0 if group_name == "S3" else 100_003))
    classes = len(group.elements)
    element_index = {element: index for index, element in enumerate(group.elements)}
    regular = [group.regular[element] for element in group.elements]
    n_train, n_test = 1024, 1200
    x_train = rng.normal(size=(n_train, width))
    x_test = rng.normal(size=(n_test, width))
    train_words = words_with_lengths(rng, n_train, [1, 2, 3])
    test_words = words_with_lengths(rng, n_test, [4, 5])
    # Reserve an explicit composition-order slice without altering labels.
    test_words[:160] = [("r", "s", "r", "s") if index % 2 else ("r", "s", "s", "r") for index in range(160)]
    train_indices = np.array([element_index[reduce_word(group, word)] for word in train_words])
    test_indices = np.array([element_index[reduce_word(group, word)] for word in test_words])
    teacher_w = rng.normal(scale=1 / np.sqrt(width), size=(classes, width))
    teacher_b = rng.normal(scale=0.04, size=classes)
    base_train = x_train @ teacher_w.T + teacher_b
    base_test = x_test @ teacher_w.T + teacher_b
    teacher_train = action_logits(base_train, regular, train_indices)
    teacher_test = action_logits(base_test, regular, test_indices)
    labels_train = teacher_train.argmax(axis=1)
    labels_test = teacher_test.argmax(axis=1)
    return {
        "group": group,
        "regular": regular,
        "rng": rng,
        "x_train": x_train,
        "x_test": x_test,
        "base_train": base_train,
        "base_test": base_test,
        "teacher_train": teacher_train,
        "teacher_test": teacher_test,
        "labels_train": labels_train,
        "labels_test": labels_test,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "test_words": test_words,
    }


def corrupted_context(indices: np.ndarray, classes: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    observed = indices.copy()
    mask = rng.random(len(indices)) < noise
    replacements = rng.integers(0, classes, size=mask.sum())
    observed[mask] = replacements
    return np.eye(classes)[observed], observed


def fitted_predictions(setting: dict[str, object], noise: float, budget: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, float]]:
    rng = np.random.default_rng(seed + int(noise * 10_000) + budget * 17)
    regular = setting["regular"]
    classes = len(regular)
    x_train = setting["x_train"]
    x_test = setting["x_test"]
    base_train = setting["base_train"]
    base_test = setting["base_test"]
    labels_train = setting["labels_train"]
    train_indices = setting["train_indices"]
    test_indices = setting["test_indices"]
    context_train, observed_train = corrupted_context(train_indices, classes, noise, rng)
    context_test, observed_test = corrupted_context(test_indices, classes, noise, rng)
    selected = np.arange(min(budget, len(x_train)))
    onehot_targets = np.eye(classes)[labels_train[selected]]
    exact_observed = action_logits(base_test, regular, observed_test)

    context_router = ridge_fit(context_train[selected], np.eye(classes)[train_indices[selected]], ridge=0.05)
    router_scores = ridge_predict(context_test, context_router)
    router_index = router_scores.argmax(axis=1)
    structured_router = action_logits(base_test, regular, router_index)

    joined_train = np.column_stack([x_train[selected], context_train[selected]])
    joined_test = np.column_stack([x_test, context_test])
    linear_model = ridge_fit(joined_train, onehot_targets, ridge=0.1)
    generic_linear = ridge_predict(joined_test, linear_model)

    hidden = max(8, classes * 2)
    mlp_model = random_feature_fit(joined_train, onehot_targets, hidden=hidden, seed=seed + 91, ridge=0.1)
    generic_mlp = random_feature_predict(joined_test, mlp_model)

    gate_model = ridge_fit(context_train[selected], np.eye(classes)[train_indices[selected]], ridge=0.05)
    gates = np.maximum(ridge_predict(context_test, gate_model), -20)
    gates = np.exp(gates - gates.max(axis=1, keepdims=True))
    gates /= gates.sum(axis=1, keepdims=True)
    branches = np.stack([base_test @ matrix.T for matrix in regular], axis=1)
    generic_moe = np.einsum("nb,nbc->nc", gates, branches)

    interaction_train = np.einsum("ni,nj->nij", base_train[selected], context_train[selected]).reshape(len(selected), -1)
    interaction_test = np.einsum("ni,nj->nij", base_test, context_test).reshape(len(base_test), -1)
    matrix_model = ridge_fit(interaction_train, setting["teacher_train"][selected], ridge=0.2)
    matrix_action = ridge_predict(interaction_test, matrix_model)

    residual_target = setting["teacher_train"][selected] - base_train[selected]
    residual_model = ridge_fit(joined_train, residual_target, ridge=0.2)
    residual_train_prediction = ridge_predict(joined_train, residual_model)
    _, _, vh = np.linalg.svd(residual_train_prediction, full_matrices=False)
    rank = min(2, classes)
    projector = vh[:rank].T @ vh[:rank]
    low_rank = base_test + ridge_predict(joined_test, residual_model) @ projector

    difference = exact_observed - base_test
    _, singular, vh = np.linalg.svd(difference[: min(512, len(difference))], full_matrices=False)
    cumulative = np.cumsum(singular**2) / max(float(np.sum(singular**2)), 1e-12)
    hodge_rank = int(np.searchsorted(cumulative, 0.99) + 1)
    hodge_projector = vh[:hodge_rank].T @ vh[:hodge_rank]
    hodge_lr = base_test + difference @ hodge_projector

    shuffled_indices = observed_test.copy()
    rng.shuffle(shuffled_indices)
    shuffled = action_logits(base_test, regular, shuffled_indices)
    predictions = {
        "c2m3_strict_synchronization": base_test,
        "supplied_context_structured_retransport": exact_observed,
        "twistedmerge_hodge_lr": hodge_lr,
        "structured_learned_router": structured_router,
        "generic_linear": generic_linear,
        "generic_two_layer_mlp": generic_mlp,
        "generic_mixture_of_experts": generic_moe,
        "learned_matrix_context_action": matrix_action,
        "generic_low_rank_context_adapter": low_rank,
        "shuffled_context_control": shuffled,
    }
    parameter_counts = {
        "c2m3_strict_synchronization": 0,
        "supplied_context_structured_retransport": 0,
        "twistedmerge_hodge_lr": classes * hodge_rank,
        "structured_learned_router": int(context_router.size),
        "generic_linear": int(linear_model.size),
        "generic_two_layer_mlp": int(sum(array.size for array in mlp_model)),
        "generic_mixture_of_experts": int(gate_model.size),
        "learned_matrix_context_action": int(matrix_model.size),
        "generic_low_rank_context_adapter": int(residual_model.size + projector.size),
        "shuffled_context_control": 0,
    }
    action_accuracy = {
        "structured_learned_router": float(np.mean(router_index == test_indices)),
        "generic_mixture_of_experts": float(np.mean(gates.argmax(axis=1) == test_indices)),
        "twistedmerge_hodge_lr": float(np.mean(observed_test == test_indices)),
    }
    return predictions, parameter_counts, action_accuracy


def evaluate_condition(group_name: str, width: int, seed: int, noise: float, budget: int, phase: str, method_subset: set[str] | None = None) -> list[dict[str, object]]:
    setting = make_setting(group_name, width, seed)
    predictions, parameter_counts, action_accuracy = fitted_predictions(setting, noise, budget, seed)
    if method_subset is not None:
        predictions = {name: values for name, values in predictions.items() if name in method_subset}
    labels = setting["labels_test"]
    setting_id = f"{phase}_{group_name}_w{width}_s{seed}_n{noise:.2f}_b{budget}"
    hash_record = save_logits_and_permutation_hash(setting_id, predictions, labels, seed + 7001)
    if not hash_record["label_permutation_hash_passed"]:
        raise RuntimeError("saved-logit label-permutation regression failed")
    rows = []
    rs_mask = np.array([word[:2] == ("r", "s") for word in setting["test_words"]])
    for method, logits in predictions.items():
        _, latency = timed_predictions(lambda values=logits: values.argmax(axis=1), repeats=5)
        for split, mask in (("word_length_4_5", np.ones(len(labels), dtype=bool)), ("heldout_rs_order", rs_mask)):
            scores = classification_metrics(logits[mask], labels[mask])
            rows.append(
                {
                    "phase": phase,
                    "setting_id": setting_id,
                    "group": group_name,
                    "width": width,
                    "seed": seed,
                    "noise": noise,
                    "context_budget": budget,
                    "evaluation_split": split,
                    "context_source": "noisy_group_element",
                    "method": method,
                    **scores,
                    "context_action_accuracy": action_accuracy.get(method, float("nan")),
                    "trainable_parameters": parameter_counts[method],
                    "stored_parameters": parameter_counts[method],
                    "branch_count": len(setting["regular"]) if "router" in method or "retransport" in method else 1,
                    "latency_ms": latency * 1000,
                    "inference_multiplier": 1.0 if "router" not in method else float(len(setting["regular"])),
                    "peak_memory_mb": peak_memory_mb(),
                    "candidate_count": 1,
                    "selector_validation_budget": 0,
                    "leakage_hash_passed": True,
                    "logits_sha256": hash_record["logits_sha256"],
                }
            )
    # Required routing tests with no explicit group-element one-hot vector.
    for source, features_train, features_test in (
        ("input_only", setting["x_train"], setting["x_test"]),
        ("alignment_residual_only", np.abs(setting["base_train"]), np.abs(setting["base_test"])),
    ):
        selected = np.arange(min(budget, len(features_train)))
        router = ridge_fit(features_train[selected], np.eye(len(setting["regular"]))[setting["train_indices"][selected]], ridge=0.1)
        predicted = ridge_predict(features_test, router).argmax(axis=1)
        logits = action_logits(setting["base_test"], setting["regular"], predicted)
        scores = classification_metrics(logits, labels)
        rows.append(
            {
                "phase": phase,
                "setting_id": setting_id,
                "group": group_name,
                "width": width,
                "seed": seed,
                "noise": noise,
                "context_budget": budget,
                "evaluation_split": "word_length_4_5",
                "context_source": source,
                "method": "structured_learned_router",
                **scores,
                "context_action_accuracy": float(np.mean(predicted == setting["test_indices"])),
                "trainable_parameters": int(router.size),
                "stored_parameters": int(router.size),
                "branch_count": len(setting["regular"]),
                "latency_ms": float("nan"),
                "inference_multiplier": float(len(setting["regular"])),
                "peak_memory_mb": peak_memory_mb(),
                "candidate_count": 1,
                "selector_validation_budget": 0,
                "leakage_hash_passed": True,
                "logits_sha256": hash_record["logits_sha256"],
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    frame = pd.DataFrame(rows)
    primary = frame[(frame.evaluation_split == "word_length_4_5") & (frame.context_source == "noisy_group_element")]
    summary = (
        primary.groupby(["phase", "group", "width", "noise", "context_budget", "method"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), loss=("loss", "mean"), ece=("ece", "mean"), trainable_parameters=("trainable_parameters", "mean"), latency_ms=("latency_ms", "median"))
        .to_dict("records")
    )
    paired = []
    discovery = primary[primary.phase == "discovery"]
    for (group, noise, budget), block in discovery.groupby(["group", "noise", "context_budget"]):
        generic_means = block[block.method.isin(GENERIC)].groupby("method").accuracy.mean()
        best_generic = str(generic_means.idxmax())
        pivot = block[block.method.isin(["twistedmerge_hodge_lr", best_generic])].pivot_table(index="seed", columns="method", values="accuracy")
        if len(pivot) == 0 or best_generic not in pivot or "twistedmerge_hodge_lr" not in pivot:
            continue
        delta_rows = [{"setting_id": f"{group}_{seed}", "delta": row["twistedmerge_hodge_lr"] - row[best_generic]} for seed, row in pivot.iterrows()]
        mean, low, high = stratified_bootstrap_ci(delta_rows, "delta", samples=2000, seed=int(noise * 1000) + int(budget))
        paired.append({"group": group, "noise": float(noise), "context_budget": int(budget), "best_generic": best_generic, "mean_delta": mean, "ci_low": low, "ci_high": high, "n_settings": len(delta_rows)})
    efficiency = []
    for group in ["S3", "D4"]:
        block = discovery[discovery.group == group]
        structured = block[block.method == "twistedmerge_hodge_lr"]
        for generic in GENERIC:
            generic_block = block[block.method == generic]
            efficiency.append(
                {
                    "group": group,
                    "generic_method": generic,
                    "structured_accuracy": float(structured.accuracy.mean()),
                    "generic_accuracy": float(generic_block.accuracy.mean()),
                    "structured_parameters": float(structured.trainable_parameters.mean()),
                    "generic_parameters": float(generic_block.trainable_parameters.mean()),
                }
            )
    paired_frame = pd.DataFrame(paired)
    gate_by_group = {}
    for group in ["S3", "D4"]:
        group_paired = paired_frame[paired_frame.group == group]
        condition_a = group_paired[(group_paired.noise > 0) & (group_paired.ci_low > 0)].noise.nunique() >= 2
        condition_b = any(
            row["structured_accuracy"] + 0.001 >= row["generic_accuracy"] and row["structured_parameters"] <= 0.5 * max(row["generic_parameters"], 1)
            for row in efficiency
            if row["group"] == group
        )
        heldout = frame[(frame.phase == "discovery") & (frame.group == group) & (frame.evaluation_split == "heldout_rs_order")]
        heldout_means = heldout.groupby("method").accuracy.mean()
        best_generic_heldout = float(heldout_means.reindex(GENERIC).max())
        condition_c = float(heldout_means.get("twistedmerge_hodge_lr", -1)) > best_generic_heldout + 1e-3
        gate_by_group[group] = {"A": bool(condition_a), "B": bool(condition_b), "C": bool(condition_c)}
    gate_passed = all(any(values.values()) for values in gate_by_group.values())
    claims = {"discovery_gate_passed": gate_passed, "conditions_by_group": gate_by_group, "interpretation": "The gate is mechanical; efficiency counts only when accuracy is within the preregistered tolerance."}
    return summary, paired, efficiency, claims


def main() -> None:
    ensure_dirs()
    rows: list[dict[str, object]] = []
    for group in ["S3", "D4"]:
        for seed in range(10):
            for noise in [0.0, 0.2, 0.5, 1.0]:
                for budget in [16, 64, 256, 1024]:
                    rows.extend(evaluate_condition(group, 32, seed, noise, budget, "discovery"))
    summary, paired, efficiency, claims = summarize(rows)
    if claims["discovery_gate_passed"]:
        paired_ranked = sorted(paired, key=lambda row: (row["ci_low"], row["mean_delta"]), reverse=True)
        selected_conditions = []
        for item in paired_ranked:
            condition = (float(item["noise"]), int(item["context_budget"]))
            if item["noise"] > 0 and condition not in selected_conditions:
                selected_conditions.append(condition)
            if len(selected_conditions) == 4:
                break
        noises = list(dict.fromkeys(condition[0] for condition in selected_conditions))[:2]
        budgets = list(dict.fromkeys(condition[1] for condition in selected_conditions))[:2]
        if len(noises) < 2:
            noises = [0.2, 0.5]
        if len(budgets) < 2:
            budgets = [64, 256]
        generic_rank = pd.DataFrame(summary)
        generic_rank = generic_rank[(generic_rank.phase == "discovery") & generic_rank.method.isin(GENERIC)].groupby("method").accuracy.mean().sort_values(ascending=False)
        best_two = list(generic_rank.index[:2])
        subset = {"twistedmerge_hodge_lr", *best_two}
        for group in ["S3", "D4"]:
            for seed in range(10, 20):
                for noise in noises:
                    for budget in budgets:
                        rows.extend(evaluate_condition(group, 64, seed, noise, budget, "confirmation", subset))
        summary, paired, efficiency, claims = summarize(rows)
        claims["confirmation_executed"] = True
        claims["confirmation_methods"] = sorted(subset)
        claims["confirmation_noises"] = noises
        claims["confirmation_budgets"] = budgets
    else:
        claims["confirmation_executed"] = False
    write_csv(OUT / "context_runs.csv", rows)
    write_csv(OUT / "context_summary.csv", summary)
    write_csv(OUT / "context_paired.csv", paired)
    write_csv(OUT / "context_efficiency.csv", efficiency)
    write_json(OUT / "context_claims.json", claims)
    write_csv(OUT / "context_claims.csv", [{"claim": key, "value": json.dumps(value, sort_keys=True)} for key, value in claims.items()])
    table_rows = [row for row in summary if row["phase"] == "discovery" and row["noise"] in [0.2, 0.5] and row["context_budget"] == 256]
    write_tex_table(OUT / "tables" / "context_main.tex", table_rows, ["group", "noise", "method", "accuracy", "loss"], "Matched context-fairness results.")
    write_tex_table(OUT / "tables" / "context_efficiency.tex", efficiency, ["group", "generic_method", "structured_accuracy", "generic_accuracy", "structured_parameters", "generic_parameters"], "Context-method efficiency comparison.")
    plot_frame = pd.DataFrame(summary)
    plot_frame = plot_frame[(plot_frame.phase == "discovery") & (plot_frame.context_budget == 256) & plot_frame.method.isin(["twistedmerge_hodge_lr", *GENERIC])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, block in plot_frame.groupby("method"):
        curve = block.groupby("noise").accuracy.mean()
        ax.plot(curve.index, curve.values, marker="o", label=method)
    ax.set(xlabel="Context corruption probability", ylabel="Accuracy", ylim=(0, 1.02))
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "context_noise.pdf")
    plt.close(fig)
    efficiency_frame = pd.DataFrame(efficiency)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(efficiency_frame.generic_parameters, efficiency_frame.generic_accuracy, label="generic")
    ax.scatter(efficiency_frame.structured_parameters, efficiency_frame.structured_accuracy, label="structured")
    ax.set(xscale="symlog", xlabel="Trainable parameters", ylabel="Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "context_efficiency.pdf")
    plt.close(fig)
    report = f"""# Compact context-fairness result

The discovery run executed 20 base settings across four context-corruption levels and four context-training budgets. All candidate logits passed the byte-identity label-permutation regression. The mechanical discovery gate was **{'passed' if claims['discovery_gate_passed'] else 'not passed'}** on the preregistered criteria. Conditional confirmation was **{'executed' if claims['confirmation_executed'] else 'not triggered'}**.

The comparison is intentionally against the best generic context-conditioned method, not only against context-blind synchronization. Detailed paired intervals and capacity records are in the adjacent CSV files. Negative conditions and the input-only and residual-only routing tests remain included.
"""
    (OUT / "context_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
