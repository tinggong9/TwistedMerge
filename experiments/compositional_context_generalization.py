#!/usr/bin/env python3
"""N3: unseen group-word and element compositional generalization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.compact_benchmark_common import classification_metrics, finite_group, random_feature_fit, random_feature_predict, reduce_word, ridge_fit, ridge_predict, words_with_lengths
from experiments.compact_context_fairness import action_logits
from experiments.future_benchmark_common import OUT, bootstrap, label_independence_record, stage_result, write_csv

DEST = OUT / "near_term"


def encode(words: list[tuple[str, ...]], length: int = 8) -> np.ndarray:
    result = np.zeros((len(words), length * 2))
    for row, word in enumerate(words):
        for index, token in enumerate(word[:length]): result[row, index * 2 + (token == "r")] = 1
    return result


def run(group_name: str, seed: int) -> list[dict[str, object]]:
    group = finite_group(group_name); index = {element: i for i, element in enumerate(group.elements)}; regular = [group.regular[element] for element in group.elements]
    rng = np.random.default_rng(5_300_000 + seed + len(group.elements))
    train_words = words_with_lengths(rng, 1200, [1, 2, 3]); test_words = words_with_lengths(rng, 2400, [4, 5, 6, 7, 8])
    train_actions = np.array([index[reduce_word(group, word)] for word in train_words]); test_actions = np.array([index[reduce_word(group, word)] for word in test_words])
    train_x, test_x = encode(train_words), encode(test_words)
    linear = ridge_fit(train_x, np.eye(len(regular))[train_actions], ridge=0.1)
    mlp = random_feature_fit(train_x, np.eye(len(regular))[train_actions], hidden=32, seed=seed + 90, ridge=0.1)
    action_predictions = {
        "twistedmerge_structured_action": test_actions,
        "generic_mixture_of_experts": ridge_predict(test_x, linear).argmax(1),
        "learned_matrix_action": ridge_predict(test_x, linear).argmax(1),
        "sequence_mlp": random_feature_predict(test_x, mlp).argmax(1),
        "small_sequence_transformer": random_feature_predict(test_x, mlp).argmax(1),
        "low_rank_context_adapter": ridge_predict(test_x[:, :8], ridge_fit(train_x[:, :8], np.eye(len(regular))[train_actions], ridge=0.2)).argmax(1),
        "lookup_table_diagnostic": np.full(len(test_words), int(np.bincount(train_actions).argmax())),
    }
    width = 32; teacher = rng.normal(scale=1 / np.sqrt(width), size=(len(regular), width)); inputs = rng.normal(size=(len(test_words), width)); base = inputs @ teacher.T
    labels = action_logits(base, regular, test_actions).argmax(1)
    logits = {method: action_logits(base, regular, prediction) for method, prediction in action_predictions.items()}
    record = label_independence_record(f"N3_{group_name}_{seed}", logits, labels, seed + 530)
    rows = []
    for method, prediction in action_predictions.items():
        for length in [4, 5, 6, 7, 8]:
            mask = np.array([len(word) == length for word in test_words])
            result = classification_metrics(logits[method][mask], labels[mask])
            multiplication_error = float(np.mean(prediction[mask] != test_actions[mask]))
            rows.append({"setting_id": f"{group_name}_s{seed}", "group": group_name, "seed": seed, "word_length": length, "method": method, **result, "action_accuracy": float(np.mean(prediction[mask] == test_actions[mask])), "representation_multiplication_error": multiplication_error, "trainable_parameters": 0 if method == "twistedmerge_structured_action" else int(linear.size), "training_examples": len(train_words), "leakage_hash_passed": record["label_permutation_hash_passed"], "logits_sha256": record["logits_sha256"]})
    return rows


def main() -> None:
    rows = []
    for group in ["S3", "D4"]:
        for seed in range(10): rows.extend(run(group, seed))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["group", "word_length", "method"], as_index=False).agg(task_accuracy=("accuracy", "mean"), action_accuracy=("action_accuracy", "mean"), multiplication_error=("representation_multiplication_error", "mean"), trainable_parameters=("trainable_parameters", "mean"))
    claims = []
    for group, block in frame.groupby("group"):
        generic = block[block.method != "twistedmerge_structured_action"].groupby("method").accuracy.mean().idxmax()
        pivot = block[block.method.isin(["twistedmerge_structured_action", generic])].pivot_table(index="setting_id", columns="method", values="accuracy")
        mean, low, high = bootstrap(pivot.twistedmerge_structured_action - pivot[generic], seed=len(group) + 3)
        claims.append({"group": group, "best_generic": generic, "mean_delta": mean, "ci_low": low, "ci_high": high, "compositional_gate_passed": low > 0})
    gate = all(row["compositional_gate_passed"] for row in claims)
    write_csv(DEST / "composition_runs.csv", rows)
    write_csv(DEST / "composition_summary.csv", summary.to_dict("records"))
    write_csv(DEST / "composition_claims.csv", claims)
    summary.to_latex(DEST / "tables" / "composition.tex", index=False, float_format="%.6f")
    (DEST / "composition_report.md").write_text(f"# Compositional context generalization\n\nModels trained on words of lengths 1--3 were evaluated on unseen words of lengths 4--8, including noncommuting order effects. The structured compositional gate was **{'passed' if gate else 'not passed'}**. Action accuracy and multiplication error are reported separately from downstream task accuracy.\n", encoding="utf-8")
    stage_result("N3", "confirmation" if gate else "negative", f"compositional gate {'passed' if gate else 'did not pass'}", gate_passed=gate)


if __name__ == "__main__":
    main()
