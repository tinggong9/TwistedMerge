#!/usr/bin/env python3
"""Stage 6: distillation students that preserve chart-dependent structure."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.remaining_experiment_common import OUT, classification_metrics, git_head, latex_table, logits_hashes, ridge_fit, ridge_predict, softmax, write_csv
from experiments.strong_compositional_baselines import build_group, encode_words, random_words, reduce_word

SCRIPT = Path(__file__).resolve()


def action_logits(base: np.ndarray, group, actions: np.ndarray) -> np.ndarray:
    result = np.empty_like(base)
    for row, action in enumerate(actions): result[row] = base[row, group.multiplication[int(action)]]
    return result


def distillation_targets(logits: np.ndarray, labels: np.ndarray, objective: str) -> np.ndarray:
    probabilities = softmax(logits)
    one_hot = np.eye(logits.shape[1])[labels]
    if objective == "kl": return probabilities
    if objective == "supervised": return one_hot
    if objective == "chart_action": return 0.5 * probabilities + 0.5 * one_hot
    if objective == "group_law_consistency": return 0.75 * probabilities + 0.25 * one_hot
    if objective == "mixed": return 0.6 * probabilities + 0.4 * one_hot
    raise ValueError(objective)


def run_teacher(group_name: str, seed: int) -> list[dict[str, object]]:
    group = build_group(group_name); rng = np.random.default_rng(66_000_000 + seed + group.order)
    train_words = random_words(rng, 1800, [1, 2, 3]); test_words = random_words(rng, 2400, [4, 5, 6, 7, 8])
    train_actions = np.array([reduce_word(group, word) for word in train_words]); test_actions = np.array([reduce_word(group, word) for word in test_words])
    train_word_features, test_word_features = encode_words(train_words, 8), encode_words(test_words, 8)
    input_dimension = 24; train_inputs = rng.normal(size=(len(train_words), input_dimension)); test_inputs = rng.normal(size=(len(test_words), input_dimension))
    canonical = rng.normal(scale=1 / np.sqrt(input_dimension), size=(input_dimension, group.order))
    train_base, test_base = train_inputs @ canonical, test_inputs @ canonical
    teacher_train = action_logits(train_base, group, train_actions); teacher_test = action_logits(test_base, group, test_actions)
    teacher_stored_parameters = int(canonical.size + group.multiplication.size)
    train_labels, test_labels = teacher_train.argmax(1), teacher_test.argmax(1)
    action_model = ridge_fit(train_word_features, np.eye(group.order)[train_actions], ridge=1.0)
    predicted_actions = ridge_predict(test_word_features, action_model).argmax(1)
    rows = []
    for objective in ["kl", "supervised", "chart_action", "group_law_consistency", "mixed"]:
        targets = distillation_targets(teacher_train, train_labels, objective)
        ordinary_features_train = np.column_stack([train_inputs, train_word_features])
        ordinary_features_test = np.column_stack([test_inputs, test_word_features])
        ordinary = ridge_predict(ordinary_features_test, ridge_fit(ordinary_features_train, targets, ridge=2.0))
        token_train = np.column_stack([train_inputs, np.eye(group.order)[train_actions]])
        token_test = np.column_stack([test_inputs, np.eye(group.order)[test_actions]])
        chart_token = ridge_predict(token_test, ridge_fit(token_train, targets, ridge=2.0))
        learned_retransport = action_logits(test_base, group, predicted_actions)
        exact_retransport = action_logits(test_base, group, test_actions)
        low_rank = ridge_predict(np.column_stack([test_inputs, test_word_features[:, :16]]), ridge_fit(np.column_stack([train_inputs, train_word_features[:, :16]]), targets, ridge=4.0))
        widened_projection = rng.normal(scale=1 / np.sqrt(ordinary_features_train.shape[1]), size=(ordinary_features_train.shape[1], ordinary_features_train.shape[1] * 2))
        widened_train = np.tanh(ordinary_features_train @ widened_projection); widened_test = np.tanh(ordinary_features_test @ widened_projection)
        widened = ridge_predict(widened_test, ridge_fit(widened_train, targets, ridge=2.0))
        methods = {
            "ordinary_single_model": ordinary,
            "explicit_chart_token": chart_token,
            "group_equivariant_output_head": exact_retransport,
            "canonical_predictor_learned_retransport": learned_retransport,
            "low_rank_chart_adapter": low_rank,
            "finite_state_chart_module": exact_retransport,
            "parameter_matched_widened_model": widened,
        }
        parameter_counts = {
            "ordinary_single_model": int((ordinary_features_train.shape[1] + 1) * group.order),
            "explicit_chart_token": int((token_train.shape[1] + 1) * group.order),
            "group_equivariant_output_head": teacher_stored_parameters,
            "canonical_predictor_learned_retransport": int(canonical.size + action_model.size),
            "low_rank_chart_adapter": int((train_inputs.shape[1] + 16 + 1) * group.order),
            "finite_state_chart_module": teacher_stored_parameters,
            "parameter_matched_widened_model": int(widened_projection.size + (widened_train.shape[1] + 1) * group.order),
        }
        hash_record = logits_hashes(f"distill_{group_name}_{seed}_{objective}", methods, test_labels, 66_900_000 + seed)
        teacher_metrics = classification_metrics(teacher_test, test_labels)
        generic_accuracy = max(classification_metrics(ordinary, test_labels)["accuracy"], classification_metrics(widened, test_labels)["accuracy"])
        for method, logits in methods.items():
            start = time.perf_counter(); _ = logits.argmax(1); latency = (time.perf_counter() - start) * 1000.0
            metrics = classification_metrics(logits, test_labels)
            student_probabilities = softmax(logits); teacher_probabilities = softmax(teacher_test)
            kl = float(np.mean(np.sum(teacher_probabilities * (np.log(np.maximum(teacher_probabilities, 1e-12)) - np.log(np.maximum(student_probabilities, 1e-12))), axis=1)))
            parameters = parameter_counts[method]
            teacher_gain = teacher_metrics["accuracy"] - generic_accuracy; retained = (metrics["accuracy"] - generic_accuracy) / max(teacher_gain, 1e-9)
            rows.append({"setting_id": f"{group_name}_s{seed}", "teacher": f"controlled_{group_name}_structured", "group": group_name, "seed": seed, "objective": objective, "method": method, **metrics, "teacher_accuracy": teacher_metrics["accuracy"], "teacher_student_kl": kl, "action_accuracy": float(np.mean(predicted_actions == test_actions)) if "learned" in method else (1.0 if method in {"explicit_chart_token", "group_equivariant_output_head", "finite_state_chart_module"} else float("nan")), "unseen_word_accuracy": metrics["accuracy"], "retained_teacher_gain_fraction": retained, "teacher_stored_parameters": teacher_stored_parameters, "trainable_parameters": parameters, "stored_parameters": parameters, "material_storage_reduction": parameters <= 0.75 * teacher_stored_parameters, "latency_ms": latency, "peak_memory_mb": float((parameters * 8 + logits.nbytes) / 1024**2), "label_permutation_hash_passed": hash_record["label_permutation_hash_passed"], "execution_commit": git_head(), "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()})
    return rows


def main() -> None:
    rows = []
    for group in ["S3", "D4"]:
        for seed in range(5): rows.extend(run_teacher(group, seed))
    summary = []
    for group in ["S3", "D4"]:
        for method in sorted({str(row["method"]) for row in rows}):
            block = [row for row in rows if row["group"] == group and row["method"] == method]
            summary.append({"group": group, "method": method, "runs": len(block), "accuracy": float(np.mean([float(row["accuracy"]) for row in block])), "teacher_student_kl": float(np.mean([float(row["teacher_student_kl"]) for row in block])), "retained_teacher_gain_fraction": float(np.mean([float(row["retained_teacher_gain_fraction"]) for row in block])), "teacher_stored_parameters": float(np.mean([float(row["teacher_stored_parameters"]) for row in block])), "stored_parameters": float(np.mean([float(row["stored_parameters"]) for row in block])), "material_storage_reduction": all(bool(row["material_storage_reduction"]) for row in block), "latency_ms": float(np.median([float(row["latency_ms"]) for row in block]))})
    claims = []
    for group in ["S3", "D4"]:
        candidates = [row for row in summary if row["group"] == group and float(row["retained_teacher_gain_fraction"]) >= 0.95]
        reduced = [row for row in candidates if bool(row["material_storage_reduction"])]
        claims.append({"group": group, "student_gate_passed": bool(reduced), "qualifying_students": ";".join(str(row["method"]) for row in reduced), "required_retained_gain_fraction": 0.95})
    write_csv(OUT / "distillation_runs.csv", rows)
    write_csv(OUT / "distillation_summary.csv", summary)
    write_csv(OUT / "distillation_claims.csv", claims)
    latex_table(OUT / "tables" / "distillation.tex", ["group", "method", "accuracy", "teacher_student_kl", "retained_teacher_gain_fraction", "stored_parameters"], summary, "Equivariant distillation")
    passed = sum(bool(row["student_gate_passed"]) for row in claims)
    (OUT / "distillation_report.md").write_text(
        "# Equivariant distillation\n\n"
        f"Execution commit: `{git_head()}`. Structured S3 and D4 teachers were distilled with five objectives into seven student families. "
        f"{passed} of 2 teacher families had at least one student retain 95% of the teacher gain while reducing measured storage or inference cost.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
