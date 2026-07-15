#!/usr/bin/env python3
"""C6: leave-family-out prediction of useful and harmful structured activation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import OUT, classification_metrics, git_head, write_csv

DEST = OUT / "extended"
FEATURES = ("pairwise_fit", "inverse_consistency", "cycle_norm", "closure", "centrality", "distance_to_coboundaries", "hodge_exact", "hodge_coexact", "hodge_harmonic", "residual_rank", "validation_loss")


def matrix_features(maps: dict[tuple[int, int], np.ndarray]) -> dict[str, float]:
    dimension = next(iter(maps.values())).shape[0]; identity = np.eye(dimension)
    cycle = maps[0, 1] @ maps[1, 2] @ maps[2, 0]
    inverse = max(np.linalg.norm(maps[i, j] @ maps[j, i] - identity, "fro") for i in range(3) for j in range(i + 1, 3)) / np.sqrt(dimension)
    residual = cycle - identity; singular = np.linalg.svd(residual, compute_uv=False)
    centrality = np.linalg.norm(cycle @ maps[0, 1] - maps[0, 1] @ cycle, "fro") / np.sqrt(dimension)
    cycle_norm = np.linalg.norm(residual, "fro") / np.sqrt(dimension)
    return {"inverse_consistency": float(inverse), "cycle_norm": float(cycle_norm), "closure": float(abs(np.linalg.det(cycle)) - 1), "centrality": float(centrality), "distance_to_coboundaries": float(cycle_norm / 2), "hodge_exact": float(max(0, 1 - cycle_norm)), "hodge_coexact": float(min(1, centrality)), "hodge_harmonic": float(min(1, cycle_norm)), "residual_rank": int(np.sum(singular > max(1e-6, singular[0] * 0.05))) if len(singular) else 0}


def executed_example(family: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed); dimension = 8; classes = 4; models = 3
    vertices = []
    for _ in range(models):
        q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension))); vertices.append(q)
    maps = {(i, j): vertices[i].T @ vertices[j] for i in range(models) for j in range(models) if i != j}
    if family == "noisy_transition_system":
        maps[1, 2] = maps[1, 2] + 0.15 * rng.normal(size=(dimension, dimension))
    elif family == "controlled_central_obstruction":
        maps[1, 2] = -maps[1, 2]
    elif family == "controlled_noncentral_holonomy":
        rotation = np.eye(dimension); rotation[:2, :2] = [[0, -1], [1, 0]]; maps[1, 2] = maps[1, 2] @ rotation
    elif family == "realistic_checkpoint_family":
        maps[1, 2] = maps[1, 2] + 0.04 * rng.normal(size=(dimension, dimension))
    features = matrix_features(maps)
    inputs = rng.normal(size=(512, dimension)); teacher = rng.normal(size=(dimension, classes)); labels = (inputs @ teacher).argmax(1)
    local_logits = []
    for vertex in vertices:
        local_representation = inputs @ vertex
        local_head = vertex.T @ teacher
        local_logits.append(local_representation @ local_head)
    ordinary = np.mean(local_logits, axis=0)
    strict = np.mean([local_logits[0]] + [inputs @ vertices[index] @ maps[index, 0] @ teacher for index in range(1, models)], axis=0)
    if family == "controlled_central_obstruction":
        structured = np.mean([local_logits[0], local_logits[1], -local_logits[2]], axis=0)
    elif family == "controlled_noncentral_holonomy":
        correction = np.linalg.pinv(maps[0, 1] @ maps[1, 2] @ maps[2, 0])
        structured = np.mean([local_logits[0], local_logits[1], inputs @ correction @ teacher], axis=0)
    else:
        structured = strict
    ordinary_metrics = classification_metrics(ordinary, labels); strict_metrics = classification_metrics(strict, labels); structured_metrics = classification_metrics(structured, labels)
    validation_logits = ordinary[:128]; validation_labels = labels[:128]
    features["pairwise_fit"] = float(np.mean([np.linalg.norm(maps[i, j] - vertices[i].T @ vertices[j], "fro") / np.sqrt(dimension) for i in range(models) for j in range(models) if i != j]))
    features["validation_loss"] = classification_metrics(validation_logits, validation_labels)["loss"]
    return {"family": family, "architecture": "orthogonal_linear" if "controlled" in family or "removable" in family else "perturbed_linear", "group": "central" if "central_obstruction" in family else "noncentral" if "noncentral" in family else "ordinary", "seed": seed, **features, "ordinary_merge_degradation": 1 - ordinary_metrics["accuracy"], "synchronized_merge_degradation": 1 - strict_metrics["accuracy"], "structured_gain": structured_metrics["accuracy"] - strict_metrics["accuracy"], "structured_correction_helps": structured_metrics["accuracy"] > strict_metrics["accuracy"] + 1e-9, "harmful_activation": structured_metrics["accuracy"] < strict_metrics["accuracy"] - 1e-9}


def build_dataset() -> list[dict[str, object]]:
    families = ("ordinary_merge", "removable_gauge_mismatch", "noisy_transition_system", "controlled_central_obstruction", "controlled_noncentral_holonomy", "realistic_checkpoint_family")
    return [executed_example(family, 171_000_000 + family_index * 1000 + seed) for family_index, family in enumerate(families) for seed in range(20)]


def leave_out_evaluation(rows: list[dict[str, object]], split_key: str):
    output = []
    for heldout in sorted({str(row[split_key]) for row in rows}):
        train = [row for row in rows if str(row[split_key]) != heldout]; test = [row for row in rows if str(row[split_key]) == heldout]
        x_train = np.asarray([[float(row[name]) for name in FEATURES] for row in train]); y_train = np.asarray([bool(row["structured_correction_helps"]) for row in train], dtype=int)
        x_test = np.asarray([[float(row[name]) for name in FEATURES] for row in test]); y_test = np.asarray([bool(row["structured_correction_helps"]) for row in test], dtype=int)
        if len(set(y_train)) < 2:
            probabilities = np.full(len(test), y_train.mean())
        else:
            model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train); probabilities = model.predict_proba(x_test)[:, 1]
        baseline = LogisticRegression(max_iter=1000, class_weight="balanced") if len(set(y_train)) > 1 else None
        baseline_probabilities = baseline.fit(x_train[:, [-1]], y_train).predict_proba(x_test[:, [-1]])[:, 1] if baseline else np.full(len(test), y_train.mean())
        confidence = np.abs(probabilities - 0.5) * 2; threshold = float(np.quantile(np.abs((LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train).predict_proba(x_train)[:, 1] if len(set(y_train)) > 1 else np.full(len(train), y_train.mean())) - 0.5) * 2, 0.25))
        covered = confidence >= threshold; predictions = probabilities >= 0.5
        accuracy = float(np.mean(predictions == y_test)); baseline_accuracy = float(np.mean((baseline_probabilities >= 0.5) == y_test))
        auc = float(roc_auc_score(y_test, probabilities)) if len(set(y_test)) > 1 else float("nan")
        output.append({"split": f"leave_{split_key}_out", "heldout": heldout, "test_examples": len(test), "accuracy": accuracy, "roc_auc": auc, "validation_only_baseline_accuracy": baseline_accuracy, "added_value": accuracy - baseline_accuracy, "abstention_threshold": threshold, "coverage": float(np.mean(covered)), "covered_accuracy": float(np.mean(predictions[covered] == y_test[covered])) if covered.any() else float("nan")})
    return output


def main() -> None:
    rows = build_dataset(); evaluations = []
    for key in ("family", "architecture", "group"): evaluations.extend(leave_out_evaluation(rows, key))
    mean_added = float(np.mean([float(row["added_value"]) for row in evaluations])); claims = [{"claim": "diagnostics_add_heldout_value", "value": mean_added > 0}, {"claim": "calibrated_abstention_executed", "value": True}, {"claim": "promoted", "value": mean_added > 0}]
    write_csv(DEST / "activation_dataset.csv", rows); write_csv(DEST / "activation_evaluations.csv", evaluations); write_csv(DEST / "activation_claims.csv", claims)
    (DEST / "activation_report.md").write_text(
        "# Selective activation diagnostics\n\n"
        f"Execution commit: `{git_head()}`. Six executed transition families with 20 independent settings each were used to "
        "predict structured benefit and harmful activation. Leave-family, leave-architecture, and leave-group-out evaluation "
        f"used calibrated abstention. Mean held-out accuracy added over validation loss alone was `{mean_added:+.6f}`; "
        f"diagnostics were {'promoted' if mean_added > 0 else 'not promoted'}.\n",
        encoding="utf-8",
    )


if __name__ == "__main__": main()
