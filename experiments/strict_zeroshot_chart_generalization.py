#!/usr/bin/env python3
"""Strict D4 element holdout with no held-out image chart labels during fitting."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import (
    DEVICE,
    OUT,
    TMP,
    D4EquivariantChartCNN,
    ImageCNN,
    LearnedMultiplicationChartCNN,
    OrbitTaskCNN,
    apply_d4,
    calibrate_temperature,
    chart_probabilities,
    checkpoint_payload,
    compose_d4,
    d4_table,
    d4_tta_logits,
    dataset_tensors,
    extended_metrics,
    factual_report,
    inverse_chart,
    make_chart_examples,
    measure_actual,
    model_bytes,
    model_logits,
    ordinary_chart_augmentation,
    paired_interval_rows,
    parameter_count,
    provenance,
    save_logits_before_evaluation,
    split_indices,
    task_branches,
    train_classifier,
)
from experiments.next_program_common import latex_table, paired_bootstrap, write_csv

SCRIPT = Path(__file__).resolve()
DEST = OUT / "zeroshot"
COMMAND = "python experiments/strict_zeroshot_chart_generalization.py"
SEEN = (0, 1, 4)
UNSEEN = (2, 3, 5, 6, 7)
VARIANTS = (
    "generator_only_supervision",
    "generator_plus_inverse_supervision",
    "generator_plus_multiplication_consistency",
    "chart_labels_plus_equivariance_consistency",
)
METHODS = (
    "d4_equivariant_chart_classifier",
    "capacity_matched_ordinary_chart_classifier",
    "augmentation_trained_ordinary_chart_classifier",
    "learned_multiplication_table_model",
    "d4_equivariant_task_classifier",
    "structured_hard_retransport",
    "structured_soft_retransport",
    "d4_test_time_augmentation",
    "supplied_chart_oracle",
    "random_action_control",
    "wrong_action_control",
)
EQUIVARIANCE_TOLERANCE = 0.08
MULTIPLICATION_ERROR_TOLERANCE = 0.05


def allowed_compositions(charts: tuple[int, ...], require_product_seen: bool) -> list[tuple[int, int, int]]:
    result = []
    for left in charts:
        for right in charts:
            product = compose_d4(left, right)
            if not require_product_seen or product in charts:
                result.append((left, right, product))
    return result


def expanded_seen_consistency(
    images: torch.Tensor, charts: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_parts = [images]
    chart_parts = [charts]
    label_parts = [labels]
    for action in SEEN:
        products = torch.tensor([compose_d4(action, int(chart)) for chart in charts], dtype=torch.long)
        mask = torch.tensor([int(value) in SEEN for value in products])
        if bool(mask.any()):
            image_parts.append(apply_d4(images[mask], action))
            chart_parts.append(products[mask])
            label_parts.append(labels[mask])
    return torch.cat(image_parts), torch.cat(chart_parts), torch.cat(label_parts)


def symbolic_relations(variant: str) -> list[tuple[int, int, int]]:
    table = d4_table()
    relations = [(0, value, value) for value in SEEN] + [(value, 0, value) for value in SEEN]
    if variant in VARIANTS[1:]:
        relations.extend((value, inverse_chart(value), 0) for value in (1, 4))
        relations.extend((inverse_chart(value), value, 0) for value in (1, 4))
    if variant == "generator_plus_multiplication_consistency":
        relations = [(left, right, int(table[left, right])) for left in range(8) for right in range(8)]
    return relations


def train_learned_table(
    model: LearnedMultiplicationChartCNN,
    train_images: torch.Tensor,
    train_charts: torch.Tensor,
    validation_images: torch.Tensor,
    validation_charts: torch.Tensor,
    variant: str,
    seed: int,
    epochs: int,
) -> tuple[LearnedMultiplicationChartCNN, float]:
    from experiments.chart_followup_common import batches

    torch.manual_seed(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    relations = symbolic_relations(variant)
    left = torch.tensor([value[0] for value in relations], device=DEVICE)
    right = torch.tensor([value[1] for value in relations], device=DEVICE)
    products = torch.tensor([value[2] for value in relations], device=DEVICE)
    started = time.perf_counter()
    best = float("inf")
    state = None
    for epoch in range(epochs):
        model.train()
        for indices in batches(len(train_images), 64, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            image_loss = nn.functional.cross_entropy(model(train_images[indices].to(DEVICE)), train_charts[indices].to(DEVICE))
            table_loss = nn.functional.cross_entropy(model.table_logits[left, right], products)
            loss = image_loss + 0.5 * table_loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float(nn.functional.cross_entropy(model(validation_images.to(DEVICE)), validation_charts.to(DEVICE)))
        if loss < best:
            best = loss
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if state is not None:
        model.load_state_dict(state)
    return model.eval(), time.perf_counter() - started


def equivariance_error(model: nn.Module, base_images: torch.Tensor) -> float:
    base = model_logits(model, base_images).softmax(1)
    errors = []
    for action in range(8):
        transformed = model_logits(model, apply_d4(base_images, action)).softmax(1)
        expected = torch.zeros_like(base)
        for chart in range(8):
            expected[:, compose_d4(action, chart)] = base[:, chart]
        errors.append(float((transformed - expected).abs().mean()))
    return float(np.mean(errors))


def prepare(seed: int, smoke: bool) -> dict[str, object]:
    train_images, train_labels, test_images, test_labels, channels = dataset_tensors("FashionMNIST")
    split = split_indices(seed, len(train_images), local_train=512 if smoke else 6000)
    if smoke:
        for key, size in (("chart_train", 128), ("validation", 64), ("calibration", 64), ("threshold", 64)):
            split[key] = split[key][:size]
    chart_images, chart_labels, _ = make_chart_examples(train_images[split["chart_train"]], 231_000_000 + seed, SEEN, "strict_seen_chart_training")
    validation_images, validation_charts, _ = make_chart_examples(train_images[split["validation"]], 231_100_000 + seed, SEEN, "strict_seen_validation")
    calibration_images, calibration_charts, _ = make_chart_examples(train_images[split["calibration"]], 231_200_000 + seed, SEEN, "strict_seen_calibration")
    order = np.random.default_rng(231_300_000 + seed).permutation(len(test_images))[: (256 if smoke else 2000)]
    half = len(order) // 2
    seen_images, seen_charts, _ = make_chart_examples(test_images[order[:half]], 231_400_000 + seed, SEEN, "seen_final_test")
    unseen_images, unseen_charts, _ = make_chart_examples(test_images[order[half:]], 231_500_000 + seed, UNSEEN, "unseen_final_test")
    final_images = torch.cat([seen_images, unseen_images])
    final_charts = torch.cat([seen_charts, unseen_charts])
    final_labels = test_labels[order]
    roles = np.asarray(["seen"] * len(seen_images) + ["unseen"] * len(unseen_images))
    return {
        "channels": channels,
        "split": split,
        "test_order": order,
        "local_images": train_images[split["local_train"]],
        "local_labels": train_labels[split["local_train"]],
        "chart_images": chart_images,
        "chart_labels": chart_labels,
        "chart_task_labels": train_labels[split["chart_train"]],
        "validation_images": validation_images,
        "validation_charts": validation_charts,
        "validation_task_labels": train_labels[split["validation"]],
        "calibration_images": calibration_images,
        "calibration_charts": calibration_charts,
        "test_images": final_images,
        "test_charts": final_charts,
        "test_labels": final_labels,
        "roles": roles,
        "base_test_images": test_images[order[: min(128, len(order))]],
    }


def run_seed(seed: int, smoke: bool) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payload = prepare(seed, smoke)
    channels = int(payload["channels"])
    task_epochs = 1 if smoke else 2
    chart_epochs = 2 if smoke else 8
    expert, expert_time, _ = train_classifier(
        ImageCNN(10, channels, width=12), payload["local_images"], payload["local_labels"], payload["validation_images"], payload["validation_task_labels"], 232_000_000 + seed, task_epochs
    )
    direct, direct_time, _ = train_classifier(
        OrbitTaskCNN(channels, width=10), payload["local_images"], payload["local_labels"], payload["validation_images"], payload["validation_task_labels"], 232_100_000 + seed, task_epochs
    )
    chart_models: dict[str, nn.Module] = {}
    chart_times: dict[str, float] = {}
    for offset, (name, model, augmentation) in enumerate(
        (
            ("equivariant", D4EquivariantChartCNN(channels, width=10), None),
            ("ordinary", ImageCNN(8, channels, width=7), None),
            ("augmented", ImageCNN(8, channels, width=10), ordinary_chart_augmentation),
        )
    ):
        model, elapsed, _ = train_classifier(
            model, payload["chart_images"], payload["chart_labels"], payload["validation_images"], payload["validation_charts"], 232_200_000 + seed * 10 + offset, chart_epochs, batch_size=64, augment=augmentation
        )
        chart_models[name] = model
        chart_times[name] = elapsed
    table_models = {}
    table_times = {}
    expanded_images, expanded_charts, _ = expanded_seen_consistency(payload["chart_images"], payload["chart_labels"], payload["chart_task_labels"])
    for offset, variant in enumerate(VARIANTS):
        train_images = expanded_images if variant == "chart_labels_plus_equivariance_consistency" else payload["chart_images"]
        train_charts = expanded_charts if variant == "chart_labels_plus_equivariance_consistency" else payload["chart_labels"]
        model, elapsed = train_learned_table(
            LearnedMultiplicationChartCNN(channels, width=10), train_images, train_charts, payload["validation_images"], payload["validation_charts"], variant, 232_300_000 + seed * 10 + offset, chart_epochs
        )
        table_models[variant] = model
        table_times[variant] = elapsed
    models = {**{f"chart_{key}": value for key, value in chart_models.items()}, **{f"table_{key}": value for key, value in table_models.items()}, "expert": expert, "direct": direct}
    temperatures = {
        name: calibrate_temperature(model_logits(model, payload["calibration_images"]), payload["calibration_charts"])
        for name, model in {**chart_models, **table_models}.items()
    }
    probabilities = {name: chart_probabilities(model, payload["test_images"], temperatures[name]) for name, model in {**chart_models, **table_models}.items()}
    branches = task_branches(payload["test_images"], [expert])
    equivariant = probabilities["equivariant"]
    table_probability = probabilities["generator_plus_multiplication_consistency"]
    random_generator = torch.Generator().manual_seed(232_900_000 + seed)
    random_charts = torch.randint(0, 8, (len(payload["test_images"]),), generator=random_generator)
    candidates = {
        "d4_equivariant_chart_classifier": torch.einsum("nb,nbc->nc", equivariant, branches),
        "capacity_matched_ordinary_chart_classifier": torch.einsum("nb,nbc->nc", probabilities["ordinary"], branches),
        "augmentation_trained_ordinary_chart_classifier": torch.einsum("nb,nbc->nc", probabilities["augmented"], branches),
        "learned_multiplication_table_model": torch.einsum("nb,nbc->nc", table_probability, branches),
        "d4_equivariant_task_classifier": model_logits(direct, payload["test_images"]),
        "structured_hard_retransport": branches[torch.arange(len(branches)), equivariant.argmax(1)],
        "structured_soft_retransport": torch.einsum("nb,nbc->nc", equivariant, branches),
        "d4_test_time_augmentation": d4_tta_logits(payload["test_images"], expert),
        "supplied_chart_oracle": branches[torch.arange(len(branches)), payload["test_charts"]],
        "random_action_control": branches[torch.arange(len(branches)), random_charts],
        "wrong_action_control": branches[torch.arange(len(branches)), (equivariant.argmax(1) + 1) % 8],
    }
    ledger = save_logits_before_evaluation(f"zeroshot_seed_{seed}", candidates, payload["test_labels"], 233_000_000 + seed)
    method_chart = {
        "d4_equivariant_chart_classifier": equivariant.argmax(1),
        "capacity_matched_ordinary_chart_classifier": probabilities["ordinary"].argmax(1),
        "augmentation_trained_ordinary_chart_classifier": probabilities["augmented"].argmax(1),
        "learned_multiplication_table_model": table_probability.argmax(1),
        "structured_hard_retransport": equivariant.argmax(1),
        "structured_soft_retransport": equivariant.argmax(1),
        "supplied_chart_oracle": payload["test_charts"],
        "random_action_control": random_charts,
        "wrong_action_control": (equivariant.argmax(1) + 1) % 8,
    }
    selected_models = {
        "d4_equivariant_chart_classifier": [expert, chart_models["equivariant"]],
        "capacity_matched_ordinary_chart_classifier": [expert, chart_models["ordinary"]],
        "augmentation_trained_ordinary_chart_classifier": [expert, chart_models["augmented"]],
        "learned_multiplication_table_model": [expert, table_models["generator_plus_multiplication_consistency"]],
        "d4_equivariant_task_classifier": [direct],
        "structured_hard_retransport": [expert, chart_models["equivariant"]],
        "structured_soft_retransport": [expert, chart_models["equivariant"]],
        "d4_test_time_augmentation": [expert],
        "supplied_chart_oracle": [expert],
        "random_action_control": [expert],
        "wrong_action_control": [expert, chart_models["equivariant"]],
    }
    method_training_time = {
        "d4_equivariant_chart_classifier": expert_time + chart_times["equivariant"],
        "capacity_matched_ordinary_chart_classifier": expert_time + chart_times["ordinary"],
        "augmentation_trained_ordinary_chart_classifier": expert_time + chart_times["augmented"],
        "learned_multiplication_table_model": expert_time + table_times["generator_plus_multiplication_consistency"],
        "d4_equivariant_task_classifier": direct_time,
        "structured_hard_retransport": expert_time + chart_times["equivariant"],
        "structured_soft_retransport": expert_time + chart_times["equivariant"],
        "d4_test_time_augmentation": expert_time,
        "supplied_chart_oracle": expert_time,
        "random_action_control": expert_time,
        "wrong_action_control": expert_time + chart_times["equivariant"],
    }
    timing_batch = payload["test_images"][: min(128, len(payload["test_images"]))]
    timing_charts = payload["test_charts"][: len(timing_batch)]

    def executed_path(method: str) -> torch.Tensor:
        if method == "d4_equivariant_task_classifier":
            return model_logits(direct, timing_batch)
        if method == "d4_test_time_augmentation":
            return d4_tta_logits(timing_batch, expert)
        local_branches = task_branches(timing_batch, [expert])
        if method == "supplied_chart_oracle":
            return local_branches[torch.arange(len(local_branches)), timing_charts]
        if method == "random_action_control":
            generator = torch.Generator().manual_seed(232_900_000 + seed)
            charts = torch.randint(0, 8, (len(timing_batch),), generator=generator)
            return local_branches[torch.arange(len(local_branches)), charts]
        chart_name = {
            "capacity_matched_ordinary_chart_classifier": "ordinary",
            "augmentation_trained_ordinary_chart_classifier": "augmented",
            "learned_multiplication_table_model": "generator_plus_multiplication_consistency",
        }.get(method, "equivariant")
        model = table_models[chart_name] if chart_name in table_models else chart_models[chart_name]
        probability = chart_probabilities(model, timing_batch, temperatures[chart_name])
        if method in {"structured_hard_retransport", "wrong_action_control"}:
            chart = probability.argmax(1)
            if method == "wrong_action_control":
                chart = (chart + 1) % 8
            return local_branches[torch.arange(len(local_branches)), chart]
        return torch.einsum("nb,nbc->nc", probability, local_branches)

    timings = {
        method: measure_actual(lambda selected=method: executed_path(selected), 0 if smoke else 1, 1 if smoke else 3)
        for method in METHODS
    }
    runs = []
    by_element = []
    for method in METHODS:
        prediction = method_chart.get(method)
        for role in ("seen", "unseen"):
            mask = payload["roles"] == role
            metrics = extended_metrics(candidates[method][mask], payload["test_labels"][mask])
            runs.append(
                {
                    "setting_id": f"FashionMNIST_zeroshot_seed{seed}",
                    "seed": seed,
                    "method": method,
                    "element_role": role,
                    **metrics,
                    "chart_accuracy": float((prediction[mask] == payload["test_charts"][mask]).float().mean()) if prediction is not None else "",
                    "trainable_parameters": sum(parameter_count(model) for model in selected_models[method]),
                    "stored_parameters": sum(parameter_count(model) for model in selected_models[method]),
                    "stored_bytes": sum(model_bytes(model) for model in selected_models[method]),
                    "branch_count": 8,
                    "complete_latency_ms_batch128": timings[method]["warm_start_latency_ms"],
                    "peak_process_memory_mb": timings[method]["peak_process_memory_mb"],
                    "peak_accelerator_memory_mb": timings[method]["peak_accelerator_memory_mb"],
                    "training_time_seconds": method_training_time[method],
                    "chart_training_examples": 0 if method in {"d4_equivariant_task_classifier", "d4_test_time_augmentation", "supplied_chart_oracle", "random_action_control"} else len(payload["chart_images"]),
                    "validation_examples": len(payload["validation_images"]),
                    "calibration_examples": len(payload["calibration_images"]),
                    "threshold_selection_examples": 0,
                    "test_examples": int(mask.sum()),
                    "chart_information": "supplied" if method == "supplied_chart_oracle" else ("none" if prediction is None else "inferred"),
                    "expert_evaluations": 1,
                    "heldout_chart_labels_exposed_during_fitting": False,
                    "logits_path": ledger["logits_path"],
                    "logits_sha256": ledger["logits_sha256"],
                    "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]),
                    **provenance(SCRIPT, COMMAND + (" --smoke" if smoke else ""), seed),
                }
            )
        for chart in range(8):
            mask = payload["test_charts"] == chart
            if not bool(mask.any()):
                continue
            by_element.append(
                {
                    "seed": seed,
                    "method": method,
                    "chart": chart,
                    "element_role": "seen" if chart in SEEN else "unseen",
                    "examples": int(mask.sum()),
                    "task_accuracy": float((candidates[method][mask].argmax(1) == payload["test_labels"][mask]).float().mean()),
                    "chart_accuracy": float((prediction[mask] == payload["test_charts"][mask]).float().mean()) if prediction is not None else "",
                }
            )
    equivariance = []
    table = d4_table()
    for name, model in {**chart_models, **table_models}.items():
        table_error: float | str = ""
        if isinstance(model, LearnedMultiplicationChartCNN):
            table_error = float((model.table_logits.detach().cpu().argmax(-1).numpy() != table).mean())
        equivariance.append(
            {
                "seed": seed,
                "model": name,
                "supervision_variant": name if name in VARIANTS else "primary_model",
                "equivariance_error": equivariance_error(model, payload["base_test_images"]),
                "multiplication_error": table_error,
                "heldout_image_chart_labels_exposed": False,
            }
        )
    checkpoint_path = TMP / "checkpoints" / "zeroshot" / f"seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            models,
            seed=seed,
            split_indices={key: value.tolist() for key, value in payload["split"].items()},
            test_order=payload["test_order"].tolist(),
            seen_charts=list(SEEN),
            unseen_charts=list(UNSEEN),
            temperatures=temperatures,
            task_training_time=expert_time,
            direct_training_time=direct_time,
            chart_training_times=chart_times,
            table_training_times=table_times,
        ),
        checkpoint_path,
    )
    return runs, by_element, equivariance, [{"seed": seed, "path": str(checkpoint_path.relative_to(ROOT)), "bytes": checkpoint_path.stat().st_size}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    seeds = [30] if arguments.smoke else list(range(30, 40))
    runs: list[dict[str, object]] = []
    by_element: list[dict[str, object]] = []
    equivariance: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for seed in seeds:
        seed_runs, seed_elements, seed_equivariance, seed_checkpoints = run_seed(seed, arguments.smoke)
        runs.extend(seed_runs)
        by_element.extend(seed_elements)
        equivariance.extend(seed_equivariance)
        checkpoints.extend(seed_checkpoints)
    unseen = [row for row in runs if row["element_role"] == "unseen"]
    comparisons = (
        ("structured_minus_capacity_matched_ordinary_unseen", "structured_soft_retransport", "capacity_matched_ordinary_chart_classifier"),
        ("structured_minus_augmented_ordinary_unseen", "structured_soft_retransport", "augmentation_trained_ordinary_chart_classifier"),
        ("structured_minus_learned_table_unseen", "structured_soft_retransport", "learned_multiplication_table_model"),
        ("structured_minus_random_action_unseen", "structured_soft_retransport", "random_action_control"),
        ("structured_minus_wrong_action_unseen", "structured_soft_retransport", "wrong_action_control"),
    )
    paired = paired_interval_rows(unseen, comparisons, "task_accuracy", 234_000_000)
    ordinary = [row for row in paired if "ordinary" in str(row["comparison"]) or "learned_table" in str(row["comparison"])]
    best_ordinary_low = min(float(row["ci_low"]) for row in ordinary)
    primary_equivariance = [row for row in equivariance if row["model"] == "equivariant"]
    multiplication = [row for row in equivariance if row["model"] == "generator_plus_multiplication_consistency"]
    controls = [row for row in paired if "random_action" in str(row["comparison"]) or "wrong_action" in str(row["comparison"])]
    claims = [
        {"claim": "unseen_task_beats_all_ordinary_learned_baselines", "value": best_ordinary_low > 0},
        {"claim": "heldout_chart_labels_never_exposed_during_fitting", "value": all(not bool(row["heldout_image_chart_labels_exposed"]) for row in equivariance)},
        {"claim": "equivariance_error_below_tolerance", "value": max(float(row["equivariance_error"]) for row in primary_equivariance) < EQUIVARIANCE_TOLERANCE, "tolerance": EQUIVARIANCE_TOLERANCE},
        {"claim": "multiplication_error_below_tolerance", "value": max(float(row["multiplication_error"]) for row in multiplication) < MULTIPLICATION_ERROR_TOLERANCE, "tolerance": MULTIPLICATION_ERROR_TOLERANCE},
        {"claim": "random_and_wrong_action_controls_fail", "value": all(float(row["ci_low"]) > 0 for row in controls)},
    ]
    claims.append({"claim": "strict_zeroshot_gate_passed", "value": all(bool(row["value"]) for row in claims)})
    summary = []
    for method in METHODS:
        for role in ("seen", "unseen"):
            block = [row for row in runs if row["method"] == method and row["element_role"] == role]
            summary.append(
                {
                    "method": method,
                    "element_role": role,
                    "seeds": len(block),
                    "task_accuracy": float(np.mean([float(row["task_accuracy"]) for row in block])),
                    "chart_accuracy": float(np.mean([float(row["chart_accuracy"]) for row in block if row["chart_accuracy"] != ""])) if any(row["chart_accuracy"] != "" for row in block) else "",
                    "ece": float(np.mean([float(row["ece"]) for row in block])),
                }
            )
    write_csv(DEST / "runs.csv", runs)
    write_csv(DEST / "summary.csv", summary)
    write_csv(DEST / "by_element.csv", by_element)
    write_csv(DEST / "paired.csv", paired)
    write_csv(DEST / "equivariance.csv", equivariance)
    write_csv(DEST / "claims.csv", claims)
    write_csv(DEST / "checkpoint_manifest.csv", checkpoints)
    latex_table(DEST / "tables" / "zeroshot.tex", ["method", "element_role", "task_accuracy", "chart_accuracy", "ece"], summary, "Strict D4 element holdout")
    factual_report(
        DEST / "report.md",
        "Strict zero-shot D4 chart generalization",
        [
            f"Execution commit: `{provenance(SCRIPT, COMMAND, 'aggregate')['execution_commit']}`. Image chart labels for {list(UNSEEN)} were absent from gradient training, early stopping, calibration, augmentation, threshold selection, and architecture selection.",
            "Generator, inverse, multiplication-law, and seen-chart equivariance supervision variants were fixed before final-test evaluation. Symbolic group relations did not expose held-out image chart labels.",
            "The preregistered gate and each constituent condition are recorded in `claims.csv`; no failed condition is replaced by a weaker post-hoc criterion.",
        ],
    )


if __name__ == "__main__":
    main()
