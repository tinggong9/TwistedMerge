#!/usr/bin/env python3
"""A1: trained Fashion-MNIST D4 chart inference and structured routing."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import (
    DATA,
    OUT,
    TMP,
    classification_metrics,
    git_head,
    latex_table,
    measure_callable,
    paired_bootstrap,
    parameter_counts,
    process_peak_mb,
    provenance,
    save_logits_before_labels,
    seed_everything,
    torch_device,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "immediate"
DEVICE = torch_device()
CONDITIONS = (
    "seen_rotations_reflections",
    "heldout_rotations",
    "heldout_reflections",
    "heldout_compositions",
    "gaussian_corruption",
    "blur",
    "chart_distribution_shift",
    "ambiguous_symmetric_inputs",
)


def apply_d4(images: torch.Tensor, charts: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(images)
    for chart in range(8):
        mask = charts == chart
        if not bool(mask.any()):
            continue
        values = images[mask]
        if chart >= 4:
            values = torch.flip(values, dims=(-1,))
        result[mask] = torch.rot90(values, chart % 4, dims=(-2, -1))
    return result


def inverse_d4(images: torch.Tensor, chart: int) -> torch.Tensor:
    values = torch.rot90(images, -(chart % 4), dims=(-2, -1))
    return torch.flip(values, dims=(-1,)) if chart >= 4 else values


class ImageCNN(nn.Module):
    def __init__(self, outputs: int, width: int = 12, return_features: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(2 * width * 49, outputs)
        self.return_features = return_features

    def forward(self, images: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images).flatten(1)
        logits = self.head(features)
        return (logits, features) if self.return_features else logits


class D4EquivariantChartCNN(nn.Module):
    """Regular-representation chart CNN built from one shared learned score CNN."""

    def __init__(self, width: int = 12):
        super().__init__()
        self.score = ImageCNN(1, width=width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # A transformed input permutes these eight learned orientation scores.
        return torch.cat([self.score(inverse_d4(images, chart)) for chart in range(8)], dim=1)


class LowRankImageContextAdapter(nn.Module):
    def __init__(self, rank: int = 4, width: int = 12):
        super().__init__()
        self.backbone = ImageCNN(10, width=width, return_features=True)
        feature_size = 2 * width * 49
        self.chart_down = nn.Linear(8, rank, bias=False)
        self.feature_down = nn.Linear(feature_size, rank, bias=False)
        self.up = nn.Linear(rank, 10, bias=False)

    def forward(self, images: torch.Tensor, chart_probabilities: torch.Tensor) -> torch.Tensor:
        logits, features = self.backbone(images)
        interaction = self.chart_down(chart_probabilities) * self.feature_down(features)
        return logits + self.up(interaction)


def split_indices(seed: int, train_size: int = 6000) -> dict[str, np.ndarray]:
    order = np.random.default_rng(91_000_000 + seed).permutation(60_000)
    return {
        "local_train": order[:train_size],
        "chart_train": order[train_size : train_size + 1000],
        "selector": order[train_size + 1000 : train_size + 1500],
        "calibration": order[train_size + 1500 : train_size + 2000],
    }


def batches(size: int, batch_size: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return list(torch.randperm(size, generator=generator).split(batch_size))


def train_task_expert(
    images: torch.Tensor, labels: torch.Tensor, seed: int, epochs: int
) -> tuple[ImageCNN, float]:
    seed_everything(seed)
    model = ImageCNN(10, width=12).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for indices in batches(len(images), 128, seed + epoch):
            x = images[indices].to(DEVICE)
            y = labels[indices].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    return model.eval(), time.perf_counter() - started


def train_chart_model(
    model: nn.Module,
    train_images: torch.Tensor,
    train_charts: torch.Tensor,
    validation_images: torch.Tensor,
    validation_charts: torch.Tensor,
    seed: int,
    epochs: int,
) -> tuple[nn.Module, float, int]:
    seed_everything(seed)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    best_loss = math.inf
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for indices in batches(len(train_images), 64, seed + epoch):
            x = train_images[indices].to(DEVICE)
            y = train_charts[indices].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.cross_entropy(model(validation_images.to(DEVICE)), validation_charts.to(DEVICE)))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), time.perf_counter() - started, epoch + 1


def task_branches(images: torch.Tensor, experts: list[ImageCNN], batch_size: int = 128) -> torch.Tensor:
    branches = []
    with torch.no_grad():
        for chart in range(8):
            canonical = inverse_d4(images, chart)
            expert_logits = []
            for expert in experts:
                parts = [expert(part.to(DEVICE)).cpu() for part in canonical.split(batch_size)]
                expert_logits.append(torch.cat(parts))
            branches.append(torch.stack(expert_logits).mean(0))
    return torch.stack(branches, dim=1)


def model_logits(model: nn.Module, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([model(part.to(DEVICE)).cpu() for part in images.split(batch_size)])


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    candidates = np.geomspace(0.4, 3.0, 31)
    losses = [float(nn.functional.cross_entropy(logits / float(value), labels)) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def choose_threshold(
    confidence: torch.Tensor, structured: torch.Tensor, fallback: torch.Tensor, labels: torch.Tensor
) -> float:
    candidates = torch.quantile(confidence, torch.linspace(0, 1, 21)).unique()
    scores = []
    for threshold in candidates:
        logits = torch.where((confidence >= threshold).unsqueeze(1), structured, fallback)
        scores.append((float((logits.argmax(1) == labels).float().mean()), -float(threshold), float(threshold)))
    return max(scores)[2]


def make_chart_examples(
    images: torch.Tensor, seed: int, role: str
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    size = len(images)
    if role == "chart_train":
        charts = rng.choice([0, 1, 4], size=size)
        conditions = np.asarray(["chart_training_seen"] * size)
    elif role in {"selector", "calibration"}:
        charts = rng.integers(0, 8, size=size)
        conditions = np.asarray([role] * size)
    else:
        per = size // len(CONDITIONS)
        chart_parts = []
        condition_parts = []
        for index, condition in enumerate(CONDITIONS):
            count = per if index < len(CONDITIONS) - 1 else size - per * (len(CONDITIONS) - 1)
            choices = {
                "seen_rotations_reflections": [0, 1, 4],
                "heldout_rotations": [2, 3],
                "heldout_reflections": [5],
                "heldout_compositions": [6, 7],
                "gaussian_corruption": list(range(8)),
                "blur": list(range(8)),
                "chart_distribution_shift": [7, 7, 7, 6],
                "ambiguous_symmetric_inputs": list(range(8)),
            }[condition]
            chart_parts.append(rng.choice(choices, size=count))
            condition_parts.extend([condition] * count)
        charts = np.concatenate(chart_parts)
        conditions = np.asarray(condition_parts)
    transformed = apply_d4(images, torch.tensor(charts, dtype=torch.long))
    if role == "test":
        gaussian = torch.tensor(conditions == "gaussian_corruption")
        transformed[gaussian] = (transformed[gaussian] + 0.25 * torch.randn_like(transformed[gaussian])).clamp(0, 1)
        blur = torch.tensor(conditions == "blur")
        transformed[blur] = nn.functional.avg_pool2d(transformed[blur], 3, stride=1, padding=1)
        ambiguous = torch.tensor(conditions == "ambiguous_symmetric_inputs")
        transformed[ambiguous] = 0.5 * (transformed[ambiguous] + torch.rot90(transformed[ambiguous], 2, (-2, -1)))
    return transformed, torch.tensor(charts, dtype=torch.long), conditions


def train_low_rank_adapter(
    images: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    validation_images: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_probabilities: torch.Tensor,
    seed: int,
    epochs: int,
) -> tuple[LowRankImageContextAdapter, float]:
    seed_everything(seed)
    model = LowRankImageContextAdapter().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    best = math.inf
    state = None
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for indices in batches(len(images), 64, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images[indices].to(DEVICE), probabilities[indices].to(DEVICE))
            loss = nn.functional.cross_entropy(logits, labels[indices].to(DEVICE))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float(nn.functional.cross_entropy(model(validation_images.to(DEVICE), validation_probabilities.to(DEVICE)), validation_labels.to(DEVICE)))
        if loss < best:
            best = loss
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if state is not None:
        model.load_state_dict(state)
    return model.eval(), time.perf_counter() - started


def run_seed(
    seed: int,
    phase: str,
    epochs: int = 8,
    task_epochs: int = 2,
    train_size: int = 6000,
    test_size: int = 2000,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    seed_everything(seed)
    training_set = FashionMNIST(DATA, train=True, download=False)
    test_set = FashionMNIST(DATA, train=False, download=False)
    all_train_images = training_set.data.float().unsqueeze(1) / 255.0
    all_train_labels = training_set.targets.long()
    all_test_images = test_set.data.float().unsqueeze(1) / 255.0
    all_test_labels = test_set.targets.long()
    split = split_indices(seed, train_size=train_size)
    local_images = all_train_images[split["local_train"]]
    local_labels = all_train_labels[split["local_train"]]
    chart_base = all_train_images[split["chart_train"]]
    chart_task_labels = all_train_labels[split["chart_train"]]
    selector_base = all_train_images[split["selector"]]
    selector_labels = all_train_labels[split["selector"]]
    calibration_base = all_train_images[split["calibration"]]
    calibration_labels = all_train_labels[split["calibration"]]
    test_order = np.random.default_rng(92_000_000 + seed).permutation(len(test_set))[:test_size]
    test_base = all_test_images[test_order]
    test_labels = all_test_labels[test_order]
    chart_images, chart_labels, _ = make_chart_examples(chart_base, 93_000_000 + seed, "chart_train")
    selector_images, selector_charts, _ = make_chart_examples(selector_base, 94_000_000 + seed, "selector")
    calibration_images, calibration_charts, _ = make_chart_examples(calibration_base, 95_000_000 + seed, "calibration")
    test_images, test_charts, test_conditions = make_chart_examples(test_base, 96_000_000 + seed, "test")
    experts = []
    expert_training_time = 0.0
    for expert_index in range(4):
        expert, elapsed = train_task_expert(local_images, local_labels, 97_000_000 + seed * 10 + expert_index, task_epochs)
        experts.append(expert); expert_training_time += elapsed
    standard, standard_time, standard_epochs = train_chart_model(
        ImageCNN(8, width=12), chart_images, chart_labels, selector_images, selector_charts, 98_000_000 + seed, epochs
    )
    matched, matched_time, matched_epochs = train_chart_model(
        ImageCNN(8, width=10), chart_images, chart_labels, selector_images, selector_charts, 98_100_000 + seed, epochs
    )
    equivariant, equivariant_time, equivariant_epochs = train_chart_model(
        D4EquivariantChartCNN(width=12), chart_images, chart_labels, selector_images, selector_charts, 98_200_000 + seed, epochs
    )
    standard_selector = model_logits(standard, selector_images)
    standard_calibration = model_logits(standard, calibration_images)
    standard_test = model_logits(standard, test_images)
    matched_test = model_logits(matched, test_images)
    equiv_selector = model_logits(equivariant, selector_images)
    equiv_calibration = model_logits(equivariant, calibration_images)
    equiv_test = model_logits(equivariant, test_images)
    standard_temperature = calibrate_temperature(standard_calibration, calibration_charts)
    equiv_temperature = calibrate_temperature(equiv_calibration, calibration_charts)
    standard_selector_prob = (standard_selector / standard_temperature).softmax(1)
    standard_test_prob = (standard_test / standard_temperature).softmax(1)
    equiv_selector_prob = (equiv_selector / equiv_temperature).softmax(1)
    equiv_test_prob = (equiv_test / equiv_temperature).softmax(1)
    selector_branches = task_branches(selector_images, experts)
    test_branches = task_branches(test_images, experts)
    selector_moe = torch.einsum("nb,nbc->nc", standard_selector_prob, selector_branches)
    test_moe = torch.einsum("nb,nbc->nc", standard_test_prob, test_branches)
    selector_structured_soft = torch.einsum("nb,nbc->nc", equiv_selector_prob, selector_branches)
    test_structured_soft = torch.einsum("nb,nbc->nc", equiv_test_prob, test_branches)
    selector_chart = equiv_selector_prob.argmax(1)
    test_chart = equiv_test_prob.argmax(1)
    selector_hard = selector_branches[torch.arange(len(selector_branches)), selector_chart]
    test_hard = test_branches[torch.arange(len(test_branches)), test_chart]
    blind = test_branches.mean(1)
    selector_blind = selector_branches.mean(1)
    confidence_selector = equiv_selector_prob.max(1).values
    confidence_test = equiv_test_prob.max(1).values
    uncertainty = confidence_test.unsqueeze(1) * test_structured_soft + (1 - confidence_test).unsqueeze(1) * blind
    threshold = choose_threshold(confidence_selector, selector_hard, selector_blind, selector_labels)
    abstaining = torch.where((confidence_test >= threshold).unsqueeze(1), test_hard, blind)
    supplied = test_branches[torch.arange(len(test_branches)), test_charts]
    random_generator = torch.Generator().manual_seed(99_000_000 + seed)
    random_action = test_branches[torch.arange(len(test_branches)), torch.randint(0, 8, (len(test_branches),), generator=random_generator)]
    wrong_action = test_branches[torch.arange(len(test_branches)), (test_chart + 1) % 8]
    adapter, adapter_time = train_low_rank_adapter(
        chart_images, chart_task_labels, (model_logits(standard, chart_images) / standard_temperature).softmax(1),
        selector_images, selector_labels, standard_selector_prob, 99_100_000 + seed, epochs,
    )
    checkpoint_dir = TMP / "checkpoints" / "chart"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": seed,
            "phase": phase,
            "split_indices": {name: values.tolist() for name, values in split.items()},
            "experts": [{name: value.detach().cpu() for name, value in model.state_dict().items()} for model in experts],
            "standard": {name: value.detach().cpu() for name, value in standard.state_dict().items()},
            "matched": {name: value.detach().cpu() for name, value in matched.state_dict().items()},
            "equivariant": {name: value.detach().cpu() for name, value in equivariant.state_dict().items()},
            "adapter": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
            "calibration_indices": split["calibration"].tolist(),
        },
        checkpoint_dir / f"{phase}_seed_{seed}.pt",
    )
    with torch.no_grad():
        adapter_test = torch.cat([
            adapter(images.to(DEVICE), probabilities.to(DEVICE)).cpu()
            for images, probabilities in zip(test_images.split(128), standard_test_prob.split(128), strict=True)
        ])
    candidates = {
        "context_blind_synchronization": blind,
        "standard_cnn_chart_classifier": test_moe,
        "parameter_matched_standard_cnn_chart_classifier": torch.einsum("nb,nbc->nc", matched_test.softmax(1), test_branches),
        "generic_mixture_of_experts": test_moe,
        "generic_low_rank_context_adapter": adapter_test,
        "trained_d4_equivariant_cnn_chart_classifier": test_structured_soft,
        "structured_group_router": test_structured_soft,
        "canonicalize_pool_retransport_inferred": test_hard,
        "uncertainty_weighted_structured_retransport": uncertainty,
        "chart_abstaining_structured_retransport": abstaining,
        "supplied_chart_oracle": supplied,
        "random_action_control": random_action,
        "wrong_action_control": wrong_action,
        "ensemble_reference": blind,
    }
    numpy_candidates = {name: values.numpy() for name, values in candidates.items()}
    ledger = save_logits_before_labels(f"chart_{phase}_{seed}", numpy_candidates, test_labels.numpy(), 99_900_000 + seed)
    chart_models = {
        "standard_cnn_chart_classifier": standard,
        "parameter_matched_standard_cnn_chart_classifier": matched,
        "trained_d4_equivariant_cnn_chart_classifier": equivariant,
    }
    chart_parameters = {name: parameter_counts(model) for name, model in chart_models.items()}
    expert_stored = sum(parameter_counts(model)[1] for model in experts)
    generic_methods = {"standard_cnn_chart_classifier", "parameter_matched_standard_cnn_chart_classifier", "generic_mixture_of_experts", "generic_low_rank_context_adapter"}
    structured_methods = {"trained_d4_equivariant_cnn_chart_classifier", "structured_group_router", "canonicalize_pool_retransport_inferred", "uncertainty_weighted_structured_retransport", "chart_abstaining_structured_retransport"}
    method_router = {
        **{name: standard for name in generic_methods if name != "parameter_matched_standard_cnn_chart_classifier"},
        "parameter_matched_standard_cnn_chart_classifier": matched,
        **{name: equivariant for name in structured_methods},
    }
    runs = []
    generalization = []
    costs = []
    chart_predictions = {
        "standard": standard_test.argmax(1),
        "matched": matched_test.argmax(1),
        "equivariant": equiv_test.argmax(1),
    }
    task_training_time = expert_training_time
    training_times = {
        "standard_cnn_chart_classifier": standard_time,
        "parameter_matched_standard_cnn_chart_classifier": matched_time,
        "generic_mixture_of_experts": standard_time,
        "generic_low_rank_context_adapter": standard_time + adapter_time,
        **{name: equivariant_time for name in structured_methods},
        "context_blind_synchronization": 0.0,
        "supplied_chart_oracle": 0.0,
        "random_action_control": 0.0,
        "wrong_action_control": equivariant_time,
        "ensemble_reference": 0.0,
    }
    for name, logits in candidates.items():
        metrics = classification_metrics(logits.numpy(), test_labels.numpy())
        if name == "parameter_matched_standard_cnn_chart_classifier":
            chart_pred = chart_predictions["matched"]
        elif name in structured_methods or name == "wrong_action_control":
            chart_pred = chart_predictions["equivariant"]
        else:
            chart_pred = chart_predictions["standard"]
        correct_chart = chart_pred == test_charts
        conditional_correct = float((logits.argmax(1)[correct_chart] == test_labels[correct_chart]).float().mean()) if bool(correct_chart.any()) else math.nan
        conditional_wrong = float((logits.argmax(1)[~correct_chart] == test_labels[~correct_chart]).float().mean()) if bool((~correct_chart).any()) else math.nan
        router = method_router.get(name)
        if router is None:
            trainable, router_stored = 0, 0
        elif name == "generic_low_rank_context_adapter":
            trainable, router_stored = parameter_counts(adapter)
        else:
            trainable, router_stored = parameter_counts(router)
        stored = router_stored + expert_stored
        runs.append(
            {
                "setting_id": f"FashionMNIST_{phase}_seed{seed}",
                "dataset": "FashionMNIST",
                "phase": phase,
                "seed": seed,
                "method": name,
                "implementation": "trained_neural" if name not in {"supplied_chart_oracle", "random_action_control", "wrong_action_control"} else "diagnostic_control",
                **metrics,
                "chart_accuracy": float((chart_pred == test_charts).float().mean()),
                "chart_cross_entropy": float(nn.functional.cross_entropy(equiv_test if name in structured_methods else standard_test, test_charts)),
                "task_accuracy_correct_chart": conditional_correct,
                "task_accuracy_incorrect_chart": conditional_wrong,
                "chart_training_examples": len(chart_images),
                "router_samples": len(chart_images),
                "selector_validation_examples": len(selector_images),
                "calibration_examples": len(calibration_images),
                "test_examples": len(test_images),
                "trainable_parameters": trainable,
                "stored_parameters": stored,
                "parameter_multiplier": stored / max(1, expert_stored),
                "branch_count": 8 if name not in {"generic_low_rank_context_adapter"} else 1,
                "candidate_count": 2 if name == "chart_abstaining_structured_retransport" else 1,
                "training_time_seconds": training_times[name] + task_training_time,
                "batch_size": 128,
                "context_mode": "supplied" if name == "supplied_chart_oracle" else ("none" if name in {"context_blind_synchronization", "ensemble_reference"} else "inferred"),
                "certificate_activated": name in structured_methods,
                "output_type": "ensemble" if name == "ensemble_reference" else ("router" if "router" in name or "mixture" in name else "branch_model"),
                "abstention_threshold": threshold if name == "chart_abstaining_structured_retransport" else "",
                "coverage": float((confidence_test >= threshold).float().mean()) if name == "chart_abstaining_structured_retransport" else 1.0,
                "logits_sha256": ledger["logits_sha256"],
                "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]),
                **provenance(SCRIPT, "python experiments/trained_chart_inference.py", seed),
            }
        )
        for condition in CONDITIONS:
            mask = test_conditions == condition
            generalization.append(
                {
                    "setting_id": f"FashionMNIST_{phase}_seed{seed}",
                    "phase": phase,
                    "seed": seed,
                    "method": name,
                    "condition": condition,
                    "examples": int(mask.sum()),
                    "task_accuracy": float((logits[mask].argmax(1) == test_labels[mask]).float().mean()),
                    "chart_accuracy": float((chart_pred[mask] == test_charts[mask]).float().mean()),
                }
            )
        # End-to-end model timing is measured for the learned router component;
        # branch retransport latency is separately present in the candidate path.
        if router is not None:
            batch = test_images[:128].to(DEVICE)
            with torch.no_grad():
                timing = measure_callable(lambda: router(batch), DEVICE, warmups=5, repeats=20)
        else:
            timing = {"latency_median_ms": 0.0, "peak_process_memory_mb": process_peak_mb(), "peak_mps_memory_mb": None}
        costs.append(
            {
                "setting_id": f"FashionMNIST_{phase}_seed{seed}",
                "phase": phase,
                "seed": seed,
                "method": name,
                "batch_size": 128,
                "latency_ms": timing["latency_median_ms"],
                "peak_process_memory_mb": timing["peak_process_memory_mb"],
                "peak_mps_memory_mb": timing["peak_mps_memory_mb"],
                "training_time_seconds": training_times[name] + task_training_time,
                "trainable_parameters": trainable,
                "stored_parameters": stored,
            }
        )
    abstention_rows = []
    for threshold_value in torch.quantile(confidence_selector, torch.linspace(0, 1, 21)).unique():
        covered = confidence_test >= threshold_value
        abstention_rows.append(
            {
                "setting_id": f"FashionMNIST_{phase}_seed{seed}",
                "phase": phase,
                "seed": seed,
                "threshold": float(threshold_value),
                "coverage": float(covered.float().mean()),
                "covered_task_accuracy": float((test_hard[covered].argmax(1) == test_labels[covered]).float().mean()) if bool(covered.any()) else math.nan,
            }
        )
    return runs, generalization, abstention_rows, costs


def gate(rows: list[dict[str, object]], generalization: list[dict[str, object]], phase: str) -> dict[str, object]:
    block = [row for row in rows if row["phase"] == phase]
    generic_names = ["standard_cnn_chart_classifier", "parameter_matched_standard_cnn_chart_classifier", "generic_mixture_of_experts", "generic_low_rank_context_adapter"]
    generic_means = {name: np.mean([float(row["accuracy"]) for row in block if row["method"] == name]) for name in generic_names}
    best_generic = max(generic_means, key=generic_means.get)
    structured_name = "chart_abstaining_structured_retransport"
    seeds = sorted({int(row["seed"]) for row in block})
    deltas = []
    worst_condition_deltas = []
    for seed in seeds:
        structured = next(float(row["accuracy"]) for row in block if row["seed"] == seed and row["method"] == structured_name)
        generic = next(float(row["accuracy"]) for row in block if row["seed"] == seed and row["method"] == best_generic)
        deltas.append(structured - generic)
        structured_conditions = [float(row["task_accuracy"]) for row in generalization if row["phase"] == phase and row["seed"] == seed and row["method"] == structured_name]
        generic_conditions = [float(row["task_accuracy"]) for row in generalization if row["phase"] == phase and row["seed"] == seed and row["method"] == best_generic]
        worst_condition_deltas.append(min(structured_conditions) - min(generic_conditions))
    mean, low, high = paired_bootstrap(deltas, seed=100_000_000 + len(seeds))
    worst_mean, worst_low, worst_high = paired_bootstrap(worst_condition_deltas, seed=100_100_000 + len(seeds))
    structured_parameters = np.mean([float(row["trainable_parameters"]) for row in block if row["method"] == structured_name])
    generic_parameters = np.mean([float(row["trainable_parameters"]) for row in block if row["method"] == best_generic])
    criterion_a = low > 0
    criterion_b = mean >= -0.002 and structured_parameters <= 0.5 * generic_parameters
    criterion_c = worst_low > 0
    return {
        "phase": phase,
        "structured_method": structured_name,
        "best_generic": best_generic,
        "mean_accuracy_delta": mean,
        "ci_low": low,
        "ci_high": high,
        "worst_condition_delta": worst_mean,
        "worst_condition_ci_low": worst_low,
        "worst_condition_ci_high": worst_high,
        "criterion_a": criterion_a,
        "criterion_b": criterion_b,
        "criterion_c": criterion_c,
        "gate_passed": criterion_a or criterion_b or criterion_c,
    }


def main() -> None:
    runs: list[dict[str, object]] = []
    generalization: list[dict[str, object]] = []
    abstention: list[dict[str, object]] = []
    costs: list[dict[str, object]] = []
    for seed in range(5):
        stage = run_seed(seed, "discovery")
        runs.extend(stage[0]); generalization.extend(stage[1]); abstention.extend(stage[2]); costs.extend(stage[3])
    paired = [gate(runs, generalization, "discovery")]
    if paired[0]["gate_passed"]:
        for seed in range(5, 15):
            stage = run_seed(seed, "confirmation")
            runs.extend(stage[0]); generalization.extend(stage[1]); abstention.extend(stage[2]); costs.extend(stage[3])
        paired.append(gate(runs, generalization, "confirmation"))
    summary = []
    for (phase, method) in sorted({(str(row["phase"]), str(row["method"])) for row in runs}):
        block = [row for row in runs if row["phase"] == phase and row["method"] == method]
        summary.append({"phase": phase, "method": method, "runs": len(block), "accuracy": float(np.mean([float(row["accuracy"]) for row in block])), "chart_accuracy": float(np.mean([float(row["chart_accuracy"]) for row in block])), "ece": float(np.mean([float(row["ece"]) for row in block]))})
    claims = [
        {"claim": "discovery_passed", "value": paired[0]["gate_passed"]},
        {"claim": "confirmation_executed", "value": len(paired) > 1},
        {"claim": "confirmation_passed", "value": len(paired) > 1 and paired[1]["gate_passed"]},
        {"claim": "cifar_triggered", "value": False},
        {"claim": "all_chart_rows_trained_neural_or_controls", "value": all(row["implementation"] in {"trained_neural", "diagnostic_control"} for row in runs)},
    ]
    write_csv(DEST / "chart_runs.csv", runs)
    write_csv(DEST / "chart_summary.csv", summary)
    write_csv(DEST / "chart_paired.csv", paired)
    write_csv(DEST / "chart_generalization.csv", generalization)
    write_csv(DEST / "chart_abstention.csv", abstention)
    write_csv(DEST / "chart_cost.csv", costs)
    write_csv(DEST / "chart_claims.csv", claims)
    latex_table(DEST / "tables" / "chart_main.tex", ["phase", "method", "runs", "accuracy", "chart_accuracy"], summary, "Trained Fashion-MNIST chart inference")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    frame = [row for row in abstention if row["phase"] == "discovery"]
    for seed in sorted({int(row["seed"]) for row in frame}):
        block = [row for row in frame if row["seed"] == seed]
        axis.plot([row["coverage"] for row in block], [row["covered_task_accuracy"] for row in block], alpha=0.65, label=f"seed {seed}")
    axis.set(xlabel="Coverage", ylabel="Covered task accuracy")
    axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(DEST / "plots" / "chart_coverage.pdf"); plt.close(figure)
    confirmation = len(paired) > 1 and bool(paired[1]["gate_passed"])
    (DEST / "chart_report.md").write_text(
        "# Trained Fashion-MNIST chart inference\n\n"
        f"Execution commit: `{git_head()}`. Five discovery collections used four independently trained task experts, "
        "three trained neural chart classifiers including a regular-representation D4-equivariant CNN, a trained low-rank "
        "adapter, disjoint chart/selector/calibration/test roles, and eight evaluation conditions. "
        f"The discovery gate {'passed' if paired[0]['gate_passed'] else 'did not pass'}. Confirmation "
        f"{'passed' if confirmation else ('did not pass' if len(paired) > 1 else 'was not triggered')}; CIFAR-10 "
        "was not triggered in this bounded program.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
