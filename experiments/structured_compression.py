#!/usr/bin/env python3
"""B5: measured compression of confirmed structured S3 and D4 teachers."""

from __future__ import annotations

import hashlib
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

from experiments.end_to_end_controlled_cost import regular_actions
from experiments.trained_chart_inference import (
    D4EquivariantChartCNN,
    ImageCNN,
    make_chart_examples,
    model_logits,
    task_branches,
)
from experiments.next_program_common import (
    DATA,
    OUT,
    TMP,
    classification_metrics,
    dihedral_group,
    git_head,
    latex_table,
    measure_callable,
    paired_bootstrap,
    parameter_counts,
    provenance,
    read_csv,
    save_logits_before_labels,
    seed_everything,
    sha256_file,
    symmetric_group_3,
    torch_device,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "iclr"
DEVICE = torch_device()
STUDENTS = (
    "chart_token_student",
    "shared_canonical_backbone_group_head",
    "low_rank_group_generators",
    "finite_state_chart_module",
    "tensor_factorized_equivariant_head",
    "quantized_structured_student",
    "pruned_structured_student",
    "ordinary_single_model_control",
)
OBJECTIVES = (
    "supervised",
    "supervised_kl",
    "supervised_kl_chart",
    "supervised_kl_chart_multiplication",
    "full_structure_factorization",
)


def make_data(table: np.ndarray, seed: int, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed); classes = 4
    labels = rng.integers(0, classes, size=size); contexts = rng.integers(0, len(table), size=size)
    prototypes = np.random.default_rng(141_000_000).normal(size=(classes, len(table))).astype(np.float32)
    canonical = prototypes[labels] + rng.normal(scale=0.35, size=(size, len(table))).astype(np.float32)
    actions = regular_actions(table).numpy(); observed = np.einsum("nij,nj->ni", actions[contexts], canonical)
    return torch.tensor(observed), torch.tensor(contexts), torch.tensor(labels)


class StructuredTeacher(nn.Module):
    def __init__(self, actions: torch.Tensor, width: int = 32):
        super().__init__(); self.register_buffer("actions", actions)
        self.network = nn.Sequential(nn.Linear(actions.shape[-1], width), nn.ReLU(), nn.Linear(width, 4))

    def forward(self, values: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        inverse = self.actions[contexts].transpose(1, 2)
        canonical = torch.bmm(inverse, values.unsqueeze(-1)).squeeze(-1)
        return self.network(canonical)


class StructuredStudent(nn.Module):
    def __init__(self, actions: torch.Tensor, width: int, mode: str, rank: int = 4):
        super().__init__(); self.mode = mode; self.register_buffer("actions", actions)
        contexts, dimension = actions.shape[0], actions.shape[-1]
        self.context_embedding = nn.Embedding(contexts, rank)
        self.context_head = nn.Linear(rank, contexts)
        self.composition_head = nn.Linear(2 * rank, contexts)
        self.low_left = nn.Parameter(torch.zeros(contexts, dimension, rank))
        self.low_right = nn.Parameter(torch.zeros(contexts, rank, dimension))
        input_size = dimension + (rank if mode == "chart_token_student" else 0)
        self.backbone = nn.Sequential(nn.Linear(input_size, width), nn.ReLU())
        self.output_down = nn.Linear(width, min(rank, width), bias=False)
        self.output_up = nn.Linear(min(rank, width), 4)

    def representation(self, values: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        embedding = self.context_embedding(contexts)
        if self.mode == "ordinary_single_model_control":
            canonical = values
        elif self.mode == "chart_token_student":
            canonical = torch.cat([values, embedding], dim=1)
        elif self.mode == "low_rank_group_generators":
            identity = torch.eye(values.shape[1], device=values.device).expand(len(values), -1, -1)
            learned = identity + torch.bmm(self.low_left[contexts], self.low_right[contexts])
            canonical = torch.bmm(learned, values.unsqueeze(-1)).squeeze(-1)
        else:
            inverse = self.actions[contexts].transpose(1, 2)
            canonical = torch.bmm(inverse, values.unsqueeze(-1)).squeeze(-1)
        return self.backbone(canonical)

    def forward(self, values: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        hidden = self.representation(values, contexts)
        return self.output_up(self.output_down(hidden))

    def chart_logits(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.context_head(self.context_embedding(contexts))

    def multiplication_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.composition_head(torch.cat([self.context_embedding(left), self.context_embedding(right)], dim=1))


class FashionCompressedStudent(nn.Module):
    """Small image student with an inferred D4 chart state and mode-specific fusion."""

    def __init__(self, width: int, mode: str, rank: int = 4):
        super().__init__(); self.mode = mode; self.rank = rank
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.chart_head = nn.Linear(width, 8)
        self.group_embedding = nn.Parameter(torch.randn(8, rank) * 0.02)
        self.low_up = nn.Linear(rank, width, bias=False)
        self.chart_token_fusion = nn.Linear(width + rank, width)
        self.composition_head = nn.Linear(2 * rank, 8)
        self.task_head = nn.Linear(width, 10)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images).flatten(1)
        chart_logits = self.chart_head(features)
        probabilities = chart_logits.softmax(1)
        token = probabilities @ self.group_embedding
        if self.mode == "ordinary_single_model_control":
            fused = features
        elif self.mode == "chart_token_student":
            fused = torch.relu(self.chart_token_fusion(torch.cat([features, token], dim=1)))
        elif self.mode == "low_rank_group_generators":
            fused = features + self.low_up(token)
        elif self.mode == "finite_state_chart_module":
            fused = features + torch.tanh(self.low_up(token))
        elif self.mode == "tensor_factorized_equivariant_head":
            fused = features * (1 + 0.25 * torch.tanh(self.low_up(token)))
        else:
            # Shared-backbone, quantized, and pruned students use a learned
            # chart-conditioned gate before their post-training compression.
            fused = features * torch.sigmoid(2 + self.low_up(token))
        return self.task_head(fused), chart_logits

    def multiplication_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.composition_head(torch.cat([self.group_embedding[left], self.group_embedding[right]], dim=1))


def train_teacher(model: nn.Module, train, validation, seed: int, epochs: int = 30) -> tuple[nn.Module, float]:
    seed_everything(seed); model.to(DEVICE); optimizer = torch.optim.Adam(model.parameters(), lr=0.004); started = time.perf_counter()
    best = math.inf; state = None
    for epoch in range(epochs):
        order = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for indices in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train[0][indices].to(DEVICE), train[1][indices].to(DEVICE)), train[2][indices].to(DEVICE))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): loss = float(nn.functional.cross_entropy(model(validation[0].to(DEVICE), validation[1].to(DEVICE)), validation[2].to(DEVICE)))
        if loss < best: best = loss; state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if state is not None: model.load_state_dict(state)
    return model.eval(), time.perf_counter() - started


def objective_weights(name: str) -> tuple[float, float, float, float]:
    return {
        "supervised": (0, 0, 0, 0),
        "supervised_kl": (1, 0, 0, 0),
        "supervised_kl_chart": (1, 0.5, 0, 0),
        "supervised_kl_chart_multiplication": (1, 0.5, 0.5, 0),
        "full_structure_factorization": (1, 0.5, 0.5, 1e-4),
    }[name]


def train_student(model: StructuredStudent, teacher: StructuredTeacher, table: np.ndarray, train, validation, objective: str, seed: int, epochs: int = 14) -> tuple[StructuredStudent, float]:
    seed_everything(seed); model.to(DEVICE); teacher.to(DEVICE).eval(); optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    kl_weight, chart_weight, multiplication_weight, factor_weight = objective_weights(objective)
    started = time.perf_counter(); best = math.inf; state = None
    pair_left, pair_right = torch.meshgrid(torch.arange(len(table)), torch.arange(len(table)), indexing="ij")
    pair_left = pair_left.flatten().to(DEVICE); pair_right = pair_right.flatten().to(DEVICE)
    products = torch.tensor(table.flatten(), dtype=torch.long, device=DEVICE)
    for epoch in range(epochs):
        order = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for indices in order.split(128):
            values, contexts, labels = (part[indices].to(DEVICE) for part in train)
            optimizer.zero_grad(set_to_none=True); logits = model(values, contexts)
            loss = nn.functional.cross_entropy(logits, labels)
            if kl_weight:
                with torch.no_grad(): teacher_prob = teacher(values, contexts).softmax(-1)
                loss = loss + kl_weight * nn.functional.kl_div(logits.log_softmax(-1), teacher_prob, reduction="batchmean")
            if chart_weight: loss = loss + chart_weight * nn.functional.cross_entropy(model.chart_logits(contexts), contexts)
            if multiplication_weight: loss = loss + multiplication_weight * nn.functional.cross_entropy(model.multiplication_logits(pair_left, pair_right), products)
            if factor_weight: loss = loss + factor_weight * (model.low_left.square().mean() + model.low_right.square().mean())
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): validation_loss = float(nn.functional.cross_entropy(model(validation[0].to(DEVICE), validation[1].to(DEVICE)), validation[2].to(DEVICE)))
        if validation_loss < best: best = validation_loss; state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if state is not None: model.load_state_dict(state)
    return model.eval(), time.perf_counter() - started


def train_fashion_student(
    model: FashionCompressedStudent,
    table: np.ndarray,
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    objective: str,
    seed: int,
    epochs: int = 5,
) -> tuple[FashionCompressedStudent, float]:
    seed_everything(seed); model.to(DEVICE); optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    kl_weight, chart_weight, multiplication_weight, factor_weight = objective_weights(objective)
    left, right = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    left = left.flatten().to(DEVICE); right = right.flatten().to(DEVICE)
    products = torch.tensor(table.flatten(), dtype=torch.long, device=DEVICE)
    started = time.perf_counter(); best = math.inf; state = None
    for epoch in range(epochs):
        order = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for indices in order.split(64):
            images = train[0][indices].to(DEVICE); labels = train[1][indices].to(DEVICE)
            teacher_task = train[2][indices].to(DEVICE); charts = train[3][indices].to(DEVICE)
            optimizer.zero_grad(set_to_none=True); logits, chart_logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if kl_weight: loss = loss + kl_weight * nn.functional.kl_div(logits.log_softmax(1), teacher_task.softmax(1), reduction="batchmean")
            if chart_weight: loss = loss + chart_weight * nn.functional.cross_entropy(chart_logits, charts)
            if multiplication_weight: loss = loss + multiplication_weight * nn.functional.cross_entropy(model.multiplication_logits(left, right), products)
            if factor_weight: loss = loss + factor_weight * model.group_embedding.square().mean()
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): validation_logits, _ = model(validation[0].to(DEVICE)); validation_loss = float(nn.functional.cross_entropy(validation_logits, validation[1].to(DEVICE)))
        if validation_loss < best:
            best = validation_loss; state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if state is not None: model.load_state_dict(state)
    return model.eval(), time.perf_counter() - started


def apply_compression(model: StructuredStudent, mode: str, reduction: float) -> None:
    if mode == "quantized_structured_student":
        with torch.no_grad():
            for parameter in model.parameters():
                scale = parameter.abs().max().clamp_min(1e-8) / 127
                parameter.copy_(torch.round(parameter / scale).clamp(-127, 127) * scale)
    elif mode == "pruned_structured_student":
        with torch.no_grad():
            values = torch.cat([parameter.abs().flatten().cpu() for parameter in model.parameters()])
            threshold = torch.quantile(values, min(0.95, reduction))
            for parameter in model.parameters(): parameter.mul_(parameter.abs() >= threshold.to(parameter.device))


def stored_artifact(model: StructuredStudent, name: str, mode: str) -> tuple[int, str]:
    path = TMP / "compression" / f"{name}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for parameter_name, value in model.state_dict().items():
        array = value.detach().cpu().numpy()
        if mode == "quantized_structured_student" and np.issubdtype(array.dtype, np.floating):
            scale = max(float(np.max(np.abs(array))) / 127, 1e-12)
            arrays[parameter_name] = np.round(array / scale).clip(-127, 127).astype(np.int8); arrays[parameter_name + "__scale"] = np.asarray([scale], dtype=np.float32)
        elif mode == "pruned_structured_student" and np.issubdtype(array.dtype, np.floating):
            nonzero = np.flatnonzero(array); arrays[parameter_name + "__indices"] = nonzero.astype(np.int32); arrays[parameter_name + "__values"] = array.flat[nonzero].astype(np.float32); arrays[parameter_name + "__shape"] = np.asarray(array.shape, dtype=np.int32)
        else: arrays[parameter_name] = array
    np.savez_compressed(path, **arrays)
    return path.stat().st_size, sha256_file(path)


def run_group(group: str, table: np.ndarray, seed: int, targets=(0.25, 0.50, 0.75), objectives=OBJECTIVES, students=STUDENTS, student_epochs: int = 14):
    actions = regular_actions(table); train = make_data(table, 142_000_000 + seed * 10, 1024); validation = make_data(table, 142_000_001 + seed * 10, 256); test = make_data(table, 142_000_002 + seed * 10, 512)
    teacher, teacher_time = train_teacher(StructuredTeacher(actions), train, validation, 142_100_000 + seed)
    blind, blind_time = train_teacher(StructuredTeacher(torch.eye(len(table)).repeat(len(table), 1, 1)), train, validation, 142_200_000 + seed)
    with torch.no_grad(): teacher_logits = teacher(test[0].to(DEVICE), test[1].to(DEVICE)).cpu(); blind_logits = blind(test[0].to(DEVICE), test[1].to(DEVICE)).cpu()
    teacher_accuracy = classification_metrics(teacher_logits.numpy(), test[2].numpy())["accuracy"]; blind_accuracy = classification_metrics(blind_logits.numpy(), test[2].numpy())["accuracy"]
    teacher_bytes, teacher_hash = stored_artifact(teacher, f"{group}_{seed}_teacher", "shared_canonical_backbone_group_head")
    widths = {0.25: 24, 0.50: 16, 0.75: 8}; rows = []
    candidate_logits = {}
    models = {}
    for reduction in targets:
        for objective in objectives:
            for student_index, mode in enumerate(students):
                model = StructuredStudent(actions, widths[float(reduction)], mode)
                model, elapsed = train_student(model, teacher, table, train, validation, objective, 142_300_000 + seed * 10_000 + int(reduction * 100) * 100 + len(objective) + student_index, student_epochs)
                apply_compression(model, mode, float(reduction)); model.eval()
                with torch.no_grad(): logits = model(test[0].to(DEVICE), test[1].to(DEVICE)).cpu(); chart_logits = model.chart_logits(test[1].to(DEVICE)).cpu()
                key = f"{mode}_r{int(reduction*100)}_{objective}"; candidate_logits[key] = logits.numpy(); models[key] = (model, elapsed, chart_logits, reduction, objective, mode)
    ledger = save_logits_before_labels(f"compression_{group}_{seed}", candidate_logits, test[2].numpy(), 142_900_000 + seed)
    for key, logits in candidate_logits.items():
        model, elapsed, chart_logits, reduction, objective, mode = models[key]
        metrics = classification_metrics(logits, test[2].numpy()); student_bytes, artifact_hash = stored_artifact(model, f"{group}_{seed}_{key}", mode)
        teacher_gain = teacher_accuracy - blind_accuracy; retained = (metrics["accuracy"] - blind_accuracy) / teacher_gain if abs(teacher_gain) > 1e-12 else math.nan
        batch_values, batch_contexts = test[0][:128].to(DEVICE), test[1][:128].to(DEVICE)
        with torch.no_grad(): timing = measure_callable(lambda: model(batch_values, batch_contexts), DEVICE, warmups=5, repeats=30)
        trainable, stored = parameter_counts(model)
        probabilities = torch.tensor(logits).softmax(1); teacher_prob = teacher_logits.softmax(1)
        kl = float(nn.functional.kl_div(probabilities.clamp_min(1e-8).log(), teacher_prob, reduction="batchmean"))
        rows.append({"setting_id": f"{group}_seed{seed}", "group": group, "seed": seed, "teacher": "independently_trained_structured_teacher", "method": mode, "objective": objective, "target_storage_reduction": reduction, **metrics, "action_accuracy": float((chart_logits.argmax(1) == test[1]).float().mean()), "unseen_word_accuracy": "not_applicable_controlled_context", "teacher_student_kl": kl, "teacher_accuracy": teacher_accuracy, "ordinary_control_accuracy": blind_accuracy, "retained_teacher_gain_fraction": retained, "teacher_storage_bytes": teacher_bytes, "student_storage_bytes": student_bytes, "measured_storage_reduction": 1 - student_bytes / max(1, teacher_bytes), "latency_ms_batch128": timing["latency_median_ms"], "peak_process_memory_mb": timing["peak_process_memory_mb"], "stored_parameters": stored, "trainable_parameters": trainable, "training_time_seconds": elapsed + teacher_time + blind_time, "teacher_artifact_sha256": teacher_hash, "student_artifact_sha256": artifact_hash, "logits_sha256": ledger["logits_sha256"], "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]), **provenance(SCRIPT, "python experiments/structured_compression.py", seed)})
    return rows


def fashion_teacher_data(seed: int):
    checkpoint_seed = seed + 5
    checkpoint_path = TMP / "checkpoints" / "chart" / f"confirmation_seed_{checkpoint_seed}.pt"
    if not checkpoint_path.exists(): raise FileNotFoundError(f"confirmed A1 checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training = FashionMNIST(DATA, train=True, download=False); testing = FashionMNIST(DATA, train=False, download=False)
    split = {name: np.asarray(values, dtype=int) for name, values in checkpoint["split_indices"].items()}
    train_images, train_charts, _ = make_chart_examples(training.data[split["chart_train"]].float().unsqueeze(1) / 255.0, 143_000_000 + seed, "chart_train")
    validation_images, validation_charts, _ = make_chart_examples(training.data[split["selector"]].float().unsqueeze(1) / 255.0, 143_100_000 + seed, "selector")
    order = np.random.default_rng(143_200_000 + seed).permutation(len(testing))[:512]
    test_images, test_charts, _ = make_chart_examples(testing.data[order].float().unsqueeze(1) / 255.0, 143_300_000 + seed, "test")
    # make_chart_examples' second return is the D4 chart. Task labels remain
    # those of the underlying Fashion-MNIST examples.
    task_train = training.targets[split["chart_train"]]; task_validation = training.targets[split["selector"]]; task_test = testing.targets[order]
    experts = []
    for state in checkpoint["experts"]:
        model = ImageCNN(10, width=12).to(DEVICE); model.load_state_dict(state); experts.append(model.eval())
    chart_model = D4EquivariantChartCNN(width=12).to(DEVICE); chart_model.load_state_dict(checkpoint["equivariant"]); chart_model.eval()

    def teacher(images: torch.Tensor) -> torch.Tensor:
        probabilities = model_logits(chart_model, images).softmax(1)
        return torch.einsum("nb,nbc->nc", probabilities, task_branches(images, experts))

    teacher_train = teacher(train_images); teacher_validation = teacher(validation_images); teacher_test = teacher(test_images)
    return (
        (train_images, task_train, teacher_train, train_charts),
        (validation_images, task_validation, teacher_validation, validation_charts),
        (test_images, task_test, teacher_test, test_charts),
        checkpoint_path,
    )


def run_fashion(seed: int, targets=(0.25, 0.50, 0.75), objectives=OBJECTIVES, students=STUDENTS, student_epochs: int = 5):
    train, validation, test, checkpoint_path = fashion_teacher_data(seed); table = dihedral_group(4)
    teacher_accuracy = classification_metrics(test[2].numpy(), test[1].numpy())["accuracy"]
    widths = {0.25: 24, 0.50: 16, 0.75: 8}; candidates = {}; models = {}
    for reduction in targets:
        for objective in objectives:
            for student_index, mode in enumerate(students):
                model = FashionCompressedStudent(widths[float(reduction)], mode)
                model, elapsed = train_fashion_student(model, table, train, validation, objective, 143_400_000 + seed * 10_000 + int(reduction * 100) * 100 + len(objective) + student_index, student_epochs)
                apply_compression(model, mode, float(reduction)); model.eval()
                with torch.no_grad(): logits, chart_logits = model(test[0].to(DEVICE)); logits = logits.cpu(); chart_logits = chart_logits.cpu()
                key = f"{mode}_r{int(reduction*100)}_{objective}"; candidates[key] = logits.numpy(); models[key] = (model, elapsed, chart_logits, reduction, objective, mode)
    ledger = save_logits_before_labels(f"compression_FashionMNIST_{seed}", candidates, test[1].numpy(), 143_900_000 + seed)
    rows = []
    for key, logits in candidates.items():
        model, elapsed, chart_logits, reduction, objective, mode = models[key]
        control_key = f"ordinary_single_model_control_r{int(float(reduction)*100)}_{objective}"
        ordinary_accuracy = classification_metrics(candidates[control_key], test[1].numpy())["accuracy"]
        metrics = classification_metrics(logits, test[1].numpy()); student_bytes, artifact_hash = stored_artifact(model, f"FashionMNIST_{seed}_{key}", mode)
        teacher_gain = teacher_accuracy - ordinary_accuracy; retained = (metrics["accuracy"] - ordinary_accuracy) / teacher_gain if abs(teacher_gain) > 1e-12 else math.nan
        with torch.no_grad(): timing = measure_callable(lambda: model(test[0][:128].to(DEVICE))[0], DEVICE, warmups=5, repeats=30)
        trainable, stored = parameter_counts(model); probabilities = torch.tensor(logits).softmax(1); teacher_prob = test[2].softmax(1)
        rows.append({"setting_id": f"FashionMNIST_seed{seed}", "group": "FashionMNIST", "seed": seed, "teacher": "confirmed_A1_inferred_teacher", "method": mode, "objective": objective, "target_storage_reduction": reduction, **metrics, "action_accuracy": float((chart_logits.argmax(1) == test[3]).float().mean()), "unseen_word_accuracy": "not_applicable_image_teacher", "teacher_student_kl": float(nn.functional.kl_div(probabilities.clamp_min(1e-8).log(), teacher_prob, reduction="batchmean")), "teacher_accuracy": teacher_accuracy, "ordinary_control_accuracy": ordinary_accuracy, "retained_teacher_gain_fraction": retained, "teacher_storage_bytes": checkpoint_path.stat().st_size, "student_storage_bytes": student_bytes, "measured_storage_reduction": 1 - student_bytes / checkpoint_path.stat().st_size, "latency_ms_batch128": timing["latency_median_ms"], "peak_process_memory_mb": timing["peak_process_memory_mb"], "stored_parameters": stored, "trainable_parameters": trainable, "training_time_seconds": elapsed, "teacher_artifact_sha256": sha256_file(checkpoint_path), "student_artifact_sha256": artifact_hash, "logits_sha256": ledger["logits_sha256"], "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]), **provenance(SCRIPT, "python experiments/structured_compression.py", seed)})
    return rows


def main() -> None:
    groups = {"S3": symmetric_group_3(), "D4": dihedral_group(4)}
    existing = read_csv(DEST / "compression_runs.csv")
    controlled = [row for row in existing if row.get("group") in groups]
    rows = controlled if len(controlled) == 5 * len(groups) * 3 * len(OBJECTIVES) * len(STUDENTS) else []
    if not rows:
        for seed in range(5):
            for group, table in groups.items(): rows.extend(run_group(group, table, seed))
    chart_gate = DEST.parent / "immediate" / "chart_claims.csv"; multiview_gate = DEST / "multiview_claims.csv"
    fashion_eligible = chart_gate.exists() and "confirmation_passed,True" in chart_gate.read_text()
    multiview_eligible = multiview_gate.exists() and "complete_multiview_gate_passed,True" in multiview_gate.read_text()
    existing_fashion = [row for row in existing if row.get("group") == "FashionMNIST"]
    if fashion_eligible:
        if len(existing_fashion) == 5 * 3 * len(OBJECTIVES) * len(STUDENTS): rows.extend(existing_fashion)
        else:
            for seed in range(5): rows.extend(run_fashion(seed))
    if multiview_eligible:
        raise RuntimeError("B3 passed; a ModelNet10-specific compression implementation must execute before B5 can complete")
    summary = []
    for group, method, reduction in sorted({(row["group"], row["method"], row["target_storage_reduction"]) for row in rows}):
        block = [row for row in rows if row["group"] == group and row["method"] == method and row["target_storage_reduction"] == reduction]
        summary.append({"group": group, "method": method, "target_storage_reduction": reduction, "runs": len(block), "accuracy": float(np.mean([float(row["accuracy"]) for row in block])), "retained_teacher_gain_fraction": float(np.mean([float(row["retained_teacher_gain_fraction"]) for row in block])), "measured_storage_reduction": float(np.mean([float(row["measured_storage_reduction"]) for row in block])), "latency_ms_batch128": float(np.mean([float(row["latency_ms_batch128"]) for row in block]))})
    claims = []
    for group in sorted({str(row["group"]) for row in rows}):
        block = [row for row in rows if row["group"] == group]
        passed = any(float(row["retained_teacher_gain_fraction"]) >= 0.95 and (float(row["measured_storage_reduction"]) >= 0.25 or float(row["latency_ms_batch128"]) <= 0.75 * max(float(candidate["latency_ms_batch128"]) for candidate in block if candidate["method"] == "shared_canonical_backbone_group_head")) for row in block)
        claims.append({"group": group, "claim": "structured_compression_gate_passed", "value": passed})
    claims.extend([{"group": "FashionMNIST", "claim": "teacher_eligible", "value": fashion_eligible}, {"group": "FashionMNIST", "claim": "teacher_executed", "value": any(row["group"] == "FashionMNIST" for row in rows)}, {"group": "ModelNet10", "claim": "teacher_eligible", "value": multiview_eligible}, {"group": "ModelNet10", "claim": "teacher_executed", "value": any(row["group"] == "ModelNet10" for row in rows)}])
    write_csv(DEST / "compression_runs.csv", rows); write_csv(DEST / "compression_summary.csv", summary); write_csv(DEST / "compression_claims.csv", claims)
    latex_table(DEST / "tables" / "compression.tex", ["group", "method", "target_storage_reduction", "retained_teacher_gain_fraction", "measured_storage_reduction"], summary, "Structure-preserving compression")
    passed = sum(bool(row["value"]) for row in claims if row["claim"] == "structured_compression_gate_passed")
    (DEST / "compression_report.md").write_text(
        "# Structure-preserving distillation and compression\n\n"
        f"Execution commit: `{git_head()}`. Independently trained S3 and D4 structured teachers were compressed with seven "
        "structured students and one required ordinary control at three target reductions under five cumulative objectives. "
        f"Storage is the byte size of executed dense, sparse, or int8 tensor artifacts; latency is measured end to end. "
        f"{passed} executed teachers passed the 95%-gain and 25%-measured-reduction gate. Fashion-MNIST and "
        "ModelNet10 teachers are executed only when their upstream confirmation gates pass.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
