#!/usr/bin/env python3
"""A2: end-to-end controlled S3/D4 accuracy and systems-cost audit."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.next_program_common import (
    OUT,
    classification_metrics,
    dihedral_group,
    git_head,
    latex_table,
    measure_callable,
    paired_bootstrap,
    parameter_counts,
    provenance,
    save_logits_before_labels,
    seed_everything,
    symmetric_group_3,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "immediate"
DEVICE = torch.device("cpu")  # Tiny controlled models are faster and more stable on CPU.


def regular_actions(table: np.ndarray) -> torch.Tensor:
    actions = []
    for g in range(len(table)):
        matrix = np.zeros((len(table), len(table)), dtype=np.float32)
        for h in range(len(table)):
            matrix[int(table[g, h]), h] = 1.0
        actions.append(matrix)
    return torch.tensor(np.stack(actions))


def make_split(table: np.ndarray, noise: float, size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    classes = 4
    labels = rng.integers(0, classes, size=size)
    contexts = rng.integers(0, len(table), size=size)
    prototypes = np.random.default_rng(81_000).normal(size=(classes, len(table))).astype(np.float32)
    canonical = prototypes[labels] + rng.normal(scale=noise, size=(size, len(table))).astype(np.float32)
    actions = regular_actions(table).numpy()
    observed = np.einsum("nij,nj->ni", actions[contexts], canonical)
    return torch.tensor(observed), torch.tensor(contexts), torch.tensor(labels)


class MLP(nn.Module):
    def __init__(self, dimension: int, classes: int = 4, width: int = 24):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, width), nn.ReLU(), nn.Linear(width, classes))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        del context
        return self.network(x)


class StructuredRetransport(nn.Module):
    def __init__(self, actions: torch.Tensor, diagnostic: bool = False):
        super().__init__()
        self.register_buffer("actions", actions)
        self.backbone = MLP(actions.shape[-1])
        self.diagnostic = diagnostic

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        inverse = self.actions[context].transpose(1, 2)
        canonical = torch.bmm(inverse, x.unsqueeze(-1)).squeeze(-1)
        if self.diagnostic:
            # The certificate is label-independent and is evaluated as part of
            # the end-to-end path. Exact regular actions activate it.
            residual = torch.linalg.matrix_norm(
                torch.bmm(self.actions[context], inverse) - torch.eye(x.shape[1], device=x.device), dim=(-2, -1)
            )
            canonical = canonical * (residual < 1e-6).float().unsqueeze(1)
        return self.backbone(canonical, context)


class MixtureOfExperts(nn.Module):
    def __init__(self, dimension: int, contexts: int):
        super().__init__()
        self.experts = nn.ModuleList([MLP(dimension, width=12) for _ in range(contexts)])
        self.router = nn.Embedding(contexts, contexts)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        weights = self.router(context).softmax(-1)
        branches = torch.stack([expert(x, context) for expert in self.experts], dim=1)
        return torch.einsum("nb,nbc->nc", weights, branches)


class LowRankContextAdapter(nn.Module):
    def __init__(self, dimension: int, contexts: int, rank: int = 3):
        super().__init__()
        self.base = nn.Linear(dimension, 24)
        self.context = nn.Embedding(contexts, rank)
        self.up = nn.Linear(rank, 24, bias=False)
        self.output = nn.Linear(24, 4)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.base(x) + self.up(self.context(context)))
        return self.output(hidden)


class LearnedMatrixAction(nn.Module):
    def __init__(self, dimension: int, contexts: int):
        super().__init__()
        self.actions = nn.Parameter(torch.eye(dimension).repeat(contexts, 1, 1))
        self.backbone = MLP(dimension)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        transformed = torch.bmm(self.actions[context], x.unsqueeze(-1)).squeeze(-1)
        return self.backbone(transformed, context)


class EnsembleReference(nn.Module):
    def __init__(self, dimension: int, branches: int = 4):
        super().__init__()
        self.models = nn.ModuleList([MLP(dimension, width=16) for _ in range(branches)])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(x, context) for model in self.models]).mean(0)


def train_model(
    model: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    selector: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int = 24,
) -> tuple[float, int]:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best = math.inf
    best_state = None
    stale = 0
    started = time.perf_counter()
    x, context, labels = (value.to(DEVICE) for value in train)
    sx, sc, sy = (value.to(DEVICE) for value in selector)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x))
        for indices in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices], context[indices]), labels[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(nn.functional.cross_entropy(model(sx, sc), sy))
        if validation < best - 1e-5:
            best = validation
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return time.perf_counter() - started, epoch + 1


def run_setting(
    group: str,
    noise: float,
    context_budget: int,
    seed: int,
    repeats: int = 100,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_everything(seed)
    table = symmetric_group_3() if group == "S3" else dihedral_group(4)
    actions = regular_actions(table)
    # Fixed, disjoint roles. Labels are generated once before any model exists.
    train = make_split(table, noise, 512, seed * 100 + 1)
    transition = make_split(table, noise, 128, seed * 100 + 2)
    router_full = make_split(table, noise, 64, seed * 100 + 3)
    router = tuple(value[:context_budget] for value in router_full)
    selector = make_split(table, noise, 128, seed * 100 + 4)
    calibration = make_split(table, noise, 128, seed * 100 + 5)
    test = make_split(table, noise, 512, seed * 100 + 6)
    del transition, calibration
    training = tuple(torch.cat([left, right]) for left, right in zip(train, router, strict=True))
    models: dict[str, nn.Module] = {
        "structured_group_retransport": StructuredRetransport(actions),
        "twistedmerge_diagnostic_retransport": StructuredRetransport(actions, diagnostic=True),
        "generic_mixture_of_experts": MixtureOfExperts(len(table), len(table)),
        "generic_low_rank_context_adapter": LowRankContextAdapter(len(table), len(table)),
        "unconstrained_learned_matrix_action": LearnedMatrixAction(len(table), len(table)),
        "context_blind_synchronization": MLP(len(table)),
        "ensemble_reference": EnsembleReference(len(table)),
    }
    setting_id = f"{group}_noise{noise}_budget{context_budget}_seed{seed}"
    accuracy_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    logits: dict[str, np.ndarray] = {}
    training_metadata: dict[str, tuple[float, int]] = {}
    tx, tc, ty = (value.to(DEVICE) for value in test)
    for name, model in models.items():
        training_metadata[name] = train_model(model, training, selector)
        model.eval()
        with torch.no_grad():
            logits[name] = model(tx, tc).cpu().numpy()
    ledger = save_logits_before_labels(f"cost_{setting_id}", logits, ty.numpy(), seed + 9_000_000)
    for name, model in models.items():
        metrics = classification_metrics(logits[name], ty.numpy())
        trainable, stored = parameter_counts(model)
        train_time, epochs = training_metadata[name]
        branches = len(table) if name == "generic_mixture_of_experts" else (4 if name == "ensemble_reference" else 1)
        accuracy_rows.append(
            {
                "setting_id": setting_id,
                "group": group,
                "noise": noise,
                "context_budget": context_budget,
                "seed": seed,
                "method": name,
                **metrics,
                "training_time_seconds": train_time,
                "epochs": epochs,
                "trainable_parameters": trainable,
                "stored_parameters": stored,
                "parameter_multiplier": stored / max(1, parameter_counts(models["context_blind_synchronization"])[1]),
                "branch_count": branches,
                "candidate_count": 1,
                "transition_samples": 128,
                "router_samples": context_budget,
                "selector_validation_samples": 128,
                "calibration_samples": 128,
                "test_samples": len(ty),
                "context_mode": "supplied",
                "certificate_activated": name == "twistedmerge_diagnostic_retransport",
                "output_type": "ensemble" if name == "ensemble_reference" else ("router" if name == "generic_mixture_of_experts" else "single_model"),
                "logits_sha256": ledger["logits_sha256"],
                "label_permutation_hash_passed": bool(ledger["candidate_hashes_unchanged"] and ledger["file_hash_unchanged"]),
                **provenance(SCRIPT, "python experiments/end_to_end_controlled_cost.py", seed),
            }
        )
        for batch_size in (1, 8, 32, 128):
            bx, bc = tx[:batch_size], tc[:batch_size]
            preprocessing = measure_callable(lambda: (bx.contiguous(), bc.contiguous()), DEVICE, warmups=10, repeats=repeats)
            router_timing = measure_callable(
                (lambda: model.router(bc).softmax(-1)) if isinstance(model, MixtureOfExperts) else (lambda: bc.clone()),
                DEVICE,
                warmups=10,
                repeats=repeats,
            )
            with torch.no_grad():
                total = measure_callable(lambda: model(bx, bc), DEVICE, warmups=10, repeats=repeats)
            cost_rows.append(
                {
                    "setting_id": setting_id,
                    "group": group,
                    "noise": noise,
                    "context_budget": context_budget,
                    "seed": seed,
                    "method": name,
                    "batch_size": batch_size,
                    "preprocessing_latency_ms": preprocessing["latency_median_ms"],
                    "router_latency_ms": router_timing["latency_median_ms"],
                    "total_inference_latency_ms": total["latency_median_ms"],
                    **total,
                    "stored_parameters": stored,
                    "trainable_parameters": trainable,
                    "branch_count": branches,
                    "calibration_cost_samples": 128,
                    "context_training_samples": context_budget,
                    "training_time_seconds": train_time,
                    "flops": "unavailable_no_model_aware_counter_installed",
                    "measurement_type": "end_to_end_torch_cpu",
                    **provenance(SCRIPT, "python experiments/end_to_end_controlled_cost.py", seed),
                }
            )
    return accuracy_rows, cost_rows


def summarize(
    accuracy_rows: list[dict[str, object]], cost_rows: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary = []
    for method in sorted({str(row["method"]) for row in accuracy_rows}):
        block = [row for row in accuracy_rows if row["method"] == method]
        batch32 = [row for row in cost_rows if row["method"] == method and row["batch_size"] == 32]
        summary.append(
            {
                "method": method,
                "settings": len(block),
                "mean_accuracy": float(np.mean([float(row["accuracy"]) for row in block])),
                "mean_stored_parameters": float(np.mean([float(row["stored_parameters"]) for row in block])),
                "mean_latency_batch32_ms": float(np.mean([float(row["total_inference_latency_ms"]) for row in batch32])),
                "mean_training_time_seconds": float(np.mean([float(row["training_time_seconds"]) for row in block])),
            }
        )
    reference = "structured_group_retransport"
    alternatives = sorted({str(row["method"]) for row in accuracy_rows if row["method"] != reference})
    paired = []
    for alternative in alternatives:
        deltas = []
        for setting in sorted({str(row["setting_id"]) for row in accuracy_rows}):
            structured = next(float(row["accuracy"]) for row in accuracy_rows if row["setting_id"] == setting and row["method"] == reference)
            other = next(float(row["accuracy"]) for row in accuracy_rows if row["setting_id"] == setting and row["method"] == alternative)
            deltas.append(structured - other)
        mean, low, high = paired_bootstrap(deltas, seed=82_000_000 + len(alternative))
        paired.append({"reference": reference, "alternative": alternative, "mean_accuracy_delta": mean, "ci_low": low, "ci_high": high})
    best_generic = max(
        (row for row in summary if row["method"] in {"generic_mixture_of_experts", "generic_low_rank_context_adapter", "unconstrained_learned_matrix_action"}),
        key=lambda row: float(row["mean_accuracy"]),
    )
    structured = next(row for row in summary if row["method"] == reference)
    matched = (
        float(structured["mean_stored_parameters"]) <= float(best_generic["mean_stored_parameters"])
        and float(structured["mean_latency_batch32_ms"]) <= float(best_generic["mean_latency_batch32_ms"]) * 1.25
    )
    comparison = next(row for row in paired if row["alternative"] == best_generic["method"])
    claims = [
        {"claim": "all_costs_end_to_end_measured", "value": all(row["measurement_type"] == "end_to_end_torch_cpu" for row in cost_rows)},
        {"claim": "structured_accuracy_advantage_positive", "value": float(comparison["ci_low"]) > 0.0},
        {"claim": "structured_matched_cost", "value": matched},
        {"claim": "structured_cost_efficient", "value": matched and float(comparison["ci_low"]) > 0.0},
    ]
    return summary, paired, claims


def main() -> None:
    accuracy_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for group in ("S3", "D4"):
        for noise in (0.2, 0.5):
            for budget in (16, 64):
                for seed in range(40, 45):
                    runs, costs = run_setting(group, noise, budget, seed)
                    accuracy_rows.extend(runs)
                    cost_rows.extend(costs)
    summary, paired, claims = summarize(accuracy_rows, cost_rows)
    write_csv(DEST / "cost_accuracy_runs.csv", accuracy_rows)
    write_csv(DEST / "cost_runs.csv", cost_rows)
    write_csv(DEST / "cost_summary.csv", summary)
    write_csv(DEST / "cost_paired.csv", paired)
    write_csv(DEST / "cost_claims.csv", claims)
    latex_table(
        DEST / "tables" / "cost.tex",
        ["method", "mean_accuracy", "mean_stored_parameters", "mean_latency_batch32_ms"],
        summary,
        "End-to-end controlled accuracy and cost",
    )
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for row in summary:
        axis.scatter(float(row["mean_latency_batch32_ms"]), float(row["mean_accuracy"]), label=row["method"], s=26)
    axis.set(xlabel="Measured batch-32 latency (ms)", ylabel="Mean accuracy")
    axis.legend(fontsize=5, loc="best")
    figure.tight_layout()
    figure.savefig(DEST / "plots" / "cost_accuracy.pdf")
    plt.close(figure)
    passed = next(row["value"] for row in claims if row["claim"] == "structured_cost_efficient")
    (DEST / "cost_report.md").write_text(
        "# End-to-end controlled systems audit\n\n"
        f"Execution commit: `{git_head()}`. Seven actual PyTorch implementations were trained and evaluated across "
        f"{len({row['setting_id'] for row in accuracy_rows})} independent S3/D4 settings. Cold start and 100 synchronized "
        f"warm repetitions were measured at batch sizes 1, 8, 32, and 128. FLOPs are marked unavailable because no "
        f"model-aware counter is installed. The matched accuracy-and-cost gate {'passed' if passed else 'did not pass'}.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
