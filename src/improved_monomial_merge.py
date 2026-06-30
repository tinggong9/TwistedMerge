"""Improved exact-ReLU monomial scaling and validation selectors.

The helpers in this module keep the operational distinction sharp:
positive hidden-unit scaling is an exact reparameterization for each ReLU MLP
before averaging, while validation chooses among already-built single models.
No selector here reads test metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .ladder_merge_methods import transform_mlp_positive_scale
from .model_merging_benchmark import (
    DatasetSpec,
    average_models,
    clone_model,
    evaluate_model,
    make_model,
    require_torch,
)


@dataclass(frozen=True)
class GlobalScaleSyncResult:
    log_scales: np.ndarray
    rms_residual: float
    max_residual: float
    gauge_fix: str


@dataclass(frozen=True)
class ScaleDiagnostics:
    mean_abs_log_scale: float
    max_abs_log_scale: float
    log_scale_variance: float
    synchronization_disagreement: float


@dataclass(frozen=True)
class ValidationChoice:
    selected: str
    selected_val_accuracy: float
    selected_val_loss: float
    margin_to_runner_up: float
    used_test_metrics: bool = False


@dataclass(frozen=True)
class SoupResult:
    model: object
    selected_indices: list[int]
    selected_labels: list[str]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None


def clip_log_scales(log_scales: np.ndarray, tau: float) -> np.ndarray:
    logs = np.asarray(log_scales, dtype=float)
    if tau is None or not isfinite(float(tau)):
        return logs.copy()
    return np.clip(logs, -float(tau), float(tau))


def shrink_log_scales(log_scales: np.ndarray, alpha: float, tau: float = float("inf")) -> np.ndarray:
    """Return ``alpha * clip(log_scales, -tau, tau)``.

    ``alpha=0`` produces all-zero log scales, i.e. ordinary C2M3 after
    permutation alignment. ``alpha=1, tau=inf`` preserves the raw monomial
    scales.
    """

    return float(alpha) * clip_log_scales(log_scales, tau)


def scales_from_logs(log_scales: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(log_scales, dtype=float))


def aligned_feature_columns(
    features: Mapping[int, np.ndarray],
    synced_permutations: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    return {
        int(idx): np.asarray(features[int(idx)], dtype=float)[:, np.asarray(perm, dtype=int)]
        for idx, perm in synced_permutations.items()
    }


def _positive_scale_from_columns(source: np.ndarray, target: np.ndarray, min_scale: float, max_scale: float) -> float:
    denom = max(float(np.dot(source, source)), 1e-12)
    scale = float(np.dot(source, target) / denom)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float(np.clip(scale, min_scale, max_scale))


def reference_log_scales_from_features(
    features: Mapping[int, np.ndarray],
    synced_permutations: Mapping[int, np.ndarray],
    *,
    ref: int,
    width: int,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
) -> np.ndarray:
    """Estimate reference-based log scales in reference hidden-unit order."""

    aligned = aligned_feature_columns(features, synced_permutations)
    logs = np.zeros((len(aligned), width), dtype=float)
    reference = aligned[int(ref)]
    for idx in sorted(aligned):
        if idx == int(ref):
            continue
        other = aligned[idx]
        for unit in range(width):
            logs[idx, unit] = np.log(
                _positive_scale_from_columns(reference[:, unit], other[:, unit], min_scale, max_scale)
            )
    return logs


def global_log_scale_synchronization(
    features: Mapping[int, np.ndarray],
    synced_permutations: Mapping[int, np.ndarray],
    *,
    n_models: int,
    width: int,
    ref: int = 0,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
) -> GlobalScaleSyncResult:
    """Least-squares synchronize pairwise positive log-scale gauges.

    In reference coordinates, pairwise estimates satisfy
    ``ell_j[k] - ell_i[k] ~= log a_ij[k]``.  The gauge is fixed by
    ``ell_ref = 0``.
    """

    aligned = aligned_feature_columns(features, synced_permutations)
    logs = np.zeros((n_models, width), dtype=float)
    all_residuals = []
    for unit in range(width):
        matrix_rows = []
        rhs = []
        for i, j in product(range(n_models), repeat=2):
            if i == j:
                continue
            scale = _positive_scale_from_columns(
                aligned[i][:, unit],
                aligned[j][:, unit],
                min_scale,
                max_scale,
            )
            row = np.zeros(n_models, dtype=float)
            row[j] = 1.0
            row[i] = -1.0
            matrix_rows.append(row)
            rhs.append(np.log(scale))
        gauge = np.zeros(n_models, dtype=float)
        gauge[int(ref)] = max(float(n_models), 1.0)
        matrix_rows.append(gauge)
        rhs.append(0.0)
        A = np.vstack(matrix_rows)
        b = np.asarray(rhs, dtype=float)
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        solution = solution - solution[int(ref)]
        logs[:, unit] = solution
        all_residuals.extend((A[:-1] @ solution - b[:-1]).tolist())
    residuals = np.asarray(all_residuals, dtype=float)
    return GlobalScaleSyncResult(
        log_scales=logs,
        rms_residual=float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0,
        max_residual=float(np.max(np.abs(residuals))) if residuals.size else 0.0,
        gauge_fix=f"ell_{int(ref)}=0",
    )


def log_scale_diagnostics(log_scales: np.ndarray, synchronization_disagreement: float = float("nan")) -> ScaleDiagnostics:
    logs = np.asarray(log_scales, dtype=float)
    return ScaleDiagnostics(
        mean_abs_log_scale=float(np.mean(np.abs(logs))),
        max_abs_log_scale=float(np.max(np.abs(logs))) if logs.size else 0.0,
        log_scale_variance=float(np.var(logs)),
        synchronization_disagreement=float(synchronization_disagreement),
    )


def build_scaled_models(
    models: Sequence,
    spec: DatasetSpec,
    width: int,
    synced_permutations: Mapping[int, np.ndarray],
    log_scales: np.ndarray,
) -> list:
    logs = np.asarray(log_scales, dtype=float)
    if logs.shape != (len(models), width):
        raise ValueError("log_scales must have shape (n_models, width)")
    return [
        transform_mlp_positive_scale(
            model,
            spec,
            width,
            np.asarray(synced_permutations[idx], dtype=int),
            scales_from_logs(logs[idx]),
        )
        for idx, model in enumerate(models)
    ]


def build_scaled_average_model(
    models: Sequence,
    spec: DatasetSpec,
    width: int,
    synced_permutations: Mapping[int, np.ndarray],
    log_scales: np.ndarray,
):
    scaled = build_scaled_models(models, spec, width, synced_permutations, log_scales)
    return average_models(scaled, "mlp", spec, width)


def choose_by_validation(
    val_metrics_by_name: Mapping[str, Mapping[str, float]],
    *,
    allowed_methods: Sequence[str] | None = None,
) -> ValidationChoice:
    """Choose by validation accuracy, with validation loss as a tie-break."""

    names = list(allowed_methods) if allowed_methods is not None else list(val_metrics_by_name)
    if not names:
        raise ValueError("at least one candidate method is required")
    ordered = sorted(
        names,
        key=lambda name: (
            float(val_metrics_by_name[name]["accuracy"]),
            -float(val_metrics_by_name[name]["loss"]),
            name,
        ),
        reverse=True,
    )
    selected = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else selected
    return ValidationChoice(
        selected=selected,
        selected_val_accuracy=float(val_metrics_by_name[selected]["accuracy"]),
        selected_val_loss=float(val_metrics_by_name[selected]["loss"]),
        margin_to_runner_up=float(
            val_metrics_by_name[selected]["accuracy"] - val_metrics_by_name[runner_up]["accuracy"]
        ),
        used_test_metrics=False,
    )


def margin_selector(
    baseline: str,
    challenger: str,
    val_metrics_by_name: Mapping[str, Mapping[str, float]],
    *,
    epsilon: float,
) -> ValidationChoice:
    delta = float(val_metrics_by_name[challenger]["accuracy"] - val_metrics_by_name[baseline]["accuracy"])
    if delta >= float(epsilon):
        selected = challenger
    else:
        selected = baseline
    other = baseline if selected == challenger else challenger
    return ValidationChoice(
        selected=selected,
        selected_val_accuracy=float(val_metrics_by_name[selected]["accuracy"]),
        selected_val_loss=float(val_metrics_by_name[selected]["loss"]),
        margin_to_runner_up=float(val_metrics_by_name[selected]["accuracy"] - val_metrics_by_name[other]["accuracy"]),
        used_test_metrics=False,
    )


def optimize_log_scales_for_validation(
    models: Sequence,
    spec: DatasetSpec,
    width: int,
    synced_permutations: Mapping[int, np.ndarray],
    initial_log_scales: np.ndarray,
    val_loader,
    device,
    *,
    ref: int = 0,
    steps: int = 50,
    lr: float = 0.05,
    bound: float = 1.0,
    l2: float = 1e-3,
) -> np.ndarray:
    """Optimize log scales of the averaged MLP on validation loss only."""

    torch, _, F = require_torch()
    n_models = len(models)
    init = np.asarray(initial_log_scales, dtype=float)
    if init.shape != (n_models, width):
        raise ValueError("initial_log_scales must have shape (n_models, width)")
    trainable_indices = [idx for idx in range(n_models) if idx != int(ref)]
    aligned_hidden_weights = []
    aligned_hidden_biases = []
    aligned_classifier_weights = []
    classifier_biases = []
    for idx, model in enumerate(models):
        perm = np.asarray(synced_permutations[idx], dtype=int)
        aligned_hidden_weights.append(model.hidden.weight.detach().cpu()[perm, :].to(device))
        aligned_hidden_biases.append(model.hidden.bias.detach().cpu()[perm].to(device))
        aligned_classifier_weights.append(model.classifier.weight.detach().cpu()[:, perm].to(device))
        classifier_biases.append(model.classifier.bias.detach().cpu().to(device))

    if not trainable_indices or steps <= 0:
        return np.clip(init, -float(bound), float(bound))

    start = np.clip(init[trainable_indices], -float(bound), float(bound))
    theta = torch.nn.Parameter(torch.tensor(start, dtype=aligned_hidden_weights[0].dtype, device=device))
    opt = torch.optim.Adam([theta], lr=float(lr))
    ref_row = torch.zeros((1, width), dtype=theta.dtype, device=device)

    for _step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        full_theta_rows = []
        cursor = 0
        for idx in range(n_models):
            if idx == int(ref):
                full_theta_rows.append(ref_row[0])
            else:
                full_theta_rows.append(theta[cursor])
                cursor += 1
        full_theta = torch.stack(full_theta_rows, dim=0).clamp(-float(bound), float(bound))
        scales = torch.exp(full_theta)
        hidden_weight = torch.stack(
            [aligned_hidden_weights[idx] / scales[idx].unsqueeze(1) for idx in range(n_models)],
            dim=0,
        ).mean(dim=0)
        hidden_bias = torch.stack(
            [aligned_hidden_biases[idx] / scales[idx] for idx in range(n_models)],
            dim=0,
        ).mean(dim=0)
        classifier_weight = torch.stack(
            [aligned_classifier_weights[idx] * scales[idx].unsqueeze(0) for idx in range(n_models)],
            dim=0,
        ).mean(dim=0)
        classifier_bias = torch.stack(classifier_biases, dim=0).mean(dim=0)
        total_loss = torch.zeros((), dtype=theta.dtype, device=device)
        total = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            flat = x.view(x.shape[0], -1)
            hidden = F.relu(flat @ hidden_weight.T + hidden_bias)
            logits = hidden @ classifier_weight.T + classifier_bias
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss = total_loss + loss
            total += int(y.numel())
        objective = total_loss / max(total, 1) + float(l2) * torch.mean(full_theta**2)
        objective.backward()
        opt.step()

    out = init.copy()
    out[trainable_indices] = theta.detach().cpu().numpy()
    out[int(ref)] = 0.0
    return np.clip(out, -float(bound), float(bound))


def greedy_soup_with_metadata(
    candidate_models: Sequence,
    candidate_labels: Sequence[str],
    val_loader,
    test_loader,
    device,
    architecture: str,
    spec: DatasetSpec,
    width: int,
    *,
    evaluate_test: bool = True,
) -> SoupResult:
    if len(candidate_models) != len(candidate_labels):
        raise ValueError("candidate_models and candidate_labels must have the same length")
    if not candidate_models:
        raise ValueError("at least one soup candidate is required")
    scored = []
    for idx, model in enumerate(candidate_models):
        metrics = evaluate_model(model, val_loader, device)
        scored.append((float(metrics["accuracy"]), -float(metrics["loss"]), idx))
    order = [idx for _acc, _neg_loss, idx in sorted(scored, reverse=True)]
    selected = [order[0]]
    soup = clone_model(candidate_models[order[0]], architecture, spec, width)
    best_val = evaluate_model(soup, val_loader, device)
    for idx in order[1:]:
        candidate_indices = selected + [idx]
        candidate = average_models([candidate_models[item] for item in candidate_indices], architecture, spec, width)
        candidate_val = evaluate_model(candidate, val_loader, device)
        better = (
            candidate_val["accuracy"] > best_val["accuracy"]
            or (
                candidate_val["accuracy"] == best_val["accuracy"]
                and candidate_val["loss"] <= best_val["loss"]
            )
        )
        if better:
            soup = candidate
            selected = candidate_indices
            best_val = candidate_val
    return SoupResult(
        model=soup,
        selected_indices=selected,
        selected_labels=[candidate_labels[idx] for idx in selected],
        val_metrics=best_val,
        test_metrics=evaluate_model(soup, test_loader, device) if evaluate_test else None,
    )


def count_parameters(model) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def assert_capacity_matched_mlp(model, spec: DatasetSpec, width: int) -> None:
    reference = make_model("mlp", spec, width)
    if count_parameters(model) != count_parameters(reference):
        raise AssertionError("model is not capacity-matched to the reference MLP")
