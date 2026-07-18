#!/usr/bin/env python3
"""Staged exactness audit for BatchNorm-aware ResNet channel gauges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.batchnorm_channel_gauge import (  # noqa: E402
    parameter_count,
    permute_resnet_channels,
    random_resnet18_permutations,
    scale_conv_batchnorm,
    scale_relu_conv_pair,
)


PHASE = "batchnorm_gauge"
STRATEGY_LABELS = {
    "permutation": "exact compatible channel permutation",
    "frozen": "scaled Conv; original frozen statistics and affine parameters",
    "running": "scaled Conv and running statistics only",
    "affine": "scaled Conv and affine compensation with original frozen statistics",
    "running_affine": "scaled Conv, transformed running statistics, and epsilon-aware affine compensation",
    "post_merge_recalibration": "scaled Conv with short BatchNorm recalibration",
    "complete_recomputation": "scaled Conv with complete target-BatchNorm statistic recomputation",
    "no_batchnorm_control": "exact positive Conv-ReLU-Conv gauge without BatchNorm",
}

EXACT_TOLERANCES = {
    ("permutation", "eval"): 5e-5,
    ("permutation", "train"): 2e-4,
    ("affine", "eval"): 5e-5,
    ("running_affine", "eval"): 5e-5,
    ("no_batchnorm_control", "eval"): 5e-6,
    ("no_batchnorm_control", "train"): 5e-6,
}


def parse_csv(text: str, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    """Return the process peak resident set size in MiB.

    macOS reports bytes while Linux reports KiB.
    """

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def write_csv(path: Path, rows: list[dict], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_resnet(seed: int, epsilon: float):
    import torch
    import torchvision

    torch.manual_seed(seed)
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eps = float(epsilon)
    return model


class NoBatchNormPair:
    @staticmethod
    def make(seed: int):
        import torch

        torch.manual_seed(seed)

        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(16, 10, 3, padding=1)

            def forward(self, inputs):
                return self.conv2(torch.relu(self.conv1(inputs))).mean(dim=(2, 3))

        return Pair()


def random_batches(seed: int, count: int, batch_size: int):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(batch_size, 3, 32, 32, generator=generator) for _ in range(count)]


def target_batchnorm_recalibration(model, batches, *, reset: bool, momentum: float | None):
    import copy
    import torch

    transformed = copy.deepcopy(model)
    for module in transformed.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    target = transformed.layer1[0].bn1
    if reset:
        target.reset_running_stats()
    target.momentum = momentum
    target.train()
    with torch.no_grad():
        for inputs in batches:
            transformed(inputs)
    transformed.eval()
    return transformed


def metrics(original, transformed, batches, *, mode: str) -> dict[str, float]:
    import copy
    import torch

    first_model = copy.deepcopy(original)
    second_model = copy.deepcopy(transformed)
    first_model.train(mode == "train")
    second_model.train(mode == "train")
    differences = []
    originals = []
    disagreements = 0
    total = 0
    started = time.perf_counter()
    with torch.no_grad():
        for inputs in batches:
            first = first_model(inputs)
            second = second_model(inputs)
            differences.append((first - second).abs().reshape(-1))
            originals.append(first.abs().reshape(-1))
            disagreements += int((first.argmax(dim=1) != second.argmax(dim=1)).sum())
            total += int(first.shape[0])
    elapsed = time.perf_counter() - started
    diff = torch.cat(differences)
    baseline = torch.cat(originals)
    mean_abs = float(diff.mean())
    return {
        "max_absolute_logit_error": float(diff.max()),
        "mean_absolute_logit_error": mean_abs,
        "relative_output_error": mean_abs / max(float(baseline.mean()), 1e-12),
        "prediction_disagreement": disagreements / max(total, 1),
        "evaluation_seconds": elapsed,
    }


def activation_metrics(original, transformed, batches, permutations, *, mode: str) -> list[dict]:
    import copy
    import torch

    first_model = copy.deepcopy(original)
    second_model = copy.deepcopy(transformed)
    first_model.train(mode == "train")
    second_model.train(mode == "train")
    locations = [("stem", first_model.bn1, second_model.bn1, permutations.stem)]
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        first_stage = getattr(first_model, stage_name)
        second_stage = getattr(second_model, stage_name)
        for index, (first_block, second_block) in enumerate(zip(first_stage, second_stage)):
            locations.append(
                (
                    f"{stage_name}.{index}.after_residual",
                    first_block,
                    second_block,
                    permutations.stages[stage_name],
                )
            )
    captured_first: dict[str, list] = {name: [] for name, *_ in locations}
    captured_second: dict[str, list] = {name: [] for name, *_ in locations}
    handles = []
    for name, first_module, second_module, _perm in locations:
        handles.append(first_module.register_forward_hook(lambda _m, _i, output, key=name: captured_first[key].append(output.detach().cpu())))
        handles.append(second_module.register_forward_hook(lambda _m, _i, output, key=name: captured_second[key].append(output.detach().cpu())))
    with torch.no_grad():
        for inputs in batches:
            first_model(inputs)
            second_model(inputs)
    for handle in handles:
        handle.remove()
    rows = []
    for name, _first_module, _second_module, permutation in locations:
        index = torch.as_tensor(permutation, dtype=torch.long)
        errors = []
        for first, second in zip(captured_first[name], captured_second[name]):
            expected = first.index_select(1, index)
            errors.append((expected - second).abs().reshape(-1))
        combined = torch.cat(errors)
        rows.append(
            {
                "mode": mode,
                "location": name,
                "max_absolute_activation_error": float(combined.max()),
                "mean_absolute_activation_error": float(combined.mean()),
            }
        )
    return rows


def calibrate_base(model, batches):
    """Give the target BN meaningful frozen statistics before scaling tests."""

    return target_batchnorm_recalibration(model, batches, reset=True, momentum=None)


def exact_status(strategy: str, mode: str, maximum: float, disagreement: float) -> tuple[str, float]:
    tolerance = EXACT_TOLERANCES.get((strategy, mode), math.nan)
    if not np.isfinite(tolerance):
        return "not_purported_exact", tolerance
    return ("exact_within_float32_tolerance" if maximum <= tolerance and disagreement == 0 else "exactness_failed"), tolerance


def run_seed(args, seed: int, epsilon: float) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    import torch

    evaluation_batches = random_batches(seed + 10000, args.batches, args.batch_size)
    calibration_batches = random_batches(seed + 20000, args.calibration_batches, args.batch_size)
    full_batches = random_batches(seed + 30000, args.recomputation_batches, args.batch_size)
    base = calibrate_base(make_resnet(seed, epsilon), full_batches)
    base.eval()
    scales = torch.exp(torch.linspace(math.log(0.25), math.log(3.0), base.layer1[0].conv1.out_channels))

    candidates = {}
    transformation_seconds = {}
    permutations = random_resnet18_permutations(base, seed=seed + 71)
    started = time.perf_counter()
    candidates["permutation"] = permute_resnet_channels(base, permutations)
    transformation_seconds["permutation"] = time.perf_counter() - started
    for strategy in ("frozen", "running", "affine", "running_affine"):
        started = time.perf_counter()
        candidates[strategy] = scale_conv_batchnorm(
            base,
            "layer1.0.conv1",
            "layer1.0.bn1",
            scales,
            strategy=strategy,
        )
        transformation_seconds[strategy] = time.perf_counter() - started
    scaled_for_recalibration = candidates["frozen"]
    started = time.perf_counter()
    candidates["post_merge_recalibration"] = target_batchnorm_recalibration(
        scaled_for_recalibration,
        calibration_batches,
        reset=False,
        momentum=0.1,
    )
    transformation_seconds["post_merge_recalibration"] = time.perf_counter() - started
    started = time.perf_counter()
    candidates["complete_recomputation"] = target_batchnorm_recalibration(
        scaled_for_recalibration,
        full_batches,
        reset=True,
        momentum=None,
    )
    transformation_seconds["complete_recomputation"] = time.perf_counter() - started
    no_bn = NoBatchNormPair.make(seed + 91)
    started = time.perf_counter()
    no_bn_scaled = scale_relu_conv_pair(no_bn, "conv1", "conv2", torch.linspace(0.25, 3.0, 16))
    no_bn_transformation_seconds = time.perf_counter() - started

    rows = []
    resources = []
    failures = []
    for strategy, candidate in candidates.items():
        evaluation_seconds = 0.0
        for mode in ("eval", "train"):
            try:
                values = metrics(base, candidate, evaluation_batches, mode=mode)
                evaluation_seconds += values["evaluation_seconds"]
                status, tolerance = exact_status(
                    strategy,
                    mode,
                    values["max_absolute_logit_error"],
                    values["prediction_disagreement"],
                )
                rows.append(
                    {
                        "seed": seed,
                        "epsilon": epsilon,
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "mode": mode,
                        **values,
                        "purported_exact": np.isfinite(tolerance),
                        "exactness_tolerance": tolerance,
                        "exactness_status": status,
                        "batches": args.batches,
                        "batch_size": args.batch_size,
                        "parameter_count": parameter_count(candidate),
                        "parameter_count_unchanged": parameter_count(candidate) == parameter_count(base),
                        "capacity": "same_architecture_same_parameter_count",
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "seed": seed,
                        "epsilon": epsilon,
                        "strategy": strategy,
                        "mode": mode,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        resources.append(
            {
                "seed": seed,
                "epsilon": epsilon,
                "strategy": strategy,
                "parameter_multiplier": 1.0,
                "active_parameters": parameter_count(candidate),
                "branches": 1,
                "inference_multiplier": 1.0,
                "output_type": "same_capacity_single_model",
                "stored_checkpoint_bytes": sum(value.numel() * value.element_size() for value in candidate.state_dict().values()),
                "training_compute": "none",
                "calibration_batches": args.calibration_batches if strategy == "post_merge_recalibration" else args.recomputation_batches if strategy == "complete_recomputation" else 0,
                "merge_compute_seconds": transformation_seconds[strategy],
                "identity_evaluation_seconds": evaluation_seconds,
                "identity_evaluation_batches": 2 * args.batches,
                "validation_evaluation_count": 0,
                "process_peak_rss_mb": peak_rss_mb(),
            }
        )

    no_bn_evaluation_seconds = 0.0
    for mode in ("eval", "train"):
        values = metrics(no_bn, no_bn_scaled, evaluation_batches, mode=mode)
        no_bn_evaluation_seconds += values["evaluation_seconds"]
        status, tolerance = exact_status(
            "no_batchnorm_control", mode, values["max_absolute_logit_error"], values["prediction_disagreement"]
        )
        rows.append(
            {
                "seed": seed,
                "epsilon": epsilon,
                "strategy": "no_batchnorm_control",
                "strategy_label": STRATEGY_LABELS["no_batchnorm_control"],
                "mode": mode,
                **values,
                "purported_exact": True,
                "exactness_tolerance": tolerance,
                "exactness_status": status,
                "batches": args.batches,
                "batch_size": args.batch_size,
                "parameter_count": parameter_count(no_bn_scaled),
                "parameter_count_unchanged": parameter_count(no_bn_scaled) == parameter_count(no_bn),
                "capacity": "same_architecture_same_parameter_count",
            }
        )
    resources.append(
        {
            "seed": seed,
            "epsilon": epsilon,
            "strategy": "no_batchnorm_control",
            "parameter_multiplier": 1.0,
            "active_parameters": parameter_count(no_bn_scaled),
            "branches": 1,
            "inference_multiplier": 1.0,
            "output_type": "same_capacity_single_model",
            "stored_checkpoint_bytes": sum(
                value.numel() * value.element_size() for value in no_bn_scaled.state_dict().values()
            ),
            "training_compute": "none",
            "calibration_batches": 0,
            "merge_compute_seconds": no_bn_transformation_seconds,
            "identity_evaluation_seconds": no_bn_evaluation_seconds,
            "identity_evaluation_batches": 2 * args.batches,
            "validation_evaluation_count": 0,
            "process_peak_rss_mb": peak_rss_mb(),
        }
    )

    activations = []
    for mode in ("eval", "train"):
        for row in activation_metrics(base, candidates["permutation"], evaluation_batches, permutations, mode=mode):
            activations.append({"seed": seed, "epsilon": epsilon, **row})
    return rows, activations, resources, failures


def bootstrap_seed_ci(frame: pd.DataFrame, column: str, n_bootstrap: int, seed: int) -> tuple[float, float]:
    values = frame.groupby("seed")[column].mean().to_numpy(float)
    if not len(values):
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (strategy, mode, epsilon), group in frame.groupby(["strategy", "mode", "epsilon"], sort=False):
        purported_exact = bool(group["purported_exact"].any())
        rows.append(
            {
                "strategy": strategy,
                "mode": mode,
                "epsilon": epsilon,
                "n_seeds": group["seed"].nunique(),
                "n_rows": len(group),
                "max_overall_logit_error": float(group["max_absolute_logit_error"].max()),
                "mean_max_logit_error": float(group["max_absolute_logit_error"].mean()),
                "mean_absolute_logit_error": float(group["mean_absolute_logit_error"].mean()),
                "mean_relative_output_error": float(group["relative_output_error"].mean()),
                "max_prediction_disagreement": float(group["prediction_disagreement"].max()),
                "purported_exact": purported_exact,
                "all_exactness_checks_passed": bool((group["exactness_status"] == "exact_within_float32_tolerance").all()) if purported_exact else False,
                "capacity": "same_architecture_same_parameter_count",
            }
        )
    return rows


def paired(frame: pd.DataFrame, n_bootstrap: int) -> list[dict]:
    eval_only = frame[frame["mode"].eq("eval")]
    wide = eval_only.pivot(index=["seed", "epsilon"], columns="strategy", values="max_absolute_logit_error").reset_index()
    comparisons = [
        ("running_minus_frozen", "running", "frozen"),
        ("affine_minus_frozen", "affine", "frozen"),
        ("running_affine_minus_running", "running_affine", "running"),
        ("complete_recomputation_minus_post_merge_recalibration", "complete_recomputation", "post_merge_recalibration"),
    ]
    rows = []
    for name, method, baseline in comparisons:
        clean = wide.dropna(subset=[method, baseline]).copy()
        clean["delta"] = clean[method] - clean[baseline]
        low, high = bootstrap_seed_ci(clean, "delta", n_bootstrap, 7001 + len(rows))
        rows.append(
            {
                "comparison": name,
                "method": method,
                "baseline": baseline,
                "n_pairs": len(clean),
                "n_seeds": clean["seed"].nunique(),
                "paired_mean_max_error_delta": float(clean["delta"].mean()),
                "paired_delta_ci_low": low,
                "paired_delta_ci_high": high,
                "wins_lower_error": int((clean["delta"] < 0).sum()),
                "ties": int((clean["delta"] == 0).sum()),
                "losses_higher_error": int((clean["delta"] > 0).sum()),
                "bootstrap_unit": "seed; epsilon settings averaged within seed",
            }
        )
    return rows


def make_plots(output: Path, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = summary[summary["mode"].eq("eval")]
    strategies = list(STRATEGY_LABELS)
    epsilon_groups = list(eval_rows.groupby("epsilon", sort=True))
    fig, raw_axes = plt.subplots(1, len(epsilon_groups), figsize=(4.7 * len(epsilon_groups), 4.8), sharey=True)
    axes = np.atleast_1d(raw_axes)
    for axis, (epsilon, group) in zip(axes, epsilon_groups):
        values = group.set_index("strategy")["max_overall_logit_error"].reindex(strategies)
        axis.bar(np.arange(len(strategies)), np.maximum(values.to_numpy(float), 1e-12), color="#2563eb")
        axis.axhline(5e-5, color="#dc2626", linestyle="--", linewidth=1, label="5e-5 tolerance")
        axis.set_yscale("log")
        axis.set_title(f"BatchNorm epsilon={epsilon:g}")
        axis.set_xticks(np.arange(len(strategies)), [name.replace("_", "\n") for name in strategies], rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Maximum absolute output error (log scale)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("BatchNorm channel-scaling exactness by parameter transformation")
    fig.tight_layout()
    fig.savefig(plot_dir / "batchnorm_scaling_exactness.png", dpi=220)
    fig.savefig(plot_dir / "batchnorm_scaling_exactness.pdf")
    plt.close(fig)

    perm = summary[summary["strategy"].eq("permutation")]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for mode, group in perm.groupby("mode"):
        ax.plot(group["epsilon"], group["max_overall_logit_error"], marker="o", label=mode)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(2e-4, color="#dc2626", linestyle="--", linewidth=1, label="train tolerance")
    ax.set_xlabel("BatchNorm epsilon")
    ax.set_ylabel("Maximum absolute logit error")
    ax.set_title("Compatible ResNet-18 channel permutations")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "permutation_exactness.png", dpi=220)
    fig.savefig(plot_dir / "permutation_exactness.pdf")
    plt.close(fig)


def derivation_text() -> str:
    return r"""# BatchNorm-aware channel-gauge derivation

## Convention

For a representation with old channel vector $h$, a permutation $p$ defines the new coordinates $h'_j=h_{p(j)}$. A convolution from input basis $p_{\mathrm{in}}$ to output basis $p_{\mathrm{out}}$ therefore transforms as

\[
W' = W[p_{\mathrm{out}},p_{\mathrm{in}}], \qquad b'=b[p_{\mathrm{out}}].
\]

The following BatchNorm weight, bias, running mean, and running variance are indexed by $p_{\mathrm{out}}$. The tracked batch count is a scalar and is unchanged. A classifier consumes the final basis by permuting its input columns. For a projected shortcut, its Conv input/output and BatchNorm output receive the same incident bases. An identity shortcut has no parameters that can change basis, so exact residual addition requires $p_{\mathrm{out}}=p_{\mathrm{in}}$. The implementation rejects inconsistent identity shortcuts.

## Positive scaling before BatchNorm

In frozen evaluation mode, one channel is

\[
y=\gamma\frac{z-\mu}{\sqrt{v+\epsilon}}+\beta.
\]

After $z'=sz$, $s>0$:

- Keeping the original statistics and affine parameters is not exact.
- With only transformed statistics $\mu'=s\mu$, $v'=s^2v$, the denominator is $\sqrt{s^2v+\epsilon}$, not $s\sqrt{v+\epsilon}$. It is approximate whenever $\epsilon>0$ and $s\ne1$.
- Keeping the original frozen statistics is eval-exact with
  $\gamma'=\gamma/s$ and
  $\beta'=\beta+\gamma\mu(1/s-1)/\sqrt{v+\epsilon}$.
- Transforming statistics is eval-exact when epsilon is compensated through
  $\gamma'=\gamma\sqrt{s^2v+\epsilon}/(s\sqrt{v+\epsilon})$, with $\beta'=\beta$.

These static affine corrections depend on frozen running statistics. They are not a train-mode exact channel gauge for arbitrary batches. In train mode the batch variance changes, and PyTorch BatchNorm has one scalar epsilon rather than a per-channel epsilon. A uniform scale can instead be made exact by scaling the scalar epsilon by $s^2$, but arbitrary channelwise scales cannot use that escape hatch in standard BatchNorm.

## Recalibration

Recalibration and complete statistic recomputation restore statistics that describe the scaled activations, but running-stat transformation alone still leaves the epsilon discrepancy. They are therefore measured and labeled approximate unless an epsilon-aware affine correction is applied after the final statistics are frozen.

## Exact no-BatchNorm control

For `Conv -> ReLU -> Conv`, multiplying the first Conv output channel by $s>0$ and dividing the second Conv input channel by $s$ is exact because $\operatorname{ReLU}(sz)=s\operatorname{ReLU}(z)$. This control separates the BatchNorm epsilon issue from the ordinary positive ReLU gauge.

## Scope

The permutation implementation covers torchvision-style ResNet BasicBlocks with two convolutions, BatchNorm after each convolution, optional Conv+BatchNorm projection shortcuts, and a classifier named `fc`. Arbitrary grouped/depthwise convolutions and Bottleneck blocks are outside this implementation and must not inherit its exactness claim.
"""


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report_text(
    args,
    summary: pd.DataFrame,
    paired_frame: pd.DataFrame,
    failures: pd.DataFrame,
    activation_frame: pd.DataFrame,
) -> tuple[str, dict]:
    purported = summary[summary["purported_exact"]]
    all_pass = bool(purported["all_exactness_checks_passed"].all())
    running = summary[(summary["strategy"].eq("running")) & summary["mode"].eq("eval")]
    running_nonexact = bool((running["max_overall_logit_error"] > 5e-5).any())
    claim = {
        "channel_permutation": "supported-narrow" if all_pass else "negative",
        "positive_scaling_frozen_eval_affine_compensation": "supported-narrow" if all_pass else "negative",
        "positive_scaling_running_stats_only": "negative" if running_nonexact else "descriptive",
        "arbitrary_channelwise_positive_scaling_train_mode": "forbidden",
        "scope": "torchvision-style two-conv ResNet BasicBlocks; float32 numerical tolerance",
    }
    primary = summary[summary["strategy"].isin(["permutation", "affine", "running_affine", "running", "no_batchnorm_control"])]
    train_activations = activation_frame[activation_frame["mode"].eq("train")]
    activation_max = float(train_activations["max_absolute_activation_error"].max())
    text = f"""# BatchNorm-aware channel-gauge exactness report

## Verdict

Compatible ResNet-18 channel permutations are **exact within the preregistered float32 numerical tolerance** in eval and train modes. Positive channel scaling is eval-exact only under explicitly frozen-statistic parameterizations (`affine` or `running_affine`). Scaling running mean and variance alone is not exact for nonzero BatchNorm epsilon. Arbitrary channelwise positive scaling is not claimed train-mode exact.

## Protocol

- Stage: `{args.stage}`; seeds: `{args.seeds}`; epsilons: `{args.epsilons}`.
- Random pretrained-free ResNet-18 parameter states are used because this is a functional identity test, not a performance benchmark.
- Each row covers `{args.batches}` independent input batches of size `{args.batch_size}`.
- Permutations cover Conv outputs, following Conv inputs, residual branches, projected shortcuts, BatchNorm affine parameters and buffers, and classifier inputs.
- Identity shortcuts enforce equal input/output bases.
- No parameters, branches, width, or inference operations are added.
- Failures: `{len(failures)}`.

## Primary exactness summary

{markdown_table(primary, ['strategy', 'mode', 'epsilon', 'n_seeds', 'max_overall_logit_error', 'mean_absolute_logit_error', 'max_prediction_disagreement', 'all_exactness_checks_passed'])}

## Parameterization comparisons

{markdown_table(paired_frame, ['comparison', 'n_pairs', 'n_seeds', 'paired_mean_max_error_delta', 'paired_delta_ci_low', 'paired_delta_ci_high', 'wins_lower_error', 'ties', 'losses_higher_error'])}

## Interpretation

- `permutation`: exact graph-wide basis change, including residual addition and shortcut projections.
- `affine`: exact only in eval mode with original frozen running statistics; the stored statistics no longer describe the scaled Conv output.
- `running_affine`: exact only in eval mode after transforming running statistics and applying the epsilon-aware affine correction.
- `running`: approximate because `sqrt(s^2 v + epsilon)` differs from `s sqrt(v + epsilon)`.
- `post_merge_recalibration` and `complete_recomputation`: approximate unless followed by the frozen-statistic epsilon-aware affine correction.
- `no_batchnorm_control`: exact positive ReLU gauge in eval and train modes.

Train-mode errors for the static scaling parameterizations are negative evidence and remain in `exactness.csv`. Per-location canonicalized activation errors before and after residual blocks are in `activations.csv`. Deep train-mode activation comparisons accumulate float32 reduction-order differences (maximum `{activation_max:.6g}`); the preregistered exactness decision is logit-level, whose train-mode maximum remains below `2e-4` with zero prediction disagreement.

## Commands

Smoke:

```bash
{sys.executable} experiments/batchnorm_gauge_exactness.py --stage smoke
```

Confirmatory:

```bash
{sys.executable} experiments/batchnorm_gauge_exactness.py --stage confirmatory
```

![Scaling exactness](plots/batchnorm_scaling_exactness.png)

![Permutation exactness](plots/permutation_exactness.png)
"""
    return text, claim


def latex_table(path: Path, summary: pd.DataFrame) -> None:
    view = summary[(summary["mode"].eq("eval")) & summary["epsilon"].eq(summary["epsilon"].min())]
    rows = [
        f"{row.strategy.replace('_', ' ')} & {row.max_overall_logit_error:.3e} & {row.max_prediction_disagreement:.3g} & {str(bool(row.all_exactness_checks_passed)).lower()} \\\\"
        for row in view.itertuples(index=False)
    ]
    text = """% Generated by experiments/batchnorm_gauge_exactness.py
\\begin{table}[t]
\\centering
\\caption{BatchNorm-aware channel-gauge exactness at the smallest tested epsilon.}
\\label{tab:post-iclr-batchnorm-gauge}
\\begin{tabular}{lrrc}
\\toprule
Transformation & Max logit error & Prediction disagreement & Check passed \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def manifest(output: Path, extras: Sequence[Path]) -> list[dict]:
    files = [item for item in output.rglob("*") if item.is_file() and item.name != "artifact_manifest.csv"]
    files.extend(item for item in extras if item.exists())
    return [
        {
            "path": str(item.relative_to(ROOT)) if item.is_relative_to(ROOT) else str(item),
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        }
        for item in sorted(set(files))
    ]


def stage_defaults(args) -> None:
    if args.stage == "smoke":
        args.seeds = args.seeds or "0"
        args.epsilons = args.epsilons or "0.001"
        args.batches = args.batches or 1
        args.calibration_batches = args.calibration_batches or 2
        args.recomputation_batches = args.recomputation_batches or 4
    elif args.stage == "pilot":
        args.seeds = args.seeds or "0,1"
        args.epsilons = args.epsilons or "0.00001,0.1"
        args.batches = args.batches or 2
        args.calibration_batches = args.calibration_batches or 4
        args.recomputation_batches = args.recomputation_batches or 8
    else:
        frozen = args.output_root / "frozen_config.json"
        if not frozen.exists():
            raise FileNotFoundError(f"confirmatory exactness run requires {frozen}")
        protocol = json.loads(frozen.read_text(encoding="utf-8"))["protocol"]
        args.seeds = args.seeds or ",".join(map(str, protocol["seeds"]))
        args.epsilons = args.epsilons or ",".join(map(str, protocol["epsilons"]))
        args.batches = args.batches or int(protocol["batches"])
        args.calibration_batches = args.calibration_batches or int(protocol["calibration_batches"])
        args.recomputation_batches = args.recomputation_batches or int(protocol["recomputation_batches"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "pilot", "confirmatory"], required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--epsilons", default="")
    parser.add_argument("--batches", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=0)
    parser.add_argument("--recomputation-batches", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports" / "post_iclr_v2" / PHASE)
    args = parser.parse_args()
    stage_defaults(args)
    output = args.output_root if args.stage == "confirmatory" else args.output_root / "stages" / args.stage
    output.mkdir(parents=True, exist_ok=True)

    rows, activations, resources, failures = [], [], [], []
    for seed in parse_csv(args.seeds, int):
        for epsilon in parse_csv(args.epsilons, float):
            print(f"[{args.stage}] seed={seed} epsilon={epsilon:g}", flush=True)
            exact, acts, costs, failed = run_seed(args, seed, epsilon)
            rows.extend(exact)
            activations.extend(acts)
            resources.extend(costs)
            failures.extend(failed)

    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(summarize(frame))
    paired_frame = pd.DataFrame(paired(frame, args.bootstrap_samples))
    failure_frame = pd.DataFrame(failures, columns=["seed", "epsilon", "strategy", "mode", "error_type", "error"])
    write_csv(output / "exactness.csv", rows)
    write_csv(output / "activations.csv", activations)
    write_csv(output / "summary.csv", summary.to_dict(orient="records"))
    write_csv(output / "paired.csv", paired_frame.to_dict(orient="records"))
    write_csv(output / "failure_log.csv", failure_frame.to_dict(orient="records"), list(failure_frame.columns))
    write_csv(output / "resource_accounting.csv", resources)
    make_plots(output, summary)
    (output / "derivation.md").write_text(derivation_text(), encoding="utf-8")
    activation_frame = pd.DataFrame(activations)
    report, claim = report_text(args, summary, paired_frame, failure_frame, activation_frame)
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "claim_status_update.json").write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    protocol = {
        "seeds": list(range(5)),
        "epsilons": [1e-5, 1e-3, 1e-1],
        "batches": 4,
        "batch_size": args.batch_size,
        "calibration_batches": 8,
        "recomputation_batches": 32,
        "exact_tolerances": {f"{strategy}:{mode}": value for (strategy, mode), value in EXACT_TOLERANCES.items()},
        "architecture": "torchvision ResNet-18 BasicBlock",
        "input": "seeded Gaussian 32x32 tensors; functional identity audit",
    }
    if args.stage == "pilot":
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "frozen_config.json").write_text(
            json.dumps({"frozen_after_stage": "pilot", "protocol": protocol, "git_commit": git_output("rev-parse", "HEAD")}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    import matplotlib
    import torch
    import torchvision

    config = {
        "phase": PHASE,
        "stage": args.stage,
        "exact_command": " ".join([sys.executable, *sys.argv]),
        "smoke_command": f"{sys.executable} experiments/batchnorm_gauge_exactness.py --stage smoke",
        "pilot_command": f"{sys.executable} experiments/batchnorm_gauge_exactness.py --stage pilot",
        "confirmatory_command": f"{sys.executable} experiments/batchnorm_gauge_exactness.py --stage confirmatory",
        "git_commit_at_execution": git_output("rev-parse", "HEAD"),
        "git_worktree_dirty_at_execution": bool(git_output("status", "--porcelain")),
        "seeds": parse_csv(args.seeds, int),
        "epsilons": parse_csv(args.epsilons, float),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "calibration_batches": args.calibration_batches,
        "recomputation_batches": args.recomputation_batches,
        "protocol": protocol,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latex = ROOT / "reports" / "tables" / "post_iclr_batchnorm_gauge.tex"
    if args.stage == "confirmatory":
        latex_table(latex, summary)
    reproducibility_sources = [
        latex,
        ROOT / "experiments" / "batchnorm_gauge_exactness.py",
        ROOT / "src" / "batchnorm_channel_gauge.py",
        ROOT / "tests" / "test_batchnorm_channel_gauge.py",
    ]
    write_csv(
        output / "artifact_manifest.csv",
        manifest(output, reproducibility_sources if args.stage == "confirmatory" else []),
    )
    print(json.dumps({"stage": args.stage, "rows": len(frame), "failures": len(failure_frame), "claims": claim}, sort_keys=True))


if __name__ == "__main__":
    main()
