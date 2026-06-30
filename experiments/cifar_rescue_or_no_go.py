#!/usr/bin/env python
"""Bounded CIFAR-10 rescue attempt with a formal no-go gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cifar_or_colored_mnist_feasibility import (  # noqa: E402
    add_row,
    average_eval,
    bootstrap_mean_ci,
    build_gauged_models,
    collect_features,
    feature_alignment_residual,
    greedy_soup,
    md_table,
    pairwise_perms,
    parse_csv,
    reference_log_scales,
    sync_perms,
    zero_logs,
)
from src.cnn_channel_gauge import (  # noqa: E402
    CnnGaugeSpec,
    SmallFashionCNN,
    average_cnn_models,
    count_parameters,
    inference_cost_units,
)
from src.greedy_safe_selector import tau_fixed_selector  # noqa: E402
from src.metrics import capture_environment  # noqa: E402
from src.model_merging_benchmark import (  # noqa: E402
    device_from_arg,
    evaluate_ensemble,
    evaluate_model,
    make_loader,
    require_torch,
    require_torchvision,
    set_seed,
)


LAYERS = ("conv1", "conv2", "fc1")
INT_COLUMNS = {
    "n_rows",
    "n_settings",
    "n_models",
    "seed",
    "epochs",
    "max_train_samples",
    "max_test_samples",
    "conv1_channels",
    "conv2_channels",
    "hidden_units",
}


@dataclass(frozen=True)
class RescueConfig:
    conv1_channels: int
    conv2_channels: int
    hidden_units: int
    epochs: int

    @property
    def label(self) -> str:
        return f"c{self.conv1_channels}_c{self.conv2_channels}_h{self.hidden_units}_e{self.epochs}"

    @property
    def spec(self) -> CnnGaugeSpec:
        return CnnGaugeSpec(
            in_channels=3,
            conv1_channels=self.conv1_channels,
            conv2_channels=self.conv2_channels,
            hidden_units=self.hidden_units,
            spatial_after_pool=8,
            num_classes=10,
        )


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | str:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return "unknown"


def parse_rescue_configs(text: str) -> list[RescueConfig]:
    configs: list[RescueConfig] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        values = [int(part.strip()) for part in item.split(",")]
        if len(values) == 3:
            values.append(-1)
        if len(values) != 4:
            raise ValueError("rescue configs must be conv1,conv2,hidden[,epochs] entries separated by semicolons")
        configs.append(RescueConfig(*values))
    return configs


def with_default_epochs(configs: list[RescueConfig], default_epochs: int) -> list[RescueConfig]:
    return [
        RescueConfig(cfg.conv1_channels, cfg.conv2_channels, cfg.hidden_units, default_epochs if cfg.epochs <= 0 else cfg.epochs)
        for cfg in configs
    ]


def subset_indices(length: int, max_samples: int, seed: int) -> list[int]:
    torch, _, _ = require_torch()
    if max_samples <= 0 or max_samples >= length:
        return list(range(length))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(length, generator=generator)[:max_samples].tolist()


def split_indices(indices: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    torch, _, _ = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator).tolist()
    n_val = max(1, int(round(len(indices) * float(val_fraction))))
    val = [indices[i] for i in order[:n_val]]
    train = [indices[i] for i in order[n_val:]]
    return train, val


def cifar_transforms(use_augmentation: bool):
    _torchvision, transforms = require_torchvision()
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if not use_augmentation:
        return eval_transform, eval_transform
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


def load_cifar_splits(args, seed: int):
    torchvision, _transforms = require_torchvision()
    torch, _, _ = require_torch()
    train_transform, eval_transform = cifar_transforms(args.augmentation)
    train_aug = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    train_eval = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=eval_transform)
    test_eval = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=eval_transform)
    train_pool = subset_indices(len(train_aug), args.max_train_samples, seed + 1000)
    train_idx, val_idx = split_indices(train_pool, args.val_fraction, seed + 2000)
    test_idx = subset_indices(len(test_eval), args.max_test_samples, seed + 3000)
    return (
        torch.utils.data.Subset(train_aug, train_idx),
        torch.utils.data.Subset(train_eval, train_idx),
        torch.utils.data.Subset(train_eval, val_idx),
        torch.utils.data.Subset(test_eval, test_idx),
        len(train_pool),
        len(test_idx),
    )


def make_model(spec: CnnGaugeSpec) -> SmallFashionCNN:
    return SmallFashionCNN(spec)


def layer_widths(spec: CnnGaugeSpec) -> dict[str, int]:
    return {"conv1": spec.conv1_channels, "conv2": spec.conv2_channels, "fc1": spec.hidden_units}


def train_rescue_model(model, loader, epochs: int, lr: float, weight_decay: float, device) -> dict[str, float]:
    torch, _, _ = require_torch()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _epoch in range(int(epochs)):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return evaluate_model(model, loader, device)


def gate_status(max_accuracy: float, args) -> tuple[bool, str, str]:
    if max_accuracy < args.plumbing_threshold:
        return False, "cifar_below_plumbing_threshold", "Formal no-go: below plumbing threshold"
    if max_accuracy < args.meaningful_threshold:
        return False, "cifar_plumbing_only", "Formal no-go for main claims: plumbing-only"
    return True, "cifar_meaningful_gate_passed", "Meaningful CIFAR gate passed"


def base_metadata(args, cfg: RescueConfig, seed: int, n_models: int, train_pool_size: int, test_size: int, individual: list[float]) -> dict:
    eligible, status, decision = gate_status(float(np.max(individual)), args)
    spec = cfg.spec
    return {
        "setting_id": f"cifar10_rescue_{cfg.label}_N{n_models}_S{seed}",
        "dataset": "cifar10",
        "dataset_role": "cifar_rescue_or_no_go",
        "architecture": "larger_relu_cnn_no_batchnorm",
        "conv1_channels": cfg.conv1_channels,
        "conv2_channels": cfg.conv2_channels,
        "hidden_units": cfg.hidden_units,
        "n_models": n_models,
        "seed": seed,
        "epochs": cfg.epochs,
        "max_train_samples": train_pool_size,
        "max_test_samples": test_size,
        "augmentation": bool(args.augmentation),
        "individual_accuracy_mean": float(np.mean(individual)),
        "individual_accuracy_max": float(np.max(individual)),
        "cifar_plumbing_threshold": args.plumbing_threshold,
        "cifar_meaningful_threshold": args.meaningful_threshold,
        "merge_claims_allowed": bool(eligible),
        "gate_decision": decision,
        "feasibility_status": status,
        "parameter_count": count_parameters(make_model(spec)),
        "inference_cost_units": inference_cost_units(spec),
        "parameter_count_multiplier": 1.0,
        "inference_time_multiplier": 1.0,
        "exact_positive_channel_scale_available": True,
    }


def run_probe(args, cfg: RescueConfig, seed: int) -> dict:
    device = device_from_arg(args.device)
    train_aug, _train_eval, val_eval, test_eval, train_pool_size, test_size = load_cifar_splits(args, seed)
    train_loader = make_loader(train_aug, args.batch_size, shuffle=True, seed=seed + 10)
    val_loader = make_loader(val_eval, args.batch_size, shuffle=False, seed=seed + 11)
    test_loader = make_loader(test_eval, args.batch_size, shuffle=False, seed=seed + 12)
    set_seed(seed + 13)
    model = make_model(cfg.spec)
    train_rescue_model(model, train_loader, cfg.epochs, args.lr, args.weight_decay, device)
    val = evaluate_model(model, val_loader, device)
    test = evaluate_model(model, test_loader, device)
    base = base_metadata(args, cfg, seed, 1, train_pool_size, test_size, [float(test["accuracy"])])
    return {
        **base,
        "setting_id": f"cifar10_rescue_probe_{cfg.label}_S{seed}",
        "method": "individual_rescue_probe",
        "val_accuracy": float(val["accuracy"]),
        "val_loss": float(val["loss"]),
        "accuracy": float(test["accuracy"]),
        "loss": float(test["loss"]),
        "selection_used_validation_only": False,
        "evaluation_status": "rescue_probe",
        "exact_relu_channel_gauge": False,
        "single_model": True,
        "capacity_matched": True,
        "ensemble_or_extra_capacity": False,
    }


def run_merge_setting(args, cfg: RescueConfig, seed: int, gate_source: str) -> list[dict]:
    device = device_from_arg(args.device)
    spec = cfg.spec
    widths = layer_widths(spec)
    train_aug, train_eval, val_eval, test_eval, train_pool_size, test_size = load_cifar_splits(args, seed)
    val_loader = make_loader(val_eval, args.batch_size, shuffle=False, seed=seed + 110)
    test_loader = make_loader(test_eval, args.batch_size, shuffle=False, seed=seed + 120)
    match_loader = make_loader(train_eval, args.batch_size, shuffle=False, seed=seed + 130)

    models = []
    individual = []
    for idx in range(args.n_models):
        model_seed = seed + idx * 1009 + 17
        set_seed(model_seed)
        model = make_model(spec)
        train_loader = make_loader(train_aug, args.batch_size, shuffle=True, seed=model_seed)
        train_rescue_model(model, train_loader, cfg.epochs, args.lr, args.weight_decay, device)
        metrics = evaluate_model(model, test_loader, device)
        individual.append(float(metrics["accuracy"]))
        model.to("cpu")
        models.append(model)

    base = base_metadata(args, cfg, seed, args.n_models, train_pool_size, test_size, individual)
    base.update({"gate_source": gate_source})
    features = {idx: collect_features(model, match_loader, device, widths, args.feature_batches) for idx, model in enumerate(models)}
    pairwise = pairwise_perms(features, args.n_models, widths)
    refs, synced, sync_residuals = sync_perms(pairwise, args.n_models)
    logs = reference_log_scales(features, synced, refs, args.n_models, widths)
    base["pairwise_alignment_residual"] = feature_alignment_residual(features, synced, args.n_models)
    base["sync_disagreement_mean"] = float(np.mean(list(sync_residuals.values())))
    rows: list[dict] = []

    _weight_model, weight_val, weight_test = average_eval(models, val_loader, test_loader, device)
    add_row(rows, base, "weight_average", weight_val, weight_test, {"exact_relu_channel_gauge": False, "single_model": True, "capacity_matched": True, "ensemble_or_extra_capacity": False})

    c2m3_models = build_gauged_models(models, synced, zero_logs(args.n_models, widths))
    _c2m3_model, c2m3_val, c2m3_test = average_eval(c2m3_models, val_loader, test_loader, device)
    add_row(rows, base, "c2m3_channel_synchronization", c2m3_val, c2m3_test, {"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "ensemble_or_extra_capacity": False})

    positive_models = build_gauged_models(models, synced, logs)
    _positive_model, positive_val, positive_test = average_eval(positive_models, val_loader, test_loader, device)
    add_row(rows, base, "positive_channel_scale", positive_val, positive_test, {"exact_relu_channel_gauge": True, "single_model": True, "capacity_matched": True, "ensemble_or_extra_capacity": False})

    soup = greedy_soup(models, [f"original:{idx}" for idx in range(args.n_models)], val_loader, test_loader, device)
    add_row(
        rows,
        base,
        "greedy_soup",
        soup["val"],
        soup["test"],
        {
            "exact_relu_channel_gauge": False,
            "single_model": True,
            "capacity_matched": True,
            "ensemble_or_extra_capacity": False,
            "soup_ingredient_count": len(soup["selected_indices"]),
            "soup_selected_labels": json.dumps(soup["selected_labels"]),
        },
    )

    by_method = {row["method"]: row for row in rows}
    metrics = {method: {"accuracy": row["val_accuracy"], "loss": row["val_loss"]} for method, row in by_method.items()}
    choice = tau_fixed_selector(
        metrics,
        challenger_pool=["positive_channel_scale", "c2m3_channel_synchronization", "weight_average"],
        tau_accuracy=args.greedy_safe_tau,
    )
    selected = by_method[choice.selected]
    add_row(
        rows,
        base,
        "greedy_safe_selector",
        {"accuracy": selected["val_accuracy"], "loss": selected["val_loss"]},
        {"accuracy": selected["accuracy"], "loss": selected["loss"]},
        {
            "exact_relu_channel_gauge": bool(selected["exact_relu_channel_gauge"]),
            "single_model": True,
            "capacity_matched": True,
            "ensemble_or_extra_capacity": False,
            "selector_chose": choice.selected,
            "selector_challenger": choice.challenger,
            "selector_val_margin": choice.validation_accuracy_delta,
            "selector_left_greedy": choice.selected != "greedy_soup",
        },
    )

    ensemble_val = evaluate_ensemble(models, val_loader, device)
    ensemble_test = evaluate_ensemble(models, test_loader, device)
    add_row(
        rows,
        base,
        "ensemble_upper_bound",
        ensemble_val,
        ensemble_test,
        {
            "exact_relu_channel_gauge": False,
            "single_model": False,
            "capacity_matched": False,
            "ensemble_or_extra_capacity": True,
            "parameter_count_multiplier": float(args.n_models),
            "inference_time_multiplier": float(args.n_models),
        },
    )

    by_method = {row["method"]: row for row in rows}
    c2m3_acc = by_method["c2m3_channel_synchronization"]["accuracy"]
    greedy_acc = by_method["greedy_soup"]["accuracy"]
    weight_acc = by_method["weight_average"]["accuracy"]
    for row in rows:
        row["accuracy_delta_vs_c2m3"] = row["accuracy"] - c2m3_acc
        row["accuracy_delta_vs_greedy_soup"] = row["accuracy"] - greedy_acc
        row["accuracy_delta_vs_weight_average"] = row["accuracy"] - weight_acc
    return rows


def summarize(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        low, high = bootstrap_mean_ci(group.get("accuracy_delta_vs_c2m3", pd.Series(dtype=float)), args.bootstrap_samples, seed=15000 + len(rows))
        rows.append(
            {
                "summary_type": "method_summary",
                "dataset": "cifar10",
                "method": method,
                "n_rows": int(len(group)),
                "n_settings": int(group["setting_id"].nunique()),
                "mean_val_accuracy": float(group["val_accuracy"].mean()) if "val_accuracy" in group else float("nan"),
                "mean_test_accuracy": float(group["accuracy"].mean()) if "accuracy" in group else float("nan"),
                "mean_individual_accuracy_max": float(group["individual_accuracy_max"].mean()) if "individual_accuracy_max" in group else float("nan"),
                "mean_delta_vs_c2m3": float(group["accuracy_delta_vs_c2m3"].mean()) if "accuracy_delta_vs_c2m3" in group else float("nan"),
                "delta_vs_c2m3_ci_low": low,
                "delta_vs_c2m3_ci_high": high,
                "mean_delta_vs_greedy_soup": float(group["accuracy_delta_vs_greedy_soup"].mean()) if "accuracy_delta_vs_greedy_soup" in group else float("nan"),
                "mean_delta_vs_weight_average": float(group["accuracy_delta_vs_weight_average"].mean()) if "accuracy_delta_vs_weight_average" in group else float("nan"),
                "merge_claims_allowed_fraction": float(group["merge_claims_allowed"].fillna(False).astype(bool).mean()) if "merge_claims_allowed" in group else 0.0,
                "feasibility_statuses": ",".join(sorted(set(group.get("feasibility_status", pd.Series(dtype=str)).dropna().astype(str)))),
            }
        )
    max_acc = float(df["individual_accuracy_max"].max()) if "individual_accuracy_max" in df else float("nan")
    meaningful, status, decision = gate_status(max_acc, args)
    merge_rows = df[df["method"].isin(["weight_average", "c2m3_channel_synchronization", "positive_channel_scale", "greedy_soup", "greedy_safe_selector"])]
    claim_reason = f"max individual accuracy={max_acc:.4f}; plumbing threshold={args.plumbing_threshold:.4f}; meaningful threshold={args.meaningful_threshold:.4f}"
    if meaningful and not merge_rows.empty:
        claim_reason += "; merge rows were evaluated, but method wins require paired CI support"
    elif max_acc < args.meaningful_threshold:
        claim_reason += "; CIFAR excluded from main merge-performance claims"
    rows.append(
        {
            "summary_type": "claim_decision",
            "dataset": "cifar10",
            "claim": "cifar_rescue_or_no_go_status",
            "claim_decision": decision,
            "claim_reason": claim_reason,
            "feasibility_statuses": status,
        }
    )
    return pd.DataFrame(rows)


def fmt(value, col):
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if col in INT_COLUMNS:
        return str(int(round(float(value))))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def local_md_table(df: pd.DataFrame, cols: list[str], max_rows=80) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in cols:
        if col not in view:
            view[col] = ""
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in view[cols].head(max_rows).to_dict("records"):
        lines.append("| " + " | ".join(fmt(row.get(col, ""), col) for col in cols) + " |")
    return "\n".join(lines)


def write_report(args, df: pd.DataFrame, summary: pd.DataFrame, path: Path):
    probes = df[df["method"] == "individual_rescue_probe"].copy()
    merges = df[df["method"] != "individual_rescue_probe"].copy()
    merge_setting_count = int(merges["setting_id"].nunique()) if not merges.empty else 0
    methods = summary[summary["summary_type"] == "method_summary"].copy()
    claims = summary[summary["summary_type"] == "claim_decision"].copy()
    claim = claims.iloc[0].to_dict() if not claims.empty else {}
    report = f"""# CIFAR Rescue Or No-Go Gate

Generated by `experiments/cifar_rescue_or_no_go.py`.

## Exact Command

```bash
{args.command_string}
```

## Bounded Rescue Scope

- Dataset: CIFAR-10 only.
- Architecture family: no-BatchNorm two-convolution ReLU CNNs with exact positive channel/permutation gauges.
- Rescue attempts: longer training, channels from `{args.rescue_configs}`, normalized inputs, and basic random crop/flip augmentation set to `{args.augmentation}`.
- Initial seeds: `{args.seeds}`.
- Gate policy: below `{args.plumbing_threshold}` is below plumbing; `{args.plumbing_threshold}` to `{args.meaningful_threshold}` is plumbing-only; above `{args.meaningful_threshold}` permits merge-method evaluation but not automatic method-win claims.
- Test labels are not used for method selection. Greedy soup and greedy-safe selection use validation metrics.

## Gate Decision

| claim | decision | reason |
| --- | --- | --- |
| {claim.get("claim", "cifar_rescue_or_no_go_status")} | {claim.get("claim_decision", "")} | {claim.get("claim_reason", "")} |

## Rescue Probes

{local_md_table(probes, ["setting_id", "conv1_channels", "conv2_channels", "hidden_units", "epochs", "augmentation", "val_accuracy", "accuracy", "feasibility_status", "gate_decision"], 40)}

## Method Summary

{local_md_table(methods, ["method", "n_rows", "n_settings", "mean_individual_accuracy_max", "mean_val_accuracy", "mean_test_accuracy", "mean_delta_vs_c2m3", "delta_vs_c2m3_ci_low", "delta_vs_c2m3_ci_high", "mean_delta_vs_greedy_soup", "feasibility_statuses"], 80)}

## Merge/Diagnostic Rows

{local_md_table(merges, ["setting_id", "method", "individual_accuracy_max", "val_accuracy", "accuracy", "accuracy_delta_vs_c2m3", "accuracy_delta_vs_greedy_soup", "single_model", "capacity_matched", "ensemble_or_extra_capacity", "feasibility_status"], 80)}

## Interpretation

- CIFAR is included in main merge-performance claims only if the bounded rescue clears the `{args.meaningful_threshold}` individual-accuracy gate and paired method statistics support the claim.
- This run has `{merge_setting_count}` merge setting(s), so method-win claims are descriptive unless a later multi-setting run supplies stronger paired intervals.
- If the rescue remains below `{args.meaningful_threshold}`, CIFAR is formally excluded from the current paper's main ML merge claims.
- The ensemble row, when present, is an upper bound with extra parameter and inference cost; it is not capacity-matched to weight averaging or C2M3.
- Positive channel scaling and channel synchronization are exact ReLU reparameterizations for this no-BatchNorm architecture.

## Environment

```json
{json.dumps({**capture_environment(), "git_commit": git_commit(), "dirty_worktree": git_dirty()}, indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescue-configs", default="32,64,256,12;64,128,256,12")
    parser.add_argument("--seeds", default="8300")
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-test-samples", type=int, default=3000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-models", type=int, default=3)
    parser.add_argument("--feature-batches", type=int, default=8)
    parser.add_argument("--plumbing-threshold", type=float, default=0.45)
    parser.add_argument("--meaningful-threshold", type=float, default=0.60)
    parser.add_argument("--run-plumbing-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--greedy-safe-tau", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    env_prefix = [
        f"{name}={os.environ[name]}"
        for name in ("PYTHONPYCACHEPREFIX", "MPLCONFIGDIR")
        if os.environ.get(name)
    ]
    args.command_string = " ".join([*env_prefix, sys.executable, *sys.argv])

    configs = parse_rescue_configs(args.rescue_configs)
    seeds = parse_csv(args.seeds, int)
    rows: list[dict] = []
    for cfg in configs:
        for seed in seeds:
            print(f"running CIFAR rescue probe config={cfg.label} seed={seed}", flush=True)
            rows.append(run_probe(args, cfg, seed))

    probe_df = pd.DataFrame(rows)
    max_probe = float(probe_df["accuracy"].max())
    best_probe = probe_df.sort_values(["accuracy", "val_accuracy"], ascending=False).iloc[0]
    best_cfg = next(cfg for cfg in configs if cfg.label == str(best_probe["setting_id"]).split("cifar10_rescue_probe_", 1)[1].rsplit("_S", 1)[0])
    best_seed = int(best_probe["seed"])
    if max_probe >= args.meaningful_threshold:
        print(f"meaningful gate passed at {max_probe:.4f}; running merge methods", flush=True)
        rows.extend(run_merge_setting(args, best_cfg, best_seed, "meaningful_probe_gate"))
    elif max_probe >= args.plumbing_threshold and args.run_plumbing_diagnostics:
        print(f"plumbing gate passed at {max_probe:.4f}; running diagnostics without claims", flush=True)
        rows.extend(run_merge_setting(args, best_cfg, best_seed, "plumbing_probe_gate"))
    else:
        print(f"stopping after probes; best CIFAR accuracy={max_probe:.4f}", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df, args)
    csv_dir = args.reports_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_path = csv_dir / "cifar_rescue_or_no_go.csv"
    summary_path = csv_dir / "cifar_rescue_or_no_go_summary.csv"
    report_path = args.reports_dir / "cifar_rescue_or_no_go_report.md"
    df.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(args, df, summary, report_path)
    for path in [results_path, summary_path, report_path]:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
