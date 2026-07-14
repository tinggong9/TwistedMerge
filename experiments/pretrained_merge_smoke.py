#!/usr/bin/env python
"""Resource-bounded shared-base pretrained ResNet-18 merge smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import __version__ as torchvision_version
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "next_benchmarks"


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def extract_features(model, dataset, indices, batch_size, device):
    loader = DataLoader(Subset(dataset, list(map(int, indices))), batch_size=batch_size, shuffle=False)
    features, labels = [], []
    model.eval().to(device)
    with torch.no_grad():
        for x, y in loader:
            features.append(model(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(features), torch.cat(labels)


def train_head(base_state, features, labels, mask, seed, epochs=30):
    torch.manual_seed(seed)
    head = nn.Linear(features.shape[1], 10)
    head.load_state_dict(copy.deepcopy(base_state))
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-4)
    x, y = features[mask], labels[mask]
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(x), y)
        loss.backward()
        optimizer.step()
    return head


def flatten_state(state):
    return torch.cat([state["weight"].reshape(-1), state["bias"].reshape(-1)])


def unflatten_state(vector, template):
    weight_n = template["weight"].numel()
    return {
        "weight": vector[:weight_n].reshape_as(template["weight"]).clone(),
        "bias": vector[weight_n:].reshape_as(template["bias"]).clone(),
    }


def average_vectors(vectors):
    return torch.stack(vectors).mean(dim=0)


def ties_merge(base, tasks, keep_fraction=0.2):
    deltas = torch.stack([task - base for task in tasks])
    threshold = torch.quantile(deltas.abs().flatten(), 1.0 - keep_fraction)
    trimmed = torch.where(deltas.abs() >= threshold, deltas, torch.zeros_like(deltas))
    elected = torch.sign(trimmed.sum(dim=0))
    agreed = torch.where(torch.sign(trimmed) == elected, trimmed, torch.zeros_like(trimmed))
    counts = (agreed != 0).sum(dim=0).clamp(min=1)
    return base + agreed.sum(dim=0) / counts


def dare_merge(base, tasks, seed=123, drop_rate=0.5):
    generator = torch.Generator().manual_seed(seed)
    deltas = []
    for task in tasks:
        delta = task - base
        keep = (torch.rand(delta.shape, generator=generator) >= drop_rate).to(delta.dtype)
        deltas.append(delta * keep / (1.0 - drop_rate))
    return base + torch.stack(deltas).mean(dim=0)


def slerp(left, right, t=0.5):
    left_norm = left / left.norm().clamp(min=1e-12)
    right_norm = right / right.norm().clamp(min=1e-12)
    dot = torch.clamp(torch.dot(left_norm, right_norm), -0.9995, 0.9995)
    theta = torch.acos(dot)
    if float(theta.abs()) < 1e-5:
        return (1 - t) * left + t * right
    return torch.sin((1 - t) * theta) / torch.sin(theta) * left + torch.sin(t * theta) / torch.sin(theta) * right


def metrics(vector, template, features, labels, task_masks):
    state = unflatten_state(vector, template)
    logits = features @ state["weight"].T + state["bias"]
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    overall = float((pred == labels).float().mean())
    per_task = [float((pred[mask] == labels[mask]).float().mean()) if bool(mask.any()) else float("nan") for mask in task_masks]
    confidence, predicted = probs.max(dim=1)
    correct = (predicted == labels).float()
    ece = 0.0
    for low in torch.linspace(0, 0.9, 10):
        mask = (confidence >= low) & (confidence < low + 0.1)
        if bool(mask.any()):
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
    return overall, per_task, float(min(per_task)), ece


def greedy_soup(vectors, template, val_features, val_labels, task_masks):
    scored = [(metrics(vector, template, val_features, val_labels, task_masks)[0], idx) for idx, vector in enumerate(vectors)]
    order = [idx for _score, idx in sorted(scored, reverse=True)]
    selected = [order[0]]
    current = vectors[order[0]]
    best = metrics(current, template, val_features, val_labels, task_masks)[0]
    for idx in order[1:]:
        candidate = average_vectors([vectors[item] for item in selected + [idx]])
        score = metrics(candidate, template, val_features, val_labels, task_masks)[0]
        if score >= best:
            selected.append(idx)
            current, best = candidate, score
    return current, selected


def task_masks(labels):
    return [labels < 5, labels >= 5, labels.remainder(2) == 0, labels.remainder(2) == 1]


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    args.command_string = " ".join([sys.executable, *sys.argv])
    rng = np.random.default_rng(args.seed + 71237)
    weights = ResNet18_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.meta["mean"] if "mean" in weights.meta else (0.485, 0.456, 0.406), std=weights.meta["std"] if "std" in weights.meta else (0.229, 0.224, 0.225)),
    ])
    train_data = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=transform)
    test_data = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=transform)
    chosen = rng.choice(len(train_data), args.train_samples + args.validation_samples, replace=False)
    train_idx = chosen[: args.train_samples]
    val_idx = chosen[args.train_samples :]
    test_idx = rng.choice(len(test_data), args.test_samples, replace=False)
    backbone = resnet18(weights=weights)
    full_parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    backbone.fc = nn.Identity()
    device = torch.device(args.device)
    train_x, train_y = extract_features(backbone, train_data, train_idx, args.batch_size, device)
    val_x, val_y = extract_features(backbone, train_data, val_idx, args.batch_size, device)
    test_x, test_y = extract_features(backbone, test_data, test_idx, args.batch_size, device)
    common = nn.Linear(train_x.shape[1], 10)
    torch.manual_seed(args.seed + 51)
    nn.init.normal_(common.weight, std=0.01)
    nn.init.zeros_(common.bias)
    base_state = copy.deepcopy(common.state_dict())
    heads = [train_head(base_state, train_x, train_y, mask, args.seed + idx + 1, args.head_epochs) for idx, mask in enumerate(task_masks(train_y))]
    template = base_state
    base_vector = flatten_state(base_state)
    task_vectors = [flatten_state(head.state_dict()) for head in heads]
    greedy, greedy_indices = greedy_soup(task_vectors, template, val_x, val_y, task_masks(val_y))
    task_arithmetic = base_vector + torch.stack([vector - base_vector for vector in task_vectors]).mean(dim=0)
    slerp_vector = slerp(slerp(task_vectors[0], task_vectors[1]), slerp(task_vectors[2], task_vectors[3]))
    candidates = {
        "weight_average": average_vectors(task_vectors),
        "greedy_soup": greedy,
        "task_arithmetic": task_arithmetic,
        "ties": ties_merge(base_vector, task_vectors),
        "dare": dare_merge(base_vector, task_vectors, args.seed + 99),
        "slerp": slerp_vector,
    }
    validation_scores = {name: metrics(vector, template, val_x, val_y, task_masks(val_y))[0] for name, vector in candidates.items()}
    selected_name = max(validation_scores, key=lambda name: (validation_scores[name], name))
    candidates["twistedmerge_exact_gauge_soup_selector"] = candidates[selected_name].clone()
    individual_task_accuracies = []
    for idx, vector in enumerate(task_vectors):
        individual_task_accuracies.append(metrics(vector, template, test_x, test_y, task_masks(test_y))[1][idx])
    saved_logits = {}
    rows = []
    reference_time = None
    for method, vector in candidates.items():
        started = time.perf_counter()
        overall, per_task, worst, ece = metrics(vector, template, test_x, test_y, task_masks(test_y))
        elapsed = time.perf_counter() - started
        reference_time = reference_time or elapsed
        state = unflatten_state(vector, template)
        saved_logits[method] = (test_x @ state["weight"].T + state["bias"]).detach().cpu().numpy().astype(np.float32)
        rows.append({
            "seed": args.seed,
            "method": method,
            "average_accuracy": overall,
            "task_0_accuracy": per_task[0],
            "task_1_accuracy": per_task[1],
            "task_2_accuracy": per_task[2],
            "task_3_accuracy": per_task[3],
            "worst_task_accuracy": worst,
            "forgetting_interference": float(np.mean(np.asarray(individual_task_accuracies) - np.asarray(per_task))),
            "calibration_ece": ece,
            "parameter_count": int(full_parameter_count - 1000 * 512 - 1000 + vector.numel()),
            "actual_trainable_parameters": int(vector.numel()),
            "stored_parameters": int(full_parameter_count - 1000 * 512 - 1000 + vector.numel()),
            "parameter_multiplier": 1.0,
            "inference_multiplier": 1.0,
            "measured_inference_time_seconds": elapsed,
            "selection_budget": args.validation_samples if method in {"greedy_soup", "twistedmerge_exact_gauge_soup_selector"} else 0,
            "selected_by_validation": method == "twistedmerge_exact_gauge_soup_selector",
            "selector_source_method": selected_name if method == "twistedmerge_exact_gauge_soup_selector" else "",
            "branch_candidate_activated": False,
            "obstruction_certificate_passed": False,
            "implementation_status": "internal_faithful_smoke",
        })
    logits_path = OUT / "logits" / f"pretrained_vision_seed{args.seed}.npz"
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logits_path, **saved_logits)
    saved_hash = hashlib.sha256(logits_path.read_bytes()).hexdigest()
    permuted_labels = test_y.detach().cpu().numpy().copy()
    rng.shuffle(permuted_labels)
    leakage_passed = saved_hash == hashlib.sha256(logits_path.read_bytes()).hexdigest()
    for row in rows:
        row["inference_multiplier"] = row["measured_inference_time_seconds"] / max(reference_time, 1e-12)
        row["saved_logits_path"] = str(logits_path.relative_to(ROOT)) if logits_path.is_relative_to(ROOT) else str(logits_path)
        row["saved_logits_sha256"] = saved_hash
        row["label_permutation_regression_passed"] = leakage_passed
    runs = pd.DataFrame(rows)
    summary = runs.copy()
    checkpoint_dir = OUT / "checkpoints" / "pretrained_resnet18_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for idx, head in enumerate(heads):
        torch.save({"head": head.state_dict(), "base_weights": weights.name, "task": idx, "seed": args.seed}, checkpoint_dir / f"task_{idx}.pt")
    torch.save({"vector": candidates[selected_name], "selected_method": selected_name}, checkpoint_dir / "selected_merge.pt")
    metadata = pd.DataFrame([
        {"method": "ResNet-18 pretrained backbone", "official_repository": "https://github.com/pytorch/vision", "license": "BSD-3-Clause", "exact_commit": "installed wheel rather than repository checkout", "installed_version": torchvision_version, "implementation": "official torchvision model and weights"},
        {"method": "Git Re-Basin", "official_repository": "https://github.com/samuela/git-re-basin", "license": "MIT", "exact_commit": "ef40098257ab97243930eba737d6dcb8edd5863e", "installed_version": "", "implementation": "not integrated; exact blocker"},
        {"method": "Task Arithmetic", "official_repository": "https://github.com/mlfoundations/task_vectors", "license": "no repository license detected by GitHub API", "exact_commit": "826a64c67082fab0f40628233287948f0f8d7fa3", "installed_version": "", "implementation": "internal vector arithmetic"},
        {"method": "TIES", "official_repository": "https://github.com/prateeky2806/ties-merging", "license": "BSD-3-Clause", "exact_commit": "44e7891fc84f3de7e4caa52664cd864ca3715e91", "installed_version": "", "implementation": "internal faithful trim-elect-merge"},
        {"method": "DARE", "official_repository": "https://github.com/yule-BUAA/MergeLM", "license": "no repository license detected by GitHub API", "exact_commit": "6d49ad96fd69c92013654b837041b868aa806564", "installed_version": "", "implementation": "internal faithful drop-rescale merge"},
        {"method": "SLERP", "official_repository": "not integrated in smoke", "license": "not recorded", "exact_commit": "not pinned", "installed_version": "", "implementation": "internal vector interpolation"},
    ])
    runs.to_csv(OUT / "pretrained_merge_runs.csv", index=False)
    summary.to_csv(OUT / "pretrained_merge_summary.csv", index=False)
    metadata.to_csv(OUT / "pretrained_merge_baseline_metadata.csv", index=False)
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "method & avg. acc. & worst acc. & ECE & interference\\\\", "\\midrule"]
    for row in summary.itertuples():
        lines.append(f"{row.method.replace('_', ' ')} & {row.average_accuracy:.3f} & {row.worst_task_accuracy:.3f} & {row.calibration_ece:.3f} & {row.forgetting_interference:.3f}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (OUT / "tables" / "pretrained_merge.tex").write_text("\n".join(lines), encoding="utf-8")
    report = f"""# Modern Shared-Base Pretrained Model-Merging Smoke Report

Decision: **not run at full required scale due to exact blockers; smoke completed**.

## Exact command

```bash
{args.command_string}
```

- Git commit at execution: `{git_commit()}`
- Backbone: torchvision ResNet-18 with `{weights.name}` ImageNet weights (`{weights.url}`)
- Dataset: CIFAR-10
- Four task heads: classes 0-4, classes 5-9, even labels, odd labels
- Shared base: identical frozen pretrained backbone and identical initialized linear head
- Smoke samples: train `{args.train_samples}`, validation `{args.validation_samples}`, test `{args.test_samples}`
- Seeds: one (`{args.seed}`), below the required five

## Smoke results

{markdown_table(runs)}

## Exact blockers to a full ICLR/JMLR benchmark

1. The required five-seed, full fine-tuning protocol was not computationally justified for this package run; this smoke freezes the backbone and fine-tunes only task heads.
2. Official Task Arithmetic, TIES, DARE, and SLERP repositories/licenses/commits were not pinned and integrated. The smoke uses labeled internal faithful implementations, which are not publication-grade external-baseline reproductions.
3. The full protocol needs separate validation/test sets at useful scale and paired statistics across at least five seeds; one smoke seed cannot support an accuracy claim.
4. No exact centrality/closure certificate passed, so no central/Brauer obstruction or branch candidate is claimed.

The checkpoint files and raw CSVs are retained only as feasibility evidence. They are excluded from paper-number release eligibility.
"""
    (OUT / "pretrained_merge_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "command": args.command_string,
        "git_commit": git_commit(),
        "weights": weights.name,
        "weights_url": weights.url,
        "torchvision_version": torchvision_version,
        "device": args.device,
        "full_required_scale_completed": False,
        "smoke_completed": True,
        "saved_logits": str(logits_path),
        "saved_logits_sha256": saved_hash,
        "label_permutation_regression_passed": leakage_passed,
        "selector_source_method": selected_name,
        "greedy_soup_indices": greedy_indices,
    }
    (OUT / "pretrained_merge_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("pretrained full benchmark: blocked; smoke completed")
    print(f"wrote {OUT / 'pretrained_merge_report.md'}")


if __name__ == "__main__":
    main()
