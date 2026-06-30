"""Tiny official Neural Sheaf Diffusion integration diagnostic.

This script intentionally runs against an external clone of
https://github.com/twitter-research/neural-sheaf-diffusion. It does not vendor
that code. The goal is to verify whether the official PyG stack can execute on
this machine and whether learned sheaf transport caches can be post-processed
for triangle/cycle diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def split_metrics(torch: Any, F: Any, model: Any, data: Any) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x)
        metrics: dict[str, float] = {}
        for name in ("train", "val", "test"):
            mask = getattr(data, f"{name}_mask")
            loss = F.nll_loss(logits[mask], data.y[mask]).item()
            pred = logits[mask].argmax(dim=1)
            acc = pred.eq(data.y[mask]).float().mean().item()
            metrics[f"{name}_loss"] = float(loss)
            metrics[f"{name}_acc"] = float(acc)
        return metrics


def undirected_triangles(edge_index: Any, num_nodes: int) -> list[tuple[int, int, int]]:
    edges = {(int(a), int(b)) for a, b in edge_index.t().tolist() if int(a) != int(b)}
    undirected = {tuple(sorted(edge)) for edge in edges}
    adjacency = {node: set() for node in range(num_nodes)}
    for a, b in undirected:
        adjacency[a].add(b)
        adjacency[b].add(a)

    triangles: list[tuple[int, int, int]] = []
    for i in range(num_nodes):
        for j in sorted(node for node in adjacency[i] if node > i):
            common = adjacency[i].intersection(adjacency[j])
            for k in sorted(node for node in common if node > j):
                triangles.append((i, j, k))
    return triangles


def cycle_diagnostics(torch: Any, model: Any, data: Any) -> dict[str, float | int | bool]:
    cached = model.sheaf_learners[0].L.detach().cpu()
    oriented_edges = model.laplacian_builder.vertex_tril_idx.detach().cpu()
    d = int(model.d)
    identity = torch.eye(d, dtype=cached.dtype)
    scale = math.sqrt(float(d))

    transitions: dict[tuple[int, int], Any] = {}
    for edge_idx, (src, dst) in enumerate(oriented_edges.t().tolist()):
        # NSD's builders cache the negative off-diagonal Laplacian block.
        # The transport convention for this diagnostic is therefore -cached.
        transitions[(int(src), int(dst))] = -cached[edge_idx]

    def transition(src: int, dst: int) -> Any | None:
        direct = transitions.get((src, dst))
        if direct is not None:
            return direct
        reverse = transitions.get((dst, src))
        if reverse is not None:
            return reverse.transpose(0, 1)
        return None

    distances: list[float] = []
    determinants: list[float] = []
    for i, j, k in undirected_triangles(data.edge_index.cpu(), int(data.x.size(0))):
        gij = transition(i, j)
        gjk = transition(j, k)
        gki = transition(k, i)
        if gij is None or gjk is None or gki is None:
            continue
        holonomy = gij @ gjk @ gki
        distances.append(float(torch.linalg.matrix_norm(holonomy - identity).item() / scale))
        determinants.append(float(torch.det(holonomy).item()))

    if not distances:
        return {
            "triangle_count": 0,
            "cycle_score_mean": float("nan"),
            "cycle_score_max": float("nan"),
            "cycle_score_min": float("nan"),
            "holonomy_det_mean": float("nan"),
            "cache_requires_grad": bool(model.sheaf_learners[0].L.requires_grad),
        }

    return {
        "triangle_count": len(distances),
        "cycle_score_mean": float(sum(distances) / len(distances)),
        "cycle_score_max": float(max(distances)),
        "cycle_score_min": float(min(distances)),
        "holonomy_det_mean": float(sum(determinants) / len(determinants)),
        "cache_requires_grad": bool(model.sheaf_learners[0].L.requires_grad),
    }


def build_nsd_args(args: argparse.Namespace, dataset: Any, data: Any, edge_weights: bool, torch: Any) -> dict[str, Any]:
    return {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "sheaf_decay": args.sheaf_decay,
        "early_stopping": args.early_stopping,
        "min_acc": 0.0,
        "stop_strategy": "loss",
        "d": args.d,
        "layers": args.layers,
        "normalised": True,
        "deg_normalised": False,
        "linear": False,
        "hidden_channels": args.hidden_channels,
        "input_dropout": 0.0,
        "dropout": 0.0,
        "left_weights": True,
        "right_weights": True,
        "add_lp": False,
        "add_hp": False,
        "use_act": True,
        "second_linear": False,
        "orth": "householder",
        "sheaf_act": "tanh",
        "edge_weights": edge_weights,
        "sparse_learner": True,
        "dataset": args.dataset,
        "seed": args.seed,
        "cuda": 0,
        "folds": 1,
        "model": "BundleSheaf",
        "entity": None,
        "evectors": 0,
        "max_t": 1.0,
        "graph_size": data.x.size(0),
        "input_dim": dataset.num_features,
        "output_dim": dataset.num_classes,
        "device": torch.device("cpu"),
    }


def run_variant(args: argparse.Namespace, edge_weights: bool, cycle_lambda: float) -> dict[str, Any]:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/private/tmp/torch_extensions")
    venv_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

    nsd_root = Path(args.nsd_root).resolve()
    sys.path.insert(0, str(nsd_root))
    os.chdir(nsd_root)

    import numpy as np
    import torch
    import torch.nn.functional as F

    from models.disc_models import DiscreteBundleSheafDiffusion
    from utils.heterophilic import get_dataset, get_fixed_splits

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = get_dataset(args.dataset)
    data = get_fixed_splits(dataset[0], args.dataset, args.fold)
    data = data.to(torch.device("cpu"))
    nsd_args = build_nsd_args(args, dataset, data, edge_weights, torch)
    model = DiscreteBundleSheafDiffusion(data.edge_index, nsd_args).to(torch.device("cpu"))

    sheaf_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam(
        [
            {"params": sheaf_params, "weight_decay": args.sheaf_decay},
            {"params": other_params, "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
    )

    best_val_acc = -1.0
    best_test_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    regularizer_applied = False
    regularizer_status = "not_requested"

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x)
        nll = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        loss = nll
        if cycle_lambda > 0:
            cached = model.sheaf_learners[0].L
            if cached.requires_grad:
                regularizer_status = "available_but_not_implemented_in_probe"
            else:
                regularizer_status = "not_applied_official_cache_is_detached"
        loss.backward()
        optimizer.step()

        metrics = split_metrics(torch, F, model, data)
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_val_acc = metrics["val_acc"]
            best_test_acc = metrics["test_acc"]
            best_epoch = epoch

    final_metrics = split_metrics(torch, F, model, data)
    cycles = cycle_diagnostics(torch, model, data)

    variant = "bundle_sheaf_weighted_cache" if edge_weights else "bundle_sheaf_connection_cache"
    if cycle_lambda > 0:
        variant = f"{variant}_cycle_regularizer_attempt"
    interpretation = (
        "weighted_restriction_cache_not_pure_connection"
        if edge_weights
        else "unweighted_orthogonal_connection_cache"
    )

    row: dict[str, Any] = {
        "dataset": args.dataset,
        "fold": args.fold,
        "seed": args.seed,
        "variant": variant,
        "run_status": "success",
        "model": "BundleSheaf",
        "epochs": args.epochs,
        "d": args.d,
        "layers": args.layers,
        "hidden_channels": args.hidden_channels,
        "edge_weights": edge_weights,
        "cycle_lambda": cycle_lambda,
        "cycle_regularizer_applied": regularizer_applied,
        "cycle_regularizer_status": regularizer_status,
        "map_interpretation": interpretation,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "final_train_loss": final_metrics["train_loss"],
        "final_val_loss": final_metrics["val_loss"],
        "final_test_loss": final_metrics["test_loss"],
        "final_train_acc": final_metrics["train_acc"],
        "final_val_acc": final_metrics["val_acc"],
        "final_test_acc": final_metrics["test_acc"],
        "nsd_root": str(nsd_root),
        "nsd_head": git_head(nsd_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    row.update(cycles)
    return row


def main() -> None:
    repo_root = Path.cwd().resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsd-root", default="/private/tmp/neural-sheaf-diffusion")
    parser.add_argument("--dataset", default="texas")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--sheaf-decay", type=float, default=5e-3)
    parser.add_argument("--early-stopping", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--edge-weight-modes", nargs="+", type=parse_bool, default=[False, True])
    parser.add_argument("--cycle-lambda-attempt", type=float, default=1.0)
    parser.add_argument("--output-csv", default="reports/csv/nsd_cycle_diagnostics.csv")
    parser.add_argument("--config-json", default="reports/configs/nsd_official_integration_config.json")
    args = parser.parse_args()

    rows = []
    for edge_weights in args.edge_weight_modes:
        rows.append(run_variant(args, edge_weights=edge_weights, cycle_lambda=0.0))
    rows.append(run_variant(args, edge_weights=False, cycle_lambda=args.cycle_lambda_attempt))

    os.chdir(repo_root)
    output_csv = repo_root / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config_json = repo_root / args.config_json
    config_json.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "command": " ".join(sys.argv),
        "twistedmerge_cwd": str(repo_root),
        "args": vars(args),
        "rows": len(rows),
        "outputs": {"csv": str(output_csv), "config_json": str(config_json)},
    }
    config_json.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {config_json}")


if __name__ == "__main__":
    main()
