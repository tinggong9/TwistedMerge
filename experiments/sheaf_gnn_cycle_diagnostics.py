#!/usr/bin/env python
"""Tiny sheaf/GNN heterophily diagnostic experiment.

This script is intentionally self-contained.  It does not vendor or import the
Neural Sheaf Diffusion codebase; it uses the same broad idea of learned
orthogonal edge transports and measures cycle inconsistency around triangles.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


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


def parse_csv_arg(text: str, cast):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class GraphData:
    x: torch.Tensor
    y: torch.Tensor
    adjacency_norm: torch.Tensor
    undirected_edges: torch.Tensor
    directed_edges: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    triangles: torch.Tensor
    cycle_indices: torch.Tensor
    observed_heterophily: float
    edge_count: int
    triangle_count: int


def stratified_masks(y: np.ndarray, seed: int, train_frac: float = 0.2, val_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    train = np.zeros(len(y), dtype=bool)
    val = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(y), dtype=bool)
    for label in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_train = max(1, int(round(train_frac * len(idx))))
        n_val = max(1, int(round(val_frac * len(idx))))
        train[idx[:n_train]] = True
        val[idx[n_train : n_train + n_val]] = True
        test[idx[n_train + n_val :]] = True
    return train, val, test


def list_triangles(n_nodes: int, edges: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    adjacency = [set() for _ in range(n_nodes)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    triangles: list[tuple[int, int, int]] = []
    for i in range(n_nodes):
        for j in adjacency[i]:
            if j <= i:
                continue
            common = adjacency[i].intersection(adjacency[j])
            for k in common:
                if k > j:
                    triangles.append((i, j, k))
    return triangles


def ensure_triangle_floor(
    edges: set[tuple[int, int]],
    y: np.ndarray,
    min_triangles: int,
    rng: np.random.Generator,
) -> None:
    n_nodes = len(y)
    triangles = list_triangles(n_nodes, sorted(edges))
    attempts = 0
    while len(triangles) < min_triangles and attempts < 2000:
        attempts += 1
        cls = int(rng.integers(0, 2))
        same = np.flatnonzero(y == cls)
        other = np.flatnonzero(y != cls)
        if len(same) < 2 or len(other) < 1:
            break
        i, j = rng.choice(same, size=2, replace=False).tolist()
        k = int(rng.choice(other))
        for u, v in [(i, j), (j, k), (i, k)]:
            a, b = sorted((int(u), int(v)))
            if a != b:
                edges.add((a, b))
        triangles = list_triangles(n_nodes, sorted(edges))


def make_graph(
    *,
    n_nodes: int,
    feature_dim: int,
    target_heterophily: float,
    avg_degree: float,
    feature_noise: float,
    min_triangles: int,
    seed: int,
) -> GraphData:
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n_nodes // 2), dtype=np.int64)
    if len(y) < n_nodes:
        y = np.concatenate([y, np.array([0], dtype=np.int64)])
    rng.shuffle(y)

    half = n_nodes // 2
    denom = max(1.0, (half - 1) * (1.0 - target_heterophily) + half * target_heterophily)
    scale = avg_degree / denom
    p_same = min(0.95, max(0.0, scale * (1.0 - target_heterophily)))
    p_diff = min(0.95, max(0.0, scale * target_heterophily))

    edges: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = p_same if y[i] == y[j] else p_diff
            if rng.random() < p:
                edges.add((i, j))

    for i in range(n_nodes):
        if not any(i in edge for edge in edges):
            candidates = [j for j in range(n_nodes) if j != i]
            j = int(rng.choice(candidates))
            edges.add(tuple(sorted((i, j))))

    ensure_triangle_floor(edges, y, min_triangles, rng)
    edge_list = sorted(edges)
    triangles = list_triangles(n_nodes, edge_list)

    prototypes = np.zeros((2, feature_dim), dtype=np.float32)
    prototypes[0, : feature_dim // 2] = 1.0
    prototypes[1, feature_dim // 2 :] = 1.0
    x = prototypes[y] + rng.normal(scale=feature_noise, size=(n_nodes, feature_dim)).astype(np.float32)

    a = torch.eye(n_nodes, dtype=torch.float32)
    for u, v in edge_list:
        a[u, v] = 1.0
        a[v, u] = 1.0
    deg = a.sum(dim=1).clamp_min(1.0)
    d_inv_sqrt = deg.pow(-0.5)
    adjacency_norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]

    directed_edges = []
    for u, v in edge_list:
        directed_edges.append((u, v))
        directed_edges.append((v, u))
    edge_lookup = {edge: idx for idx, edge in enumerate(directed_edges)}
    cycle_indices = []
    for i, j, k in triangles:
        cycle_indices.append([edge_lookup[(i, j)], edge_lookup[(j, k)], edge_lookup[(k, i)]])

    hetero_edges = sum(int(y[u] != y[v]) for u, v in edge_list)
    observed_heterophily = hetero_edges / max(1, len(edge_list))
    train, val, test = stratified_masks(y, seed + 10007)

    return GraphData(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        adjacency_norm=adjacency_norm,
        undirected_edges=torch.tensor(edge_list, dtype=torch.long),
        directed_edges=torch.tensor(directed_edges, dtype=torch.long),
        train_mask=torch.tensor(train, dtype=torch.bool),
        val_mask=torch.tensor(val, dtype=torch.bool),
        test_mask=torch.tensor(test, dtype=torch.bool),
        triangles=torch.tensor(triangles, dtype=torch.long),
        cycle_indices=torch.tensor(cycle_indices, dtype=torch.long),
        observed_heterophily=float(observed_heterophily),
        edge_count=len(edge_list),
        triangle_count=len(triangles),
    )


class DenseGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.last_hidden: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, adjacency_norm: torch.Tensor) -> torch.Tensor:
        h = adjacency_norm @ x
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = adjacency_norm @ h
        self.last_hidden = h
        return self.lin2(h)


def rotation_matrices(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )


class RotationSheafGNN(nn.Module):
    def __init__(self, input_dim: int, channels: int, output_dim: int, dropout: float):
        super().__init__()
        self.channels = channels
        self.input_lin = nn.Linear(input_dim, channels * 2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels * 4, channels * 2),
            nn.Tanh(),
            nn.Linear(channels * 2, 1),
        )
        self.self_lin = nn.Linear(channels * 2, channels * 2, bias=False)
        self.out_lin = nn.Linear(channels * 2, output_dim)
        self.dropout = dropout
        self.last_rotations: torch.Tensor | None = None
        self.last_hidden: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, directed_edges: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.input_lin(x))
        z = F.dropout(z, p=self.dropout, training=self.training)
        n_nodes = z.size(0)
        h = z.view(n_nodes, self.channels, 2)
        src = directed_edges[:, 0]
        dst = directed_edges[:, 1]
        edge_features = torch.cat([z[src], z[dst]], dim=1)
        theta = math.pi * torch.tanh(self.edge_mlp(edge_features).squeeze(-1))
        rotations = rotation_matrices(theta)
        messages = torch.einsum("eab,ecb->eca", rotations, h[src])
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)
        deg = torch.bincount(dst, minlength=n_nodes).to(z.dtype).clamp_min(1.0).view(-1, 1, 1)
        aggregated = aggregated / deg
        hidden = F.elu(self.self_lin(z) + aggregated.reshape(n_nodes, -1))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        self.last_rotations = rotations
        self.last_hidden = hidden
        return self.out_lin(hidden)


def cycle_loss_from_rotations(rotations: torch.Tensor | None, cycle_indices: torch.Tensor) -> torch.Tensor:
    if rotations is None or cycle_indices.numel() == 0:
        device = rotations.device if rotations is not None else cycle_indices.device
        return torch.tensor(0.0, device=device)
    r01 = rotations[cycle_indices[:, 0]]
    r12 = rotations[cycle_indices[:, 1]]
    r20 = rotations[cycle_indices[:, 2]]
    composed = r20 @ r12 @ r01
    eye = torch.eye(composed.size(-1), device=composed.device, dtype=composed.dtype).expand_as(composed)
    return ((composed - eye) ** 2).sum(dim=(1, 2)).mean() / composed.size(-1)


def cycle_diagnostics(rotations: torch.Tensor | None, cycle_indices: torch.Tensor) -> tuple[float, float]:
    if rotations is None or cycle_indices.numel() == 0:
        return float("nan"), float("nan")
    with torch.no_grad():
        r01 = rotations[cycle_indices[:, 0]]
        r12 = rotations[cycle_indices[:, 1]]
        r20 = rotations[cycle_indices[:, 2]]
        composed = r20 @ r12 @ r01
        eye = torch.eye(composed.size(-1), device=composed.device, dtype=composed.dtype).expand_as(composed)
        scores = torch.linalg.matrix_norm(composed - eye, ord="fro", dim=(-2, -1)) / math.sqrt(composed.size(-1))
        return float(scores.mean().item()), float(scores.std(unbiased=False).item())


def accuracy_and_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    if int(mask.sum()) == 0:
        return float("nan"), float("nan")
    loss = F.cross_entropy(logits[mask], y[mask]).item()
    pred = logits[mask].argmax(dim=1)
    acc = (pred == y[mask]).float().mean().item()
    return float(acc), float(loss)


def hidden_variance(hidden: torch.Tensor | None) -> float:
    if hidden is None:
        return float("nan")
    return float(hidden.detach().var(dim=0, unbiased=False).mean().item())


def dirichlet_energy(hidden: torch.Tensor | None, undirected_edges: torch.Tensor) -> float:
    if hidden is None or undirected_edges.numel() == 0:
        return float("nan")
    h = hidden.detach()
    u = undirected_edges[:, 0]
    v = undirected_edges[:, 1]
    numerator = ((h[u] - h[v]) ** 2).sum(dim=1).mean()
    denominator = (h**2).sum(dim=1).mean().clamp_min(1e-12)
    return float((numerator / denominator).item())


def train_one(
    method: str,
    graph: GraphData,
    *,
    hidden_dim: int,
    sheaf_channels: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    cycle_lambda: float,
    seed: int,
) -> dict:
    set_seed(seed)
    output_dim = int(graph.y.max().item()) + 1
    if method == "gcn":
        model: nn.Module = DenseGCN(graph.x.size(1), hidden_dim, output_dim, dropout)
    else:
        model = RotationSheafGNN(graph.x.size(1), sheaf_channels, output_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    epochs_ran = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if method == "gcn":
            logits = model(graph.x, graph.adjacency_norm)
            regularizer = torch.tensor(0.0)
        else:
            logits = model(graph.x, graph.directed_edges)
            regularizer = cycle_loss_from_rotations(model.last_rotations, graph.cycle_indices)
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        if method == "rotation_sheaf_cycle_reg":
            loss = loss + cycle_lambda * regularizer
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(graph.x, graph.adjacency_norm) if method == "gcn" else model(graph.x, graph.directed_edges)
            val_acc, val_loss = accuracy_and_loss(eval_logits, graph.y, graph.val_mask)
        epochs_ran = epoch + 1
        improved = val_loss < best_val_loss - 1e-6 or (abs(val_loss - best_val_loss) <= 1e-6 and val_acc > best_val_acc)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.adjacency_norm) if method == "gcn" else model(graph.x, graph.directed_edges)
        train_acc, train_loss = accuracy_and_loss(logits, graph.y, graph.train_mask)
        val_acc, val_loss = accuracy_and_loss(logits, graph.y, graph.val_mask)
        test_acc, test_loss = accuracy_and_loss(logits, graph.y, graph.test_mask)

    rotations = None if method == "gcn" else model.last_rotations
    cycle_mean, cycle_std = cycle_diagnostics(rotations, graph.cycle_indices)
    hidden = model.last_hidden

    return {
        "method": method,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "test_loss": test_loss,
        "cycle_inconsistency_mean": cycle_mean,
        "cycle_inconsistency_std": cycle_std,
        "cycle_regularizer_lambda": cycle_lambda if method == "rotation_sheaf_cycle_reg" else 0.0,
        "hidden_feature_variance": hidden_variance(hidden),
        "dirichlet_energy": dirichlet_energy(hidden, graph.undirected_edges),
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
        else:
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header_line = "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def write_plot(df: pd.DataFrame, path: Path) -> None:
    plot_df = df[df["method"].str.contains("sheaf")].dropna(subset=["cycle_inconsistency_mean", "test_accuracy"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    markers = {"rotation_sheaf": "o", "rotation_sheaf_cycle_reg": "s"}
    for method, group in plot_df.groupby("method"):
        scatter = ax.scatter(
            group["cycle_inconsistency_mean"],
            group["test_accuracy"],
            c=group["observed_heterophily"],
            cmap="viridis",
            marker=markers.get(method, "o"),
            edgecolor="black",
            linewidth=0.4,
            label=method,
        )
    ax.set_xlabel("Mean triangle cycle inconsistency")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Sheaf cycle inconsistency vs accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Observed edge heterophily")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(df: pd.DataFrame, args: argparse.Namespace, report_path: Path, plot_path: Path) -> None:
    summary = (
        df.groupby("method")
        .agg(
            rows=("test_accuracy", "size"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_validation_accuracy=("validation_accuracy", "mean"),
            mean_cycle_inconsistency=("cycle_inconsistency_mean", "mean"),
            mean_hidden_feature_variance=("hidden_feature_variance", "mean"),
            mean_dirichlet_energy=("dirichlet_energy", "mean"),
        )
        .reset_index()
    )
    sheaf_df = df[df["method"].str.contains("sheaf")].copy()
    correlations = pd.DataFrame(
        [
            {
                "scope": "sheaf_rows",
                "cycle_vs_test_accuracy_pearson": corr(sheaf_df["cycle_inconsistency_mean"], sheaf_df["test_accuracy"]),
                "heterophily_vs_cycle_pearson": corr(sheaf_df["observed_heterophily"], sheaf_df["cycle_inconsistency_mean"]),
                "heterophily_vs_test_accuracy_pearson": corr(sheaf_df["observed_heterophily"], sheaf_df["test_accuracy"]),
            }
        ]
    )
    by_target = (
        df.groupby(["target_heterophily", "method"])
        .agg(
            mean_observed_heterophily=("observed_heterophily", "mean"),
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_cycle_inconsistency=("cycle_inconsistency_mean", "mean"),
            mean_triangle_count=("triangle_count", "mean"),
        )
        .reset_index()
    )

    lines = [
        "# Optional Sheaf/GNN Cycle Diagnostic Report",
        "",
        "## Integration Status",
        "",
        "The official Neural Sheaf Diffusion code was inspected but not run.  The local TwistedMerge venv lacks `torch_geometric`, `torch_sparse`, and `torch_scatter`, so official NSD/WebKB runs would require a separate PyG environment.  This optional run is a self-contained PyTorch-only synthetic smoke test inspired by the NSD bundle-sheaf construction.",
        "",
        "No external code was vendored or imported.",
        "",
        "## Run Configuration",
        "",
        f"- Synthetic graph family: two-class heterophilic stochastic-block graphs with explicit triangle floor.",
        f"- Target heterophily levels: `{args.heterophily_targets}`.",
        f"- Seeds: `{args.seeds}`.",
        f"- Epochs: `{args.epochs}` with validation-loss early stopping.",
        f"- Methods: dense GCN, rotation-sheaf GNN, rotation-sheaf GNN with cycle regularizer.",
        f"- Git commit: `{git_commit()}`; dirty tree during run: `{git_dirty()}`.",
        "",
        "## Method Summary",
        "",
        markdown_table(summary),
        "",
        "## Heterophily Slices",
        "",
        markdown_table(by_target),
        "",
        "## Correlations",
        "",
        markdown_table(correlations),
        "",
        "## Interpretation",
        "",
        "- The cycle score is a diagnostic over learned sheaf transports around observed triangles, not a proof of a cohomology class.",
        "- The regularized sheaf row is included only as a small ablation.  A win here would not support a general GNN regularization claim.",
        "- Because this is synthetic and small, the supported claim is limited to: cycle inconsistency can be measured and may help diagnose learned sheaf behavior on heterophilic graphs.",
        "- The unsupported boundary remains: twisted sheaf regularization improves GNNs in general.",
        "",
        "## Artifacts",
        "",
        "- CSV: `reports/csv/sheaf_gnn_cycle_diagnostics.csv`",
        f"- Plot: `{plot_path.relative_to(ROOT)}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--heterophily-targets", default="0.25,0.55,0.85")
    parser.add_argument("--n-nodes", type=int, default=120)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--avg-degree", type=float, default=8.0)
    parser.add_argument("--feature-noise", type=float, default=1.25)
    parser.add_argument("--min-triangles", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--sheaf-channels", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--cycle-lambda", type=float, default=0.05)
    args = parser.parse_args()

    seeds = parse_csv_arg(args.seeds, int)
    targets = parse_csv_arg(args.heterophily_targets, float)
    rows = []
    start = time.time()
    for target in targets:
        for seed in seeds:
            graph = make_graph(
                n_nodes=args.n_nodes,
                feature_dim=args.feature_dim,
                target_heterophily=target,
                avg_degree=args.avg_degree,
                feature_noise=args.feature_noise,
                min_triangles=args.min_triangles,
                seed=seed + int(round(target * 1000)),
            )
            graph_meta = {
                "dataset": "synthetic_heterophilic_sbm",
                "target_heterophily": target,
                "seed": seed,
                "observed_heterophily": graph.observed_heterophily,
                "edge_count": graph.edge_count,
                "triangle_count": graph.triangle_count,
                "train_nodes": int(graph.train_mask.sum().item()),
                "validation_nodes": int(graph.val_mask.sum().item()),
                "test_nodes": int(graph.test_mask.sum().item()),
                "official_nsd_run": False,
                "implementation": "self_contained_pytorch_rotation_sheaf",
            }
            for method in ["gcn", "rotation_sheaf", "rotation_sheaf_cycle_reg"]:
                result = train_one(
                    method,
                    graph,
                    hidden_dim=args.hidden_dim,
                    sheaf_channels=args.sheaf_channels,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    patience=args.patience,
                    cycle_lambda=args.cycle_lambda,
                    seed=100000 + seed * 17 + int(round(target * 1000)),
                )
                rows.append({**graph_meta, **result})

    df = pd.DataFrame(rows)
    csv_path = ROOT / "reports/csv/sheaf_gnn_cycle_diagnostics.csv"
    plot_path = ROOT / "reports/plots/sheaf_gnn_cycle_vs_accuracy.pdf"
    report_path = ROOT / "reports/sheaf_gnn_optional_report.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_plot(df, plot_path)
    write_report(df, args, report_path, plot_path)

    config = {
        "command": " ".join(sys.argv),
        "elapsed_seconds": time.time() - start,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "args": vars(args),
        "outputs": {
            "csv": str(csv_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    config_path = ROOT / "reports/configs/sheaf_gnn_cycle_diagnostics_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    print(f"Wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
