"""Small PyTorch model-merging benchmark utilities.

The benchmark intentionally uses architectures with one permutable hidden
layer/channel block.  That keeps permutation alignment and cycle-defect
measurement explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import numpy as np

try:  # Keep this module importable enough to produce dependency errors cleanly.
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
except Exception:  # pragma: no cover - depends on optional dependency state
    _torch = None
    _nn = None
    _F = None


def require_torch():
    if _torch is None or _nn is None or _F is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required. Install with `python -m pip install -r requirements.txt`."
        )
    return _torch, _nn, _F


def require_torchvision():
    try:
        import torchvision
        import torchvision.transforms as T
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torchvision is required. Install with `python -m pip install -r requirements.txt`."
        ) from exc
    return torchvision, T


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_shape: tuple[int, int, int]
    num_classes: int = 10

    @property
    def input_dim(self) -> int:
        c, h, w = self.input_shape
        return c * h * w


_MODULE_BASE = _nn.Module if _nn is not None else object
_DATASET_BASE = _torch.utils.data.Dataset if _torch is not None else object


class PermutableMLP(_MODULE_BASE):
    def __init__(self, input_dim: int, width: int, num_classes: int = 10):
        _, nn, _ = require_torch()
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(input_dim, width)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x, return_features: bool = False):
        _, _, F = require_torch()
        h = F.relu(self.hidden(self.flatten(x)))
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


class PermutableCNN(_MODULE_BASE):
    def __init__(self, in_channels: int, width: int, num_classes: int = 10):
        torch, nn, _ = require_torch()
        super().__init__()
        self.conv = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x, return_features: bool = False):
        _, _, F = require_torch()
        h_map = F.relu(self.conv(x))
        h = self.pool(h_map).flatten(1)
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


def make_model(architecture: str, spec: DatasetSpec, width: int):
    if architecture == "mlp":
        return PermutableMLP(spec.input_dim, width, spec.num_classes)
    if architecture == "cnn":
        return PermutableCNN(spec.input_shape[0], width, spec.num_classes)
    raise ValueError(f"unknown architecture: {architecture}")


def set_seed(seed: int) -> None:
    torch, _, _ = require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device: str):
    torch, _, _ = require_torch()
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


class DomainShiftDataset(_DATASET_BASE):
    def __init__(self, base, shift: str, model_index: int, n_models: int):
        self.base = base
        self.shift = shift
        self.model_index = model_index
        self.n_models = max(n_models, 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        torch, _, _ = require_torch()
        x, y = self.base[idx]
        if self.shift == "none":
            return x, y
        if self.shift == "input_noise":
            scale = 0.04 * (self.model_index + 1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1000003 * self.model_index + idx)
            x = torch.clamp(x + scale * torch.randn(x.shape, generator=generator), 0.0, 1.0)
            return x, y
        if self.shift == "brightness":
            factor = 0.75 + 0.5 * self.model_index / max(self.n_models - 1, 1)
            x = torch.clamp(x * factor, 0.0, 1.0)
            return x, y
        raise ValueError(f"unknown domain shift: {self.shift}")


def subset_dataset(dataset, max_samples: int | None, seed: int):
    torch, _, _ = require_torch()
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return torch.utils.data.Subset(dataset, indices)


def load_dataset(name: str, root: Path, max_train_samples: int, max_test_samples: int, seed: int):
    torchvision, T = require_torchvision()
    torch, _, _ = require_torch()
    name = name.lower()
    if name == "mnist":
        transform = T.ToTensor()
        train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
        spec = DatasetSpec(name="mnist", input_shape=(1, 28, 28))
    elif name in {"fashion_mnist", "fashion-mnist", "fashionmnist"}:
        transform = T.ToTensor()
        train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        spec = DatasetSpec(name="fashion_mnist", input_shape=(1, 28, 28))
    elif name == "cifar10":
        transform = T.ToTensor()
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        spec = DatasetSpec(name="cifar10", input_shape=(3, 32, 32))
    elif name == "fake-mnist":
        transform = T.ToTensor()
        train = torchvision.datasets.FakeData(size=max_train_samples, image_size=(1, 28, 28), num_classes=10, transform=transform)
        test = torchvision.datasets.FakeData(size=max_test_samples, image_size=(1, 28, 28), num_classes=10, transform=transform)
        spec = DatasetSpec(name="fake-mnist", input_shape=(1, 28, 28))
    elif name == "fake-cifar10":
        transform = T.ToTensor()
        train = torchvision.datasets.FakeData(size=max_train_samples, image_size=(3, 32, 32), num_classes=10, transform=transform)
        test = torchvision.datasets.FakeData(size=max_test_samples, image_size=(3, 32, 32), num_classes=10, transform=transform)
        spec = DatasetSpec(name="fake-cifar10", input_shape=(3, 32, 32))
    else:
        raise ValueError(f"unknown dataset: {name}")
    train = subset_dataset(train, max_train_samples, seed)
    test = subset_dataset(test, max_test_samples, seed + 1)
    return spec, train, test


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int):
    torch, _, _ = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_model(model, loader, epochs: int, lr: float, device) -> dict[str, float]:
    torch, _, _ = require_torch()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return evaluate_model(model, loader, device)


def evaluate_model(model, loader, device) -> dict[str, float]:
    torch, _, _ = require_torch()
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
            total += int(y.numel())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def evaluate_ensemble(models: list, loader, device) -> dict[str, float]:
    torch, _, _ = require_torch()
    for model in models:
        model.to(device)
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = torch.stack([model(x) for model in models], dim=0).mean(dim=0)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
            total += int(y.numel())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def collect_features(model, loader, device, max_batches: int = 8) -> np.ndarray:
    torch, _, _ = require_torch()
    model.to(device)
    model.eval()
    features = []
    with torch.no_grad():
        for batch_idx, (x, _y) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            _, h = model(x, return_features=True)
            features.append(h.detach().cpu())
    return torch.cat(features, dim=0).numpy()


def linear_sum_assignment_max(similarity: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-similarity)
        perm = np.empty(similarity.shape[0], dtype=int)
        perm[rows] = cols
        return perm
    except Exception:
        remaining = set(range(similarity.shape[1]))
        perm = np.empty(similarity.shape[0], dtype=int)
        for row in np.argsort(-similarity.max(axis=1)):
            col = max(remaining, key=lambda candidate: similarity[row, candidate])
            perm[row] = col
            remaining.remove(col)
        return perm


def activation_permutation(features_i: np.ndarray, features_j: np.ndarray) -> np.ndarray:
    xi = features_i - features_i.mean(axis=0, keepdims=True)
    xj = features_j - features_j.mean(axis=0, keepdims=True)
    xi = xi / np.maximum(np.linalg.norm(xi, axis=0, keepdims=True), 1e-12)
    xj = xj / np.maximum(np.linalg.norm(xj, axis=0, keepdims=True), 1e-12)
    similarity = xi.T @ xj
    return linear_sum_assignment_max(similarity)


def weight_permutation(model_i, model_j, architecture: str) -> np.ndarray:
    if architecture == "mlp":
        wi = model_i.hidden.weight.detach().cpu().numpy()
        wj = model_j.hidden.weight.detach().cpu().numpy()
    elif architecture == "cnn":
        wi = model_i.conv.weight.detach().cpu().numpy().reshape(model_i.conv.out_channels, -1)
        wj = model_j.conv.weight.detach().cpu().numpy().reshape(model_j.conv.out_channels, -1)
    else:
        raise ValueError(architecture)
    wi = wi / np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-12)
    wj = wj / np.maximum(np.linalg.norm(wj, axis=1, keepdims=True), 1e-12)
    return linear_sum_assignment_max(wi @ wj.T)


def permutation_matrix(perm: np.ndarray) -> np.ndarray:
    p = np.zeros((len(perm), len(perm)), dtype=float)
    p[np.arange(len(perm)), perm] = 1.0
    return p


def perturb_permutation(perm: np.ndarray, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of `perm` after deterministic random transpositions."""
    out = np.asarray(perm, dtype=int).copy()
    if len(out) < 2 or n_swaps <= 0:
        return out
    for _ in range(n_swaps):
        a, b = rng.choice(len(out), size=2, replace=False)
        out[a], out[b] = out[b], out[a]
    return out


def inject_pairwise_permutation_noise(
    pairwise_perms: dict[tuple[int, int], np.ndarray],
    n_models: int,
    width: int,
    swap_fraction: float,
    seed: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Perturb directed pairwise alignments while preserving valid permutations.

    This is a controlled diagnostic intervention, not a model-training effect:
    it varies cycle defects at fixed trained models so the verification report
    can separate alignment-score sensitivity from ordinary weight-average
    degradation.
    """
    rng = np.random.default_rng(seed)
    n_swaps = int(round(max(0.0, swap_fraction) * width))
    out: dict[tuple[int, int], np.ndarray] = {}
    for i, j in product(range(n_models), repeat=2):
        perm = pairwise_perms[(i, j)]
        out[(i, j)] = np.arange(width) if i == j else perturb_permutation(perm, n_swaps, rng)
    return out


def compute_pairwise_permutations(models: list, architecture: str, loader, device, method: str) -> dict[tuple[int, int], np.ndarray]:
    n = len(models)
    features = None
    if method == "activation":
        features = [collect_features(model, loader, device) for model in models]
    perms: dict[tuple[int, int], np.ndarray] = {}
    for i, j in product(range(n), repeat=2):
        if i == j:
            width = model_width(models[i], architecture)
            perms[(i, j)] = np.arange(width)
        elif method == "activation":
            assert features is not None
            perms[(i, j)] = activation_permutation(features[i], features[j])
        elif method == "weight":
            perms[(i, j)] = weight_permutation(models[i], models[j], architecture)
        else:
            raise ValueError(f"unknown matching method: {method}")
    return perms


def model_width(model, architecture: str) -> int:
    if architecture == "mlp":
        return int(model.hidden.out_features)
    if architecture == "cnn":
        return int(model.conv.out_channels)
    raise ValueError(architecture)


def cycle_score(pairwise_perms: dict[tuple[int, int], np.ndarray], n_models: int, width: int) -> tuple[float, list[dict[str, float]]]:
    rows = []
    scores = []
    eye = np.eye(width)
    denom = np.sqrt(2.0 * width)
    for i, j, k in combinations(range(n_models), 3):
        defect = (
            permutation_matrix(pairwise_perms[(i, j)])
            @ permutation_matrix(pairwise_perms[(j, k)])
            @ permutation_matrix(pairwise_perms[(k, i)])
        )
        score = float(np.linalg.norm(defect - eye, ord="fro") / denom)
        scores.append(score)
        rows.append({"i": i, "j": j, "k": k, "cycle_defect": score})
    return (float(np.mean(scores)) if scores else 0.0), rows


def invert_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def compose_perm(p_ab: np.ndarray, p_bc: np.ndarray) -> np.ndarray:
    """Compose permutations represented by matrices P_ab @ P_bc."""
    return p_bc[p_ab]


def permutation_disagreement(observed: np.ndarray, implied: np.ndarray) -> float:
    return float(np.mean(observed != implied))


def synchronize_permutations(pairwise_perms: dict[tuple[int, int], np.ndarray], n_models: int) -> tuple[int, dict[int, np.ndarray], float]:
    """Choose a global reference whose induced pairwise permutations best fit observations."""
    best_ref = 0
    best_score = float("inf")
    best_q: dict[int, np.ndarray] = {}
    for ref in range(n_models):
        q = {i: pairwise_perms[(ref, i)] for i in range(n_models)}
        disagreements = []
        for i, j in product(range(n_models), repeat=2):
            implied = compose_perm(invert_perm(q[i]), q[j])
            disagreements.append(permutation_disagreement(pairwise_perms[(i, j)], implied))
        score = float(np.mean(disagreements))
        if score < best_score:
            best_score = score
            best_ref = ref
            best_q = q
    return best_ref, best_q, best_score


def clone_model(model, architecture: str, spec: DatasetSpec, width: int):
    torch, _, _ = require_torch()
    cloned = make_model(architecture, spec, width)
    cloned.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    return cloned


def permute_model_to_reference(model, architecture: str, spec: DatasetSpec, width: int, perm: np.ndarray):
    aligned = clone_model(model, architecture, spec, width)
    with require_torch()[0].no_grad():
        if architecture == "mlp":
            aligned.hidden.weight.copy_(model.hidden.weight.detach().cpu()[perm, :])
            aligned.hidden.bias.copy_(model.hidden.bias.detach().cpu()[perm])
            aligned.classifier.weight.copy_(model.classifier.weight.detach().cpu()[:, perm])
            aligned.classifier.bias.copy_(model.classifier.bias.detach().cpu())
        elif architecture == "cnn":
            aligned.conv.weight.copy_(model.conv.weight.detach().cpu()[perm, :, :, :])
            aligned.conv.bias.copy_(model.conv.bias.detach().cpu()[perm])
            aligned.classifier.weight.copy_(model.classifier.weight.detach().cpu()[:, perm])
            aligned.classifier.bias.copy_(model.classifier.bias.detach().cpu())
        else:
            raise ValueError(architecture)
    return aligned


def average_models(models: list, architecture: str, spec: DatasetSpec, width: int):
    torch, _, _ = require_torch()
    merged = make_model(architecture, spec, width)
    state = merged.state_dict()
    source_states = [model.state_dict() for model in models]
    with torch.no_grad():
        for key in state:
            state[key].copy_(torch.stack([src[key].detach().cpu() for src in source_states], dim=0).mean(dim=0))
    merged.load_state_dict(state)
    return merged


def greedy_soup(models: list, val_loader, test_loader, device, architecture: str, spec: DatasetSpec, width: int) -> tuple[object, list[int], dict[str, float]]:
    scored = []
    for idx, model in enumerate(models):
        metrics = evaluate_model(model, val_loader, device)
        scored.append((metrics["accuracy"], idx))
    order = [idx for _acc, idx in sorted(scored, reverse=True)]
    soup_indices = [order[0]]
    soup = clone_model(models[order[0]], architecture, spec, width)
    best_acc = evaluate_model(soup, val_loader, device)["accuracy"]
    for idx in order[1:]:
        candidate_indices = soup_indices + [idx]
        candidate = average_models([models[item] for item in candidate_indices], architecture, spec, width)
        candidate_acc = evaluate_model(candidate, val_loader, device)["accuracy"]
        if candidate_acc >= best_acc:
            soup = candidate
            soup_indices = candidate_indices
            best_acc = candidate_acc
    return soup, soup_indices, evaluate_model(soup, test_loader, device)


def rank_lifted_branch_models(
    aligned_models: list,
    pairwise_perms: dict[tuple[int, int], np.ndarray],
    n_branches: int,
    architecture: str,
    spec: DatasetSpec,
    width: int,
) -> list:
    if n_branches <= 1 or len(aligned_models) <= 1:
        return [average_models(aligned_models, architecture, spec, width)]
    n = len(aligned_models)
    distances = []
    for idx in range(n):
        avg = np.mean([
            permutation_disagreement(pairwise_perms[(idx, j)], np.arange(width))
            for j in range(n)
            if j != idx
        ])
        distances.append((float(avg), idx))
    seeds = [idx for _score, idx in sorted(distances, reverse=True)[: min(n_branches, n)]]
    clusters = {seed: [] for seed in seeds}
    for idx in range(n):
        seed = min(seeds, key=lambda candidate: permutation_disagreement(pairwise_perms[(candidate, idx)], np.arange(width)))
        clusters[seed].append(idx)
    branches = []
    for indices in clusters.values():
        branches.append(average_models([aligned_models[idx] for idx in indices], architecture, spec, width))
    return branches


def save_checkpoint(model, path: Path, metadata: dict) -> None:
    torch, _, _ = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def format_markdown_table(rows: Iterable[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
