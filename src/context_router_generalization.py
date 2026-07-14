"""Context-router generalization utilities using executed branch logits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.controlled_nonabelian_holonomy import controlled_group, named_generators
from src.executed_two_loop_holonomy import ExecutedMLP


TRAIN_WORDS = ("e", "s", "r", "sr")
HELDOUT_WORDS = ("rs", "ss", "rr", "srsr", "srr", "rsr", "srs", "rrs")
ROUTERS = (
    "no_router",
    "random_router",
    "majority_branch",
    "validation_face_table_router",
    "learned_feature_router",
    "supplied_context_oracle",
)


@dataclass(frozen=True)
class RouterCase:
    group_name: str
    group: object
    seed: int
    base_model: ExecutedMLP
    train_words: tuple[str, ...]
    heldout_words: tuple[str, ...]


def make_case(group_name: str, seed: int, base_dim: int = 8, n_classes: int = 4) -> RouterCase:
    group = controlled_group(group_name)
    rng = np.random.default_rng(50021 * seed + group.order)
    hidden = 24
    model = ExecutedMLP(
        hidden_weight=rng.normal(scale=0.5, size=(hidden, base_dim + 8)),
        hidden_bias=rng.normal(scale=0.1, size=hidden),
        output_weight=rng.normal(scale=0.4, size=(n_classes, hidden)),
        output_bias=rng.normal(scale=0.05, size=n_classes),
    )
    # The teacher ignores context-signature coordinates. Routers may inspect
    # those raw features, but candidate branches remain fixed executed models.
    model.hidden_weight[:, base_dim:] = 0.0
    return RouterCase(str(group_name).upper(), group, int(seed), model, TRAIN_WORDS, HELDOUT_WORDS)


def word_element(case: RouterCase, word: str):
    s, r = named_generators(case.group_name)
    out = case.group.identity
    if word == "e":
        return out
    for token in word:
        out = case.group.multiply(out, s if token == "s" else r)
    return out


def word_features(word: str) -> np.ndarray:
    if word == "e":
        tokens = ""
    else:
        tokens = word
    transitions = sum(tokens[idx] != tokens[idx - 1] for idx in range(1, len(tokens)))
    return np.asarray(
        [
            len(tokens),
            tokens.count("s"),
            tokens.count("r"),
            float(tokens.startswith("s")),
            float(tokens.endswith("s")),
            transitions,
            sum((1 if token == "s" else -1) * (idx + 1) for idx, token in enumerate(tokens)),
            1.0,
        ],
        dtype=float,
    )


def class_action(case: RouterCase, element) -> np.ndarray:
    n_classes = case.base_model.n_classes
    perm = np.arange(n_classes, dtype=int)
    natural = np.asarray(element, dtype=int)
    perm[: len(natural)] = natural
    return perm


def apply_class_action(logits: np.ndarray, perm: np.ndarray) -> np.ndarray:
    out = np.empty_like(logits)
    out[:, np.asarray(perm, dtype=int)] = logits
    return out


def generate_context_dataset(case: RouterCase, words: tuple[str, ...], n_per_word: int, split: str):
    offset = {"train": 17, "validation": 29, "test": 43}[split]
    rng = np.random.default_rng(60013 * case.seed + 601 * case.group.order + offset)
    rows_x = []
    labels = []
    word_rows = []
    element_indices = []
    element_to_index = {element: idx for idx, element in enumerate(case.group.elements)}
    base_dim = case.base_model.input_dim - 8
    for word in words:
        base = rng.normal(size=(int(n_per_word), base_dim))
        signature = np.tile(word_features(word), (int(n_per_word), 1))
        signature += rng.normal(scale=0.03, size=signature.shape)
        inputs = np.concatenate([base, signature], axis=1)
        element = word_element(case, word)
        oracle = apply_class_action(case.base_model.logits(inputs), class_action(case, element))
        rows_x.append(inputs)
        labels.append(np.argmax(oracle, axis=1).astype(np.int64))
        word_rows.extend([word] * int(n_per_word))
        element_indices.extend([element_to_index[element]] * int(n_per_word))
    return (
        np.concatenate(rows_x, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(word_rows, dtype=object),
        np.asarray(element_indices, dtype=np.int64),
    )


def all_branch_logits(case: RouterCase, inputs: np.ndarray) -> np.ndarray:
    base = case.base_model.logits(inputs)
    return np.stack([apply_class_action(base, class_action(case, element)) for element in case.group.elements], axis=1)


def infer_face_table(case: RouterCase, branch_logits: np.ndarray, labels: np.ndarray, words: np.ndarray):
    table = {}
    for word in sorted(set(words.tolist())):
        mask = words == word
        scores = [float(np.mean(np.argmax(branch_logits[mask, idx], axis=1) == labels[mask])) for idx in range(case.group.order)]
        table[word] = int(np.argmax(scores))
    return table


def fit_feature_router(features: np.ndarray, branch_targets: np.ndarray, n_branches: int, ridge: float = 1e-3):
    X = np.asarray(features, dtype=float)
    Y = np.eye(n_branches)[np.asarray(branch_targets, dtype=int)]
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)


def feature_router_predict(weights: np.ndarray, features: np.ndarray):
    scores = np.asarray(features) @ weights
    shifted = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    return np.argmax(probs, axis=1), probs.max(axis=1)


def router_assignments(
    case: RouterCase,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    validation_words: np.ndarray,
    evaluation_inputs: np.ndarray,
    evaluation_words: np.ndarray,
    true_branches: np.ndarray,
):
    validation_branches = all_branch_logits(case, validation_inputs)
    table = infer_face_table(case, validation_branches, validation_labels, validation_words)
    table_targets = np.asarray([table[word] for word in validation_words], dtype=int)
    feature_weights = fit_feature_router(validation_inputs[:, -8:], table_targets, case.group.order)
    learned, confidence = feature_router_predict(feature_weights, evaluation_inputs[:, -8:])
    counts = np.bincount(table_targets, minlength=case.group.order)
    majority = int(np.argmax(counts))
    rng = np.random.default_rng(70001 * case.seed + len(evaluation_inputs))
    return {
        "no_router": (np.zeros(len(evaluation_inputs), dtype=int), np.ones(len(evaluation_inputs))),
        "random_router": (rng.integers(0, case.group.order, len(evaluation_inputs)), np.full(len(evaluation_inputs), 1.0 / case.group.order)),
        "majority_branch": (np.full(len(evaluation_inputs), majority), np.full(len(evaluation_inputs), counts[majority] / max(counts.sum(), 1))),
        "validation_face_table_router": (
            np.asarray([table.get(str(word), majority) for word in evaluation_words], dtype=int),
            np.asarray([1.0 if str(word) in table else counts[majority] / max(counts.sum(), 1) for word in evaluation_words]),
        ),
        "learned_feature_router": (learned.astype(int), confidence.astype(float)),
        "supplied_context_oracle": (np.asarray(true_branches, dtype=int), np.ones(len(evaluation_inputs))),
    }, feature_weights


def execute_router_logits(branch_logits: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    return branch_logits[np.arange(len(branch_logits)), np.asarray(assignments, dtype=int)]
