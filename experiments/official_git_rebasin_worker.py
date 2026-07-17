#!/usr/bin/env python3
"""Execute the pinned Git Re-Basin weight-matching core in a JAX environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


def load_official_module(source_root: Path):
    from jax import random

    utils = types.ModuleType("utils")
    utils.rngmix = lambda rng, value: random.fold_in(rng, hash(value))
    sys.modules["utils"] = utils
    path = source_root / "src" / "weight_matching.py"
    spec = importlib.util.spec_from_file_location("official_git_rebasin_weight_matching", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official Git Re-Basin source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    from jax import random

    official = load_official_module(args.source_root)
    payload = np.load(args.input, allow_pickle=False)
    names = {
        "dense0_kernel": "Dense_0/kernel",
        "dense0_bias": "Dense_0/bias",
        "dense1_kernel": "Dense_1/kernel",
        "dense1_bias": "Dense_1/bias",
    }
    params_a = {official_name: payload[f"a_{name}"] for name, official_name in names.items()}
    params_b = {official_name: payload[f"b_{name}"] for name, official_name in names.items()}
    permutation_spec = official.mlp_permutation_spec(num_hidden_layers=1)
    permutation = official.weight_matching(
        random.PRNGKey(args.seed),
        permutation_spec,
        params_a,
        params_b,
        max_iter=args.max_iter,
        silent=True,
    )
    aligned = official.apply_permutation(permutation_spec, permutation, params_b)
    output = {name: np.asarray(aligned[official_name]) for name, official_name in names.items()}
    np.savez_compressed(args.output, **output)
    args.metadata.write_text(
        json.dumps(
            {
                "implementation": "official_git_rebasin_weight_matching",
                "source_file": str(args.source_root / "src" / "weight_matching.py"),
                "seed": args.seed,
                "max_iter": args.max_iter,
                "permutation": {key: np.asarray(value).tolist() for key, value in permutation.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
