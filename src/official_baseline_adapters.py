"""Thin, auditable adapters around pinned official baseline source trees.

The adapters intentionally keep checkpoint conversion and evaluation in this
repository while executing the matching kernels from the official source
files.  Callers must label results ``adapter-assisted official core`` rather
than unmodified end-to-end official runs.
"""

from __future__ import annotations

import copy
import enum
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import NamedTuple

import numpy as np


GIT_REBASIN_ARRAY_KEYS = (
    "dense0_kernel",
    "dense0_bias",
    "dense1_kernel",
    "dense1_bias",
)

_SOURCE_MODULE_CACHE: dict[tuple[str, str], object] = {}


def torch_state_to_git_rebasin_arrays(state_dict) -> dict[str, np.ndarray]:
    """Convert a one-hidden-layer PyTorch MLP state to official Flax axes."""

    return {
        "dense0_kernel": state_dict["hidden.weight"].detach().cpu().numpy().T,
        "dense0_bias": state_dict["hidden.bias"].detach().cpu().numpy(),
        "dense1_kernel": state_dict["classifier.weight"].detach().cpu().numpy().T,
        "dense1_bias": state_dict["classifier.bias"].detach().cpu().numpy(),
    }


def git_rebasin_arrays_to_torch_state(arrays: dict[str, np.ndarray], template_state):
    """Convert official Flax-axis arrays back to a PyTorch state dictionary."""

    import torch

    output = copy.deepcopy(template_state)
    values = {
        "hidden.weight": arrays["dense0_kernel"].T,
        "hidden.bias": arrays["dense0_bias"],
        "classifier.weight": arrays["dense1_kernel"].T,
        "classifier.bias": arrays["dense1_bias"],
    }
    for key, value in values.items():
        output[key] = torch.as_tensor(value, dtype=template_state[key].dtype)
    return output


def average_state_dicts(states):
    """Capacity-preserving arithmetic average used after official alignment."""

    import torch

    output = copy.deepcopy(states[0])
    with torch.no_grad():
        for key in output:
            if output[key].is_floating_point():
                output[key] = torch.stack([state[key] for state in states]).mean(dim=0)
            else:
                output[key] = states[0][key].clone()
    return output


def flatten_float_state(state_dict):
    """Flatten floating tensors in stable key order for task-vector adapters."""

    import torch

    keys = [key for key in sorted(state_dict) if state_dict[key].is_floating_point()]
    vector = torch.cat([state_dict[key].detach().reshape(-1).cpu() for key in keys])
    meta = [(key, tuple(state_dict[key].shape), state_dict[key].dtype, state_dict[key].numel()) for key in keys]
    return vector, meta


def vector_to_float_state(vector, meta, template_state):
    """Restore a stable flat vector into a copied state dictionary."""

    output = copy.deepcopy(template_state)
    offset = 0
    for key, shape, dtype, count in meta:
        output[key] = vector[offset : offset + count].reshape(shape).to(dtype=dtype).clone()
        offset += count
    if offset != vector.numel():
        raise ValueError(f"vector has {vector.numel()} values but metadata consumed {offset}")
    return output


def load_official_task_vector_core(source_root: Path):
    cache_key = ("task_vectors", str(source_root.resolve()))
    if cache_key in _SOURCE_MODULE_CACHE:
        return _SOURCE_MODULE_CACHE[cache_key]
    path = source_root / "src" / "task_vectors.py"
    if not path.exists():
        raise FileNotFoundError(path)
    module = _load_module("official_task_vectors_core", path)
    _SOURCE_MODULE_CACHE[cache_key] = module
    return module


def official_task_arithmetic_state(base_state, task_states, source_root: Path, *, scale: float):
    """Apply the author TaskVector addition kernel to state-dict deltas."""

    official = load_official_task_vector_core(source_root)
    vectors = []
    for state in task_states:
        delta = {
            key: state[key].detach().cpu() - base_state[key].detach().cpu()
            for key in base_state
            if base_state[key].is_floating_point()
        }
        vectors.append(official.TaskVector(vector=delta))
    combined = sum(vectors)
    output = copy.deepcopy(base_state)
    for key, value in combined.vector.items():
        output[key] = base_state[key].detach().cpu() + float(scale) * value.detach().cpu()
    return output


def load_official_ties_core(source_root: Path):
    cache_key = ("ties", str(source_root.resolve()))
    if cache_key in _SOURCE_MODULE_CACHE:
        return _SOURCE_MODULE_CACHE[cache_key]
    path = source_root / "src" / "utils" / "merge_utils.py"
    if not path.exists():
        raise FileNotFoundError(path)
    _module("src.utils")
    analysis = _module("src.utils.analysis_utils", logging=logging)
    analysis.__all__ = ["logging"]
    module = _load_module("official_ties_merge_utils", path)
    _SOURCE_MODULE_CACHE[cache_key] = module
    return module


def official_ties_state(base_state, task_states, source_root: Path, *, density: float, scale: float):
    """Run the official TIES trim/elect/disjoint-mean kernel."""

    import torch

    if not 0.0 < float(density) <= 1.0:
        raise ValueError(f"TIES density must be in (0, 1], got {density}")
    official = load_official_ties_core(source_root)
    base_vector, meta = flatten_float_state(base_state)
    deltas = torch.stack([flatten_float_state(state)[0] - base_vector for state in task_states])
    # Official topk_values_mask computes kthvalue(d - int(d * K)); K=1
    # therefore requests the invalid index zero.  The immediately preceding
    # float yields index one and the intended keep-all mask for finite d.
    retained_fraction = np.nextafter(1.0, 0.0) if float(density) == 1.0 else float(density)
    merged = official.merge_methods(
        reset_type="topk",
        flat_task_checks=deltas,
        reset_thresh=retained_fraction,
        resolve_method="mass",
        merge_func="dis-mean",
    )
    return vector_to_float_state(base_vector + float(scale) * merged, meta, base_state)


def load_official_dare_core(source_root: Path):
    cache_key = ("dare", str(source_root.resolve()))
    if cache_key in _SOURCE_MODULE_CACHE:
        return _SOURCE_MODULE_CACHE[cache_key]
    utils_path = source_root / "utils" / "utils.py"
    task_vector_path = source_root / "model_merging_methods" / "task_vector.py"
    mask_path = source_root / "model_merging_methods" / "mask_weights_utils.py"
    for path in (utils_path, task_vector_path, mask_path):
        if not path.exists():
            raise FileNotFoundError(path)
    _module("utils")
    official_utils = _load_module("utils.utils", utils_path)
    del official_utils
    _module("model_merging_methods")
    task_vector = _load_module("model_merging_methods.task_vector", task_vector_path)
    del task_vector
    module = _load_module("official_dare_mask_weights", mask_path)
    _SOURCE_MODULE_CACHE[cache_key] = module
    return module


def official_dare_state(base_state, task_states, source_root: Path, *, drop_rate: float, scale: float, seed: int):
    """Run DARE's official random drop-and-rescale kernel on task deltas."""

    import torch

    official = load_official_dare_core(source_root)
    base_vector, meta = flatten_float_state(base_state)
    masked = []
    for index, state in enumerate(task_states):
        delta = flatten_float_state(state)[0] - base_vector
        torch.manual_seed(int(seed) + index)
        masked.append(
            official.mask_input_with_mask_rate(
                input_tensor=delta,
                mask_rate=float(drop_rate),
                use_rescale=True,
                mask_strategy="random",
            )
        )
    merged = torch.stack(masked).mean(dim=0)
    return vector_to_float_state(base_vector + float(scale) * merged, meta, base_state)


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class C2M3PermutationSpec(NamedTuple):
    perm_to_layers_and_axes: dict
    layer_and_axes_to_perm: dict


def c2m3_mlp_permutation_spec() -> C2M3PermutationSpec:
    axes = {
        "hidden.weight": ("P_0", None),
        "hidden.bias": ("P_0",),
        "classifier.weight": (None, "P_0"),
        "classifier.bias": (None,),
    }
    reverse: dict[str, list[tuple[str, int]]] = {"P_0": []}
    for key, values in axes.items():
        for axis, value in enumerate(values):
            if value is not None:
                reverse[value].append((key, axis))
    return C2M3PermutationSpec(reverse, axes)


def load_official_c2m3_core(source_root: Path):
    """Load official C2M3 matching files with import-only compatibility shims.

    C2M3's package initializer requires its full Hydra/Lightning application.
    The matching files themselves need only Torch/SciPy plus type-level imports.
    These shims avoid executing the application initializer and do not replace
    any optimization or permutation function used by the run.
    """

    import torch

    cache_key = ("c2m3", str(source_root.resolve()))
    if cache_key in _SOURCE_MODULE_CACHE:
        return _SOURCE_MODULE_CACHE[cache_key]

    matching = source_root / "src" / "ccmm" / "matching"
    required = [
        matching / "utils.py",
        matching / "weight_matching.py",
        matching / "frank_wolfe_matching.py",
        matching / "frank_wolfe_sync_matching.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing official C2M3 files: {missing}")

    _module("ccmm")
    _module("ccmm.matching")
    _module("ccmm.utils")
    permutation_module = _module(
        "ccmm.matching.permutation_spec",
        PermutationSpec=C2M3PermutationSpec,
    )
    del permutation_module
    _module(
        "ccmm.utils.utils",
        ModelParams=dict,
        get_model=lambda model: getattr(model, "model", model),
        to_np=lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value),
    )
    _module(
        "pytorch_lightning",
        LightningModule=torch.nn.Module,
        seed_everything=lambda seed: torch.manual_seed(int(seed)),
    )
    _module("backports")
    _module("backports.strenum", StrEnum=enum.StrEnum)

    utils = _load_module("ccmm.matching.utils", matching / "utils.py")
    weight_matching = _load_module("ccmm.matching.weight_matching", matching / "weight_matching.py")
    del weight_matching
    frank_wolfe = _load_module("ccmm.matching.frank_wolfe_matching", matching / "frank_wolfe_matching.py")
    del frank_wolfe
    sync = _load_module("ccmm.matching.frank_wolfe_sync_matching", matching / "frank_wolfe_sync_matching.py")
    result = (utils, sync)
    _SOURCE_MODULE_CACHE[cache_key] = result
    return result


def official_c2m3_synchronized_states(states, source_root: Path, *, max_iter: int = 30):
    """Synchronize MLP states with C2M3's official Frank-Wolfe core."""

    import torch

    utils, sync = load_official_c2m3_core(source_root)
    spec = c2m3_mlp_permutation_spec()
    symbols = [chr(ord("a") + index) for index in range(len(states))]
    params = {symbol: copy.deepcopy(state) for symbol, state in zip(symbols, states)}
    combinations = [(left, right) for index, left in enumerate(symbols) for right in symbols[index + 1 :]]
    torch.manual_seed(0)
    permutations, optimization = sync.frank_wolfe_synchronized_matching(
        params=params,
        perm_spec=spec,
        symbols=symbols,
        combinations=combinations,
        max_iter=max_iter,
        initialization_method="identity",
        keep_soft_perms=False,
        device="cpu",
        verbose=False,
    )
    aligned = []
    applied = {}
    for symbol in symbols:
        per_symbol = {}
        for name, indices in permutations[symbol].items():
            matrix = utils.perm_indices_to_perm_matrix(indices).T
            per_symbol[name] = utils.perm_matrix_to_perm_indices(matrix)
        aligned.append(utils.apply_permutation_to_statedict(spec, per_symbol, params[symbol]))
        applied[symbol] = {name: value.detach().cpu().tolist() for name, value in per_symbol.items()}
    return aligned, applied, optimization
