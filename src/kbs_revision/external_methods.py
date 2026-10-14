"""Reviewed, bounded KnOTS/Core Space tensor adaptations and independent parity.

This does not import or run either external repository. Exact, reviewed tensor
function excerpts in the checked-in JSON are executed only by the small-fixture
reference harness, after checking a hard-coded SHA256 of that entire manifest.
No upstream model loader, network operation or pickle loader is executed.

Inputs use the existing ``scale * B @ A`` convention. Official source settings
use alpha/r=1. Nonunit scales are handled explicitly here. Default task weights
are ones, with a separate common ``scaling`` multiplier. Upstream kernels use
``weights[0]`` for every task: heterogeneous weights therefore require opting
into the named ``corrected_per_task`` adaptation and cannot claim official
heterogeneous-weight parity. Nonlinear masks/signs are elected before weights.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
from contextlib import redirect_stdout
from types import MethodType, SimpleNamespace
from typing import Mapping, Sequence

import torch

from .merge_baselines import LoRAUpdate, _validate, dense_reference_svd, effective_delta


_MANIFEST = Path(__file__).resolve().parents[2] / "configs/kbs_revision/external_parity_sources.json"
_REFERENCE_MANIFEST_SHA256 = "cafa1ee44cae52763d999268d86891a41f9e9585bc7eef973b9b385e29bfe234"


def _sources() -> dict:
    raw = _MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _REFERENCE_MANIFEST_SHA256:
        raise RuntimeError("reviewed external reference manifest SHA256 mismatch")
    return json.loads(raw)


def external_method_definitions() -> dict:
    """Return source pins, scope and exact conventions without executable text."""
    sources = _sources()
    return {"manifest_sha256": _REFERENCE_MANIFEST_SHA256,
            "methods": [{**{k: v for k, v in item.items() if k not in {"reviewed_kernels", "license_text"}},
                         "source_files": [{k: v for k, v in kernel.items() if k != "source"}
                                          for kernel in item["reviewed_kernels"]]}
                        for item in sources["methods"]]}


def _options(updates, weights, density, merge_method, scaling, weight_semantics):
    values = _validate(updates, [1.] * len(updates) if weights is None else weights)
    if any(u.B.dtype not in (torch.float32, torch.float64) for u in updates):
        raise ValueError("external methods require real float32/float64 factors")
    if not math.isfinite(float(density)) or not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if merge_method not in {"ties", "sum", "mean"}:
        raise ValueError("merge_method must be ties, sum or mean")
    if not math.isfinite(float(scaling)):
        raise ValueError("scaling must be finite")
    if weight_semantics not in {"official_equal", "corrected_per_task"}:
        raise ValueError("unknown weight_semantics")
    if weight_semantics == "official_equal" and len(set(values)) != 1:
        raise ValueError("heterogeneous weights require corrected_per_task adaptation")
    if updates[0].B.device.type == "mps":
        raise ValueError("reviewed external SVD adaptations need float64; MPS is unsupported")
    return values


def _mask_and_merge(coordinates, weights, density, method):
    """Independent vectorized rendition of upstream threshold/sign/mean rules.

    Upstream uses inclusive kthvalue threshold, so ties and floor+one behavior
    can retain more than the requested density. Zero elected signs use the
    majority of elected coordinate signs, then source's >0/else<0 predicate.
    """
    vectors = torch.stack([tensor.reshape(-1) for tensor in coordinates])
    if vectors.shape[1] == 0:
        return vectors.new_empty((0,))
    if method == "ties":
        if density < 1:
            d = vectors.shape[1]
            threshold = vectors.abs().kthvalue(d - int(d * density), dim=1, keepdim=True).values
            vectors = vectors * (vectors.abs() >= threshold)
        signs = vectors.sum(0).sign()
        signs = torch.where(signs == 0, signs.sum().sign(), signs)
        selected = vectors * torch.where(signs[None] > 0, vectors > 0, vectors < 0)
    else:
        selected = vectors
    weighted = selected * vectors.new_tensor(weights)[:, None]
    if method == "sum":
        return weighted.sum(0)
    if method == "mean":
        return weighted.mean(0)
    return weighted.sum(0) / (weighted != 0).sum(0).clamp(min=1)


def _rank_guard(rank):
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
        raise ValueError("rank must be None or a nonnegative integer")


def knots_merge_layers(layers: Mapping[str, Sequence[LoRAUpdate]], rank=None, weights=None, *,
                      density=1.0, merge_method="ties", scaling=1.0,
                      weight_semantics="official_equal") -> dict[str, LoRAUpdate]:
    """KnOTS aligned sV merge with globally flattened multilayer TIES.

    Each layer's horizontally concatenated scaled dense updates gets SVD in
    float64, absolute singular cutoff >1e-5, then U/s/V are float32 as upstream.
    The shared U is reconstructed with globally TIES-merged sV blocks. Heads,
    biases, DARE and alternative concatenation orientation are outside scope.
    ``rank`` adds dense output compression to the official uncompressed method.
    """
    _rank_guard(rank)
    if not layers:
        raise ValueError("at least one layer is required")
    names = sorted(layers)
    bases, coordinates, values = {}, {}, None
    count = len(layers[names[0]])
    first = layers[names[0]][0] if count else None
    for name in names:
        updates = layers[name]
        current = _options(updates, weights, density, merge_method, scaling, weight_semantics)
        if len(updates) != count:
            raise ValueError("every layer must have the same task count/order")
        if updates[0].B.device != first.B.device:
            raise ValueError("all KnOTS layers must share a device")
        values = current
        stack = torch.cat([effective_delta(update) for update in updates], dim=1)
        u, s, vh = torch.linalg.svd(stack.to(torch.float64), full_matrices=False)
        keep = s > 1e-5
        u, s, vh = u[:, keep].float(), s[keep].float(), vh[keep].float()
        bases[name] = u
        coordinates[name] = list((s[:, None] * vh).split(updates[0].A.shape[1], dim=1))
    per_task = [torch.cat([coordinates[name][task].reshape(-1) for name in names]) for task in range(count)]
    merged = _mask_and_merge(per_task, values, density, merge_method) * scaling
    outputs, offset = {}, 0
    for name in names:
        shape = coordinates[name][0].shape
        size = coordinates[name][0].numel()
        aligned = merged[offset:offset + size].reshape(shape)
        offset += size
        update = LoRAUpdate(bases[name], aligned)
        if rank is not None:
            update = dense_reference_svd([update], rank, [1.])
        outputs[name] = update
    return outputs


def knots_merge(updates: Sequence[LoRAUpdate], rank=None, weights=None, **kwargs) -> LoRAUpdate:
    """Single-layer convenience wrapper; use layers wrapper for model merging."""
    return knots_merge_layers({"layer": updates}, rank, weights, **kwargs)["layer"]


def core_space_merge(updates: Sequence[LoRAUpdate], rank=None, weights=None, *, density=1.0,
                     merge_method="ties", scaling=1.0,
                     weight_semantics="official_equal") -> LoRAUpdate:
    """Core Space per-layer merge, using SVD bases from concatenated factors.

    Bases span stacked unscaled A and B factors. The core for task i is
    scale_i (U_B.T B_i)(A_i V_A); nonlinear TIES is applied to these cores before
    reconstructing. This is distinct from SVD compression of a weighted sum.
    ``rank=None`` retains the left reference basis capacity; ``rank`` adds
    compact core-SVD output compression. Returned scaling is always one.
    """
    _rank_guard(rank)
    values = _options(updates, weights, density, merge_method, scaling, weight_semantics)
    dtype = updates[0].B.dtype
    a_stack = torch.cat([update.A for update in updates]).double()
    b_stack = torch.cat([update.B for update in updates], dim=1).double()
    right = torch.linalg.svd(a_stack, full_matrices=False).Vh
    left = torch.linalg.svd(b_stack, full_matrices=False).U
    cores = [(left.T @ u.B.double()) @ (u.A.double() @ right.T) * u.scale for u in updates]
    core = (_mask_and_merge(cores, values, density, merge_method) * scaling).reshape(left.shape[1], right.shape[0])
    if rank is None:
        return LoRAUpdate(left.to(dtype), (core @ right).to(dtype))
    u, s, vh = torch.linalg.svd(core, full_matrices=False)
    k = min(rank, s.numel())
    root = s[:k].sqrt()
    return LoRAUpdate(((left @ u[:, :k]) * root).to(dtype), ((root[:, None] * vh[:k]) @ right).to(dtype))


def _reference_namespace(method):
    """Execute only reviewed exact function excerpts after manifest SHA check."""
    entry = next(item for item in _sources()["methods"] if item["id"] == method)
    scope = {"torch": torch, "defaultdict": defaultdict, "deepcopy": deepcopy,
             "tqdm": lambda iterable, **_: iterable,
             "__builtins__": {"isinstance": isinstance, "list": list, "dict": dict, "len": len,
                              "range": range, "zip": zip, "print": print, "int": int,
                              "enumerate": enumerate, "ValueError": ValueError,
                              "min": min, "max": max, "__import__": __import__}}
    for kernel in entry["reviewed_kernels"]:
        source = kernel["source"]
        if hashlib.sha256(source.encode()).hexdigest() != kernel["excerpt_sha256"]:
            raise RuntimeError("reviewed tensor excerpt SHA256 mismatch")
        tree = ast.parse(source)
        if not all(isinstance(node, ast.FunctionDef) for node in tree.body):
            raise RuntimeError("reviewed reference must contain function definitions only")
        exec(compile(tree, kernel["source_url"], "exec"), scope)
    return scope


def official_knots_reference(layers, *, density=1., scaling=1.) -> dict[str, torch.Tensor]:
    """Independent exact upstream tensor pipeline, excluding model I/O.

    Adapter scale is absorbed in the input effective matrices; original
    alpha/r=1 fixture parity is checked separately. Only common weights work.
    """
    scope = _reference_namespace("knots")
    helper = SimpleNamespace(device="cpu")
    for name in ("variable_extend_dim", "dict_of_concat_matrices", "apply_svd"):
        setattr(helper, name, MethodType(scope[name], helper))
    names = sorted(layers)
    tasks = [{name: effective_delta(layers[name][task]) for name in names}
             for task in range(len(layers[names[0]]))]
    with redirect_stdout(io.StringIO()):
        bases, _, coords, _ = helper.apply_svd(tasks, concat_across_output=True)
        vectors = [torch.cat([item[name].reshape(-1) for name in names]) for item in coords]
        mask = scope["ties_masking"](vectors, topK=100 * density)
        merged = scope["masked_merge"](torch.stack(vectors) * mask, "mean", weights=torch.tensor([scaling]))
    outputs, offset = {}, 0
    for name in names:
        shape, size = coords[0][name].shape, coords[0][name].numel()
        outputs[name] = (bases[name] @ merged[offset:offset + size].reshape(shape)).float()
        offset += size
    return outputs


def official_core_reference(updates, *, density=1., scaling=1.) -> torch.Tensor:
    """Exact upstream four-SVD core construction and nonlinear TIES kernels."""
    scope = _reference_namespace("core_space")
    helper = SimpleNamespace(cache={})
    parameters = [{"layer": (u.A, u.B)} for u in updates]
    cores, left, right = scope["get_core_matrices"](helper, parameters, "layer", {})
    with redirect_stdout(io.StringIO()):
        merged, _, _ = scope["ties_merging"]([u.scale * core.reshape(-1) for u, core in zip(updates, cores)],
                                             topK=100 * density)
    return (left @ merged.reshape(left.shape[1], right.shape[0]) @ right) * scaling


def _conditioning_stress(updates):
    """Retain controlled failures; nonlinear gauge invariance is not assumed."""
    generator = torch.Generator().manual_seed(7765)
    out = []
    for condition in (1., 1e4, 1e8, 1e12):
        changed = []
        for update in updates:
            r = update.rank
            left, _ = torch.linalg.qr(torch.randn(r, r, generator=generator, dtype=torch.float64))
            right, _ = torch.linalg.qr(torch.randn(r, r, generator=generator, dtype=torch.float64))
            gauge = left @ torch.diag(torch.logspace(0, math.log10(condition), r, dtype=torch.float64)) @ right.T
            changed.append(LoRAUpdate(update.B @ gauge, torch.linalg.solve(gauge, update.A), update.scale))
        expected = sum(effective_delta(u) for u in changed)
        gauge_error = max(float(torch.linalg.vector_norm(effective_delta(u) - effective_delta(v)) /
                                torch.linalg.vector_norm(effective_delta(v))) for u, v in zip(changed, updates))
        for method, function, reference in (("knots", knots_merge, None),
                                             ("core_space", core_space_merge, official_core_reference)):
            try:
                linear = effective_delta(function(changed, merge_method="sum")).double()
                relative = float(torch.linalg.vector_norm(linear - expected) / torch.linalg.vector_norm(expected))
                row = {"method": method, "gauge_condition_number_planned": condition,
                       "gauge_preservation_relative_error": gauge_error,
                       "linear_roundtrip_relative_error": relative, "strict_tolerance": 1e-10,
                       "passed": relative <= 1e-10,
                       "status": "passed" if relative <= 1e-10 else "failed_strict_parity",
                       "working_precision": "float32_after_float64_svd" if method == "knots" else "float64",
                       "routine_tolerance": 2e-6 if method == "knots" else 1e-10,
                       "routine_passed": relative <= (2e-6 if method == "knots" else 1e-10),
                       "scope": "controlled CPU conditioning stress; no neural or nonlinear gauge-invariance claim"}
                if reference is not None:
                    actual, original = effective_delta(function(changed)).double(), reference(changed).double()
                    parity = float(torch.linalg.vector_norm(actual - original) / max(float(torch.linalg.vector_norm(original)), 1e-300))
                    row["nonlinear_official_kernel_relative_error"] = parity
                    row["nonlinear_official_kernel_strict_passed"] = parity <= 1e-10
                out.append(row)
            except (RuntimeError, ValueError) as exc:
                out.append({"method": method, "gauge_condition_number_planned": condition,
                            "status": "failed", "passed": False, "error": str(exc)})
    return out


def verify_external_parity() -> dict:
    """Offline independent small CPU fixture evidence; no neural metrics."""
    generator = torch.Generator().manual_seed(20261014)
    layers = {name: [LoRAUpdate(torch.randn(out, rank, generator=generator, dtype=torch.float64),
                               torch.randn(rank, inp, generator=generator, dtype=torch.float64))
                     for rank in (2, 2, 2)]
              for name, out, inp in (("layer0.q_proj", 13, 11), ("layer1.o_proj", 11, 17))}
    checks = []

    def compare(name, actual, expected, tolerance=2e-6, **details):
        error = float(torch.linalg.vector_norm(actual.double() - expected.double()) /
                      max(float(torch.linalg.vector_norm(expected.double())), 1e-300))
        checks.append({"name": name, "relative_error": error, "tolerance": tolerance,
                       "passed": error <= tolerance, **details})

    for density in (1., .5):
        reference = official_knots_reference(layers, density=density, scaling=.5)
        actual = knots_merge_layers(layers, density=density, scaling=.5)
        for name in layers:
            compare(f"knots_global_ties_density_{density}_{name}", effective_delta(actual[name]), reference[name],
                    output_factor_rank=actual[name].rank, official_tensor_kernel=True)
        for name, updates in layers.items():
            compare(f"core_ties_density_{density}_{name}", effective_delta(core_space_merge(updates, density=density, scaling=.5)),
                    official_core_reference(updates, density=density, scaling=.5), 1e-10,
                    official_tensor_kernel=True)
    updates = layers["layer0.q_proj"]
    weighted = [LoRAUpdate(u.B, u.A, scale) for u, scale in zip(updates, (.5, 2., -1.5))]
    weights = [.2, -.4, .9]
    dense = sum(w * effective_delta(u) for u, w in zip(weighted, weights))
    for name, function in (("knots", knots_merge), ("core_space", core_space_merge)):
        compare(f"{name}_corrected_weighted_sum", effective_delta(function(weighted, weights=weights, merge_method="sum", weight_semantics="corrected_per_task")),
                dense, 2e-6 if name == "knots" else 1e-10, official_tensor_kernel=False)
        full = function(updates, density=.5)
        compressed = function(updates, rank=2, density=.5)
        reference = dense_reference_svd([full], 2, [1.])
        compare(f"{name}_explicit_rank_compression", effective_delta(compressed), effective_delta(reference),
                2e-6 if name == "knots" else 1e-10, output_factor_rank=compressed.rank,
                official_tensor_kernel=False, capacity_adaptation=True)
    return {"status": "software_only_cpu_fixture", "passed": all(c["passed"] for c in checks),
            "checks": checks, "sources": external_method_definitions(),
            "stress_panel": _conditioning_stress(updates),
            "neural_evaluation": "not_run", "full_upstream_model_runner": "not_run",
            "limitation": "Independent parity against exact reviewed upstream tensor excerpts; no pretrained weights, training, task accuracy or full external repository runner execution."}
