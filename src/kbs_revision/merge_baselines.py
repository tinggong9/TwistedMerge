"""Bounded matrix baselines for the KBS revision; no neural-performance claims.

The effective update is ``scale * B @ A`` in output-by-input orientation.
Explicit real weights are used verbatim, without normalization. Output factors
have unit scaling. These helpers assume evaluation mode, a common base/module,
and ordinary LoRA; they do not merge biases, heads, DoRA, or quantized bases.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass(frozen=True)
class LoRAUpdate:
    B: torch.Tensor
    A: torch.Tensor
    scale: float = 1.0

    @property
    def rank(self) -> int:
        return self.B.shape[1]


def _validate(updates: Sequence[LoRAUpdate], weights=None) -> list[float]:
    if not updates:
        raise ValueError("at least one update is required")
    first = updates[0]
    for update in updates:
        b, a = update.B, update.A
        if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError("incompatible B/A matrix shapes")
        if (b.shape[0], a.shape[1]) != (first.B.shape[0], first.A.shape[1]):
            raise ValueError("all updates must describe the same ambient layer")
        if b.dtype != a.dtype or b.device != a.device or b.dtype != first.B.dtype or b.device != first.B.device:
            raise ValueError("all factors must have the same dtype and device")
        if not (b.is_floating_point() or b.is_complex()):
            raise ValueError("factors must have floating or complex dtype")
        if not torch.isfinite(b).all() or not torch.isfinite(a).all() or not math.isfinite(update.scale):
            raise ValueError("factors and scaling must be finite")
    result = [1.0 / len(updates)] * len(updates) if weights is None else list(weights)
    if len(result) != len(updates) or any(isinstance(w, complex) or not math.isfinite(float(w)) for w in result):
        raise ValueError("one finite real weight is required per update")
    return [float(w) for w in result]


def effective_delta(update: LoRAUpdate) -> torch.Tensor:
    _validate([update])
    return update.scale * (update.B @ update.A)


def weighted_sum(updates: Sequence[LoRAUpdate], weights=None) -> torch.Tensor:
    """Dense Task Arithmetic reference, including adapter scaling."""
    values = _validate(updates, weights)
    result = updates[0].B.new_zeros((updates[0].B.shape[0], updates[0].A.shape[1]))
    for update, weight in zip(updates, values):
        result.add_(update.B @ update.A, alpha=weight * update.scale)
    return result


def concatenate_exact(updates: Sequence[LoRAUpdate], weights=None) -> LoRAUpdate:
    """Exact weighted sum with rank-sum capacity, including negative weights."""
    values = _validate(updates, weights)
    return LoRAUpdate(
        torch.cat([update.B for update in updates], dim=1),
        torch.cat([weight * update.scale * update.A for update, weight in zip(updates, values)], dim=0),
    )


def compact_qr_svd(updates: Sequence[LoRAUpdate], rank: int, weights=None) -> LoRAUpdate:
    """Truncate a weighted sum using thin QR and a possibly rectangular core.

    No ambient update is formed. When total input rank is below both ambient
    dimensions the core is strictly smaller than the ambient update. If total
    rank saturates both dimensions, the core itself has ambient shape: no
    factor-only memory advantage is claimed in that regime.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    joined = concatenate_exact(updates, weights)
    if rank == 0 or joined.rank == 0:
        return LoRAUpdate(joined.B[:, :0], joined.A[:0])
    q_left, t_left = torch.linalg.qr(joined.B, mode="reduced")
    q_right, t_right = torch.linalg.qr(joined.A.mH, mode="reduced")
    core = t_left @ t_right.mH
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    k = min(rank, singular.numel())
    root = singular[:k].sqrt()
    return LoRAUpdate((q_left @ u[:, :k]) * root, (root[:, None] * vh[:k]) @ q_right.mH)


def dense_reference_svd(updates: Sequence[LoRAUpdate], rank: int, weights=None) -> LoRAUpdate:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    u, singular, vh = torch.linalg.svd(weighted_sum(updates, weights), full_matrices=False)
    k = min(rank, singular.numel())
    root = singular[:k].sqrt()
    return LoRAUpdate(u[:, :k] * root, root[:, None] * vh[:k])


def naive_factor_average(updates: Sequence[LoRAUpdate], weights=None) -> LoRAUpdate:
    """Return (sum_i w_i B_i)(sum_j w_j s_j A_j), including cross terms.

    This is gauge-sensitive and generally differs from sum_i w_i s_i B_i A_i.
    Equal factor rank is required; no artificial padding of different ranks.
    """
    values = _validate(updates, weights)
    if len({update.rank for update in updates}) != 1:
        raise ValueError("factor averaging requires equal ranks")
    return LoRAUpdate(
        sum((weight * update.B for update, weight in zip(updates, values)), torch.zeros_like(updates[0].B)),
        sum((weight * update.scale * update.A for update, weight in zip(updates, values)), torch.zeros_like(updates[0].A)),
    )


def aligned_factor_average(updates: Sequence[LoRAUpdate], weights=None, *, method="global") -> LoRAUpdate:
    """Reuse the repository's real-CPU whitened pairwise/global alignment.

    Equal ranks and full-column-rank B are prerequisites. Scaling is absorbed
    into A before alignment. Singleton avoids the global solver's edge guard.
    This wrapper does not claim a weighted-sum identity for unrelated adapters.
    """
    values = _validate(updates, weights)
    if len({update.rank for update in updates}) != 1:
        raise ValueError("repository alignment requires equal ranks")
    if any(update.B.dtype not in {torch.float32, torch.float64} or update.B.device.type != "cpu" for update in updates):
        raise ValueError("repository alignment wrapper supports float32/float64 CPU factors only")
    if method not in {"pairwise", "global"}:
        raise ValueError("method must be pairwise or global")
    if len(updates) == 1:
        return naive_factor_average(updates, values)
    from src.lora_gauge_practical import globally_aligned_factors, reference_aligned_factors

    factors = [(update.B.detach().numpy(), (update.scale * update.A).detach().numpy()) for update in updates]
    function = globally_aligned_factors if method == "global" else reference_aligned_factors
    aligned, _, _ = function(factors)
    return naive_factor_average([LoRAUpdate(torch.from_numpy(b), torch.from_numpy(a)) for b, a in aligned], values)


_PEFT_CORE = None


def _peft_core():
    """Load the reviewed tensor-only installed PEFT module without model code."""
    global _PEFT_CORE
    if _PEFT_CORE is None:
        spec = importlib.util.find_spec("peft")
        if spec is None or spec.origin is None:
            raise RuntimeError("PEFT tensor merge kernels are unavailable")
        path = Path(spec.origin).parent / "utils" / "merge_utils.py"
        loader_spec = importlib.util.spec_from_file_location("kbs_peft_tensor_merge_core", path)
        if loader_spec is None or loader_spec.loader is None:
            raise RuntimeError("cannot load installed PEFT tensor merge kernels")
        _PEFT_CORE = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(_PEFT_CORE)
    return _PEFT_CORE


def peft_core_provenance() -> dict:
    core = _peft_core()
    path = Path(core.__file__)
    version = importlib.metadata.version("peft")
    return {
        "integration": "effective-update adapter around unmodified installed PEFT tensor kernels",
        "fidelity": "PEFT kernel parity; not the original TIES/MergeLM end-to-end implementation",
        "package": "peft",
        "version": version,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_url": f"https://github.com/huggingface/peft/blob/v{version}/src/peft/utils/merge_utils.py",
        "license": "Apache-2.0",
        "weight_convention": "default weights one; weighted sign-consistent sum divided by retained entry count; mass sign elected before weights; no weight normalization",
        "dare_convention": "random drop/rescale followed by sign election/disjoint mean, without a second magnitude trim (PEFT dare_ties)",
    }


def ties_effective(updates: Sequence[LoRAUpdate], weights=None, *, density=0.5) -> torch.Tensor:
    """PEFT TIES kernel on scaled dense effective updates, never A/B separately."""
    values = _validate(updates, [1.0] * len(updates) if weights is None else weights)
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if updates[0].B.is_complex():
        raise ValueError("TIES sign election requires real updates")
    return _peft_core().ties([effective_delta(u) for u in updates], updates[0].B.new_tensor(values), density, "total")


def dare_ties_effective(updates: Sequence[LoRAUpdate], weights=None, *, density=0.5, seed=0) -> torch.Tensor:
    """Seeded PEFT DARE/TIES effective-update CPU adaptation, preserving RNG."""
    values = _validate(updates, [1.0] * len(updates) if weights is None else weights)
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    if updates[0].B.is_complex() or updates[0].B.device.type != "cpu":
        raise ValueError("DARE/TIES reference wrapper supports real CPU updates only")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _peft_core().dare_ties([effective_delta(u) for u in updates], updates[0].B.new_tensor(values), density, "total")


class ShapeTrace(TorchDispatchMode):
    """Observe actual tensor outputs of every dispatched Torch operation."""

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))

        def visit(value):
            if isinstance(value, torch.Tensor):
                self.records.append({"operator": str(func), "shape": list(value.shape), "bytes": value.numel() * value.element_size()})
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        visit(result)
        return result


def _fixture(seed=20261014, out=17, inp=23, ranks=(2, 3, 1), *, complex_dtype=False):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.complex128 if complex_dtype else torch.float64
    return [LoRAUpdate(torch.randn(out, r, generator=generator, dtype=dtype), torch.randn(r, inp, generator=generator, dtype=dtype), scale) for r, scale in zip(ranks, (0.5, 2.0, -1.5, 1.0))]


def _relative(actual, expected):
    return float(torch.linalg.vector_norm(actual - expected) / max(float(torch.linalg.vector_norm(expected)), 1e-300))


def profile_factor_only(out=97, inp=79, *, ranks=(3, 4, 2), output_rank=5) -> dict:
    """Actual dispatch shapes plus CPU allocation/shape profiler evidence."""
    updates = _fixture(out=out, inp=inp, ranks=ranks)
    trace = ShapeTrace()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as profiler:
        with trace:
            compact_qr_svd(updates, output_rank, [0.2, -0.4, 0.9])
    ambient = [out, inp]
    ambient_outputs = [row for row in trace.records if row["shape"] == ambient]
    kernels = [{"operator": event.name, "input_shapes": event.input_shapes, "cpu_memory_usage_bytes": event.cpu_memory_usage, "self_cpu_memory_usage_bytes": event.self_cpu_memory_usage, "cpu_time_us": event.cpu_time_total} for event in profiler.events() if event.name in {"aten::mm", "aten::linalg_qr", "aten::_linalg_svd"}]
    return {
        "status": "controlled_algebraic",
        "device": "cpu",
        "ambient_shape": ambient,
        "total_input_rank": sum(ranks),
        "output_rank": output_rank,
        "scope": "total input rank strictly smaller than both ambient dimensions; saturated cores have no ambient-memory guarantee",
        "observed_tensor_outputs": trace.records,
        "ambient_shape_output_count": len(ambient_outputs),
        "profiler_kernels": kernels,
        "passed": not ambient_outputs and any(row["operator"] == "aten::mm" for row in kernels),
        "limitation": "Torch-dispatched tensor outputs are instrumented; opaque LAPACK workspace is not a Torch tensor shape",
    }


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _timing_worker(method, out=128, inp=96):
    torch.set_num_threads(2)
    start = time.perf_counter()
    baseline = _rss_bytes()
    updates = _fixture(out=out, inp=inp, ranks=(4, 4, 4))
    fixture_seconds = time.perf_counter() - start
    def merge():
        if method == "compact_qr_svd":
            return compact_qr_svd(updates, 4)
        if method == "dense_reference_svd":
            return dense_reference_svd(updates, 4)
        if method == "exact_rank_sum":
            return concatenate_exact(updates)
        if method == "naive_factor_average":
            return naive_factor_average(updates)
        if method in {"pairwise", "global"}:
            return aligned_factor_average(updates, method=method)
        if method == "ties":
            return ties_effective(updates)
        if method == "dare_ties":
            return dare_ties_effective(updates)
        raise ValueError(method)

    warmup_start = time.perf_counter()
    warmup = merge()
    warmup_seconds = time.perf_counter() - warmup_start
    del warmup
    merge_start = time.perf_counter()
    result = merge()
    merge_seconds = time.perf_counter() - merge_start
    if isinstance(result, LoRAUpdate):
        stored = result.B.numel() + result.A.numel()
        output_rank = result.rank
    else:
        stored, output_rank = result.numel(), None
    return {"method": method, "fixture_seconds": fixture_seconds, "warmup_seconds": warmup_seconds, "merge_seconds": merge_seconds, "worker_body_seconds": time.perf_counter() - start, "baseline_peak_rss_bytes": baseline, "peak_process_rss_bytes": _rss_bytes(), "incremental_highwater_rss_bytes": max(0, _rss_bytes() - baseline), "output_parameters": stored, "output_rank": output_rank, "device": "cpu", "dtype": "float64", "threads": 2, "ambient_shape": [out, inp], "ranks": [4, 4, 4]}


def isolated_cpu_timings(repeats=3, seed=719, *, architecture_shapes=False) -> dict:
    methods = ["compact_qr_svd", "exact_rank_sum", "naive_factor_average", "global"] if architecture_shapes else ["compact_qr_svd", "dense_reference_svd", "exact_rank_sum", "naive_factor_average", "pairwise", "global", "ties", "dare_ties"]
    shapes = [(1536, 1536), (8960, 1536)] if architecture_shapes else [(128, 96)]
    schedule = [(repeat, method, shape) for repeat in range(repeats) for method in methods for shape in shapes]
    random.Random(seed).shuffle(schedule)
    rows = []
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for order, (repeat, method, shape) in enumerate(schedule):
        start = time.perf_counter()
        command = [sys.executable, "-m", "src.kbs_revision.merge_baselines", "--timing-worker", method, str(shape[0]), str(shape[1])]
        try:
            process = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired as exc:
            rows.append({"method": method, "status": "failed", "error": "worker_timeout_120_seconds", "stderr": str(exc.stderr), "ambient_shape": list(shape), "repeat": repeat, "randomized_order": order, "full_process_wall_seconds": time.perf_counter() - start, "command": command})
            continue
        elapsed = time.perf_counter() - start
        if process.returncode:
            row = {"method": method, "status": "failed", "stderr": process.stderr, "returncode": process.returncode}
        else:
            row = {**json.loads(process.stdout), "status": "controlled_algebraic"}
        rows.append({**row, "ambient_shape": list(shape), "repeat": repeat, "randomized_order": order, "full_process_wall_seconds": elapsed, "command": command})
    result = {"status": "controlled_algebraic", "repeats": repeats, "seed": seed, "rows": rows, "scope": "CPU random-factor algebra systems smoke; includes cold runtime/import/process overhead; no GPU, model-loading, inference, or modern-model speedup claim", "rss_definition": "process ru_maxrss; incremental highwater difference from post-Torch-import baseline; includes warmup; may be zero because prior runtime allocation set the highwater", "timing_definition": "one warmup then one measured merge per fresh process; merge_seconds includes validation, update materialization where required, whitening, alignment, synchronization, compression; warmup includes method imports; fixture, warmup, worker-body, full process costs reported separately; TIES/DARE return dense updates without rank compression"}
    if architecture_shapes:
        result["shape_source"] = {"model": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323", "shapes": {"square_attention_output": [1536, 1536], "mlp_up_projection": [8960, 1536]}, "trained_factors": False, "dense_reference": "only small correctness/timing fixtures; not allocated for architecture-shaped panel"}
    return result


def architecture_shaped_cpu_timings() -> dict:
    result = isolated_cpu_timings(architecture_shapes=True)
    result["allocation_profiles"] = [profile_factor_only(1536, 1536, ranks=(4, 4, 4), output_rank=4), profile_factor_only(8960, 1536, ranks=(4, 4, 4), output_rank=4)]
    return result


def verify_all(*, include_timings=True) -> dict:
    """Deterministic measured CPU correctness, stress, parity, and cost evidence."""
    torch.set_num_threads(1)
    checks = []

    def record(name, observed, *, threshold=1e-10, detail=None):
        checks.append({"name": name, "observed_relative_error": observed, "threshold": threshold, "passed": math.isfinite(observed) and observed <= threshold, "detail": detail})

    cases = [
        ("heterogeneous_negative_scaled", _fixture(), [0.4, -1.2, 0.0]),
        ("rectangular_complex", _fixture(complex_dtype=True), [0.0, -0.7, 1.1]),
        ("rank_sum_exceeds_output", _fixture(out=3, inp=11, ranks=(3, 4, 2)), [0.2, 0.3, 0.5]),
        ("rank_sum_exceeds_both", _fixture(out=3, inp=4, ranks=(3, 4, 2)), [0.2, 0.3, 0.5]),
    ]
    for name, updates, weights in cases:
        dense = weighted_sum(updates, weights)
        record(name + "/exact_concatenation", _relative(effective_delta(concatenate_exact(updates, weights)), dense))
        full = compact_qr_svd(updates, min(dense.shape), weights)
        record(name + "/full_reconstruction", _relative(effective_delta(full), dense))
        small = compact_qr_svd(updates, 2, weights)
        reference = dense_reference_svd(updates, 2, weights)
        record(name + "/separated_rank2", _relative(effective_delta(small), effective_delta(reference)))
        record(name + "/permutation", _relative(effective_delta(compact_qr_svd(updates[::-1], 2, weights[::-1])), effective_delta(small)))

    updates = _fixture(ranks=(3, 3, 3))
    weights = [0.3, -0.6, 1.2]
    reference = effective_delta(compact_qr_svd(updates, 4, weights))
    generator = torch.Generator().manual_seed(809)
    for family, condition in [("orthogonal", 1), ("positive_diagonal", 8), ("dense", 30)]:
        changed = []
        for update in updates:
            q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
            if family == "positive_diagonal":
                q = torch.diag(torch.tensor([1.0, math.sqrt(condition), condition], dtype=torch.float64))
            elif family == "dense":
                v, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
                q = q @ torch.diag(torch.tensor([1.0, math.sqrt(condition), condition], dtype=torch.float64)) @ v.mH
            transformed = LoRAUpdate(update.B @ q, torch.linalg.solve(q, update.A), update.scale)
            record("gauge/" + family + "/individual", _relative(effective_delta(transformed), effective_delta(update)))
            changed.append(transformed)
        record("gauge/" + family + "/weighted_full", _relative(weighted_sum(changed, weights), weighted_sum(updates, weights)))
        record("gauge/" + family + "/compressed", _relative(effective_delta(compact_qr_svd(changed, 4, weights)), reference))

    rank_deficient = [LoRAUpdate(torch.cat([u.B[:, :1], u.B[:, :1]], 1), torch.cat([u.A[:1], 2 * u.A[:1]], 0), u.scale) for u in updates]
    record("rank_deficient", _relative(effective_delta(compact_qr_svd(rank_deficient, 7)), weighted_sum(rank_deficient)))
    zero = LoRAUpdate(torch.zeros(7, 3, dtype=torch.float64), torch.ones(3, 11, dtype=torch.float64))
    record("all_zero", float(effective_delta(compact_qr_svd([zero], 3)).abs().max()))
    record("zero_weights", float(effective_delta(compact_qr_svd(updates, 3, [0, 0, 0])).abs().max()))
    for name, method in [("compact", lambda u: compact_qr_svd([u], u.rank)), ("naive", lambda u: naive_factor_average([u])), ("pairwise", lambda u: aligned_factor_average([u], method="pairwise")), ("global", lambda u: aligned_factor_average([u]))]:
        record("singleton/" + name, _relative(effective_delta(method(updates[0])), effective_delta(updates[0])))
    diagonal = torch.diag(torch.tensor([5.0, 3.0, 3.0, 1.0], dtype=torch.float64))
    repeated = [LoRAUpdate(torch.eye(4, dtype=torch.float64), diagonal)]
    truncated = effective_delta(compact_qr_svd(repeated, 2))
    optimal_error = float(torch.linalg.vector_norm(torch.tensor([3.0, 1.0], dtype=torch.float64)))
    record("repeated_cutoff/optimal_error", abs(float(torch.linalg.vector_norm(truncated - diagonal)) - optimal_error) / optimal_error, detail="Nonunique rank-2 matrix; compare optimal error, no factor equality requirement")
    record("repeated_cutoff/separated_top_subspace", float(torch.linalg.vector_norm(truncated[:, 0] - diagonal[:, 0])) / 5)

    # Shared full-rank column space is the exact identity fixture used by the
    # repository's prior tests. This does not prove general weighted-sum equality.
    shared = [LoRAUpdate(updates[0].B.clone(), u.A.clone(), u.scale) for u in updates]
    shared_changed = []
    for update in shared:
        q = torch.randn(3, 3, generator=generator, dtype=torch.float64) + 3 * torch.eye(3, dtype=torch.float64)
        shared_changed.append(LoRAUpdate(update.B @ q, torch.linalg.solve(q, update.A), update.scale))
    for method in ("pairwise", "global"):
        result = aligned_factor_average(shared, method=method)
        changed_result = aligned_factor_average(shared_changed, method=method)
        record("repository_alignment/" + method + "/exact_shared_space", _relative(effective_delta(result), weighted_sum(shared)))
        record("repository_alignment/" + method + "/gauge", _relative(effective_delta(changed_result), effective_delta(result)))

    stress = []
    for condition in (1e4, 1e8, 1e12, 1e16):
        changed = []
        individual = []
        for update in updates:
            u, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
            v, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
            q = u @ torch.diag(torch.tensor([1.0, math.sqrt(condition), condition], dtype=torch.float64)) @ v.mH
            try:
                changed_update = LoRAUpdate(update.B @ q, torch.linalg.solve(q, update.A), update.scale)
                individual.append(_relative(effective_delta(changed_update), effective_delta(update)))
                changed.append(changed_update)
            except torch.linalg.LinAlgError as exc:
                stress.append({"condition_requested": condition, "status": "failed", "error": str(exc)})
                break
        else:
            error = _relative(effective_delta(compact_qr_svd(changed, 4, weights)), reference)
            stress.append({"condition_requested": condition, "individual_update_relative_errors": individual, "merge_relative_error": error, "threshold": 1e-10, "passed": error <= 1e-10, "status": "controlled_algebraic", "panel": "numerical_stress_not_primary"})

    parity = {"repository_alignment": "reused src/lora_gauge_practical.py; equal-rank real CPU full-column-rank B; tests on shared-column-space identity only", "compact": "standard weighted-sum baseline, not Core Space Merging or KnOTS"}
    try:
        core = _peft_core()
        tensor = torch.tensor([[[4.0, -3.0, 2.0, -1.0]], [[1.0, 5.0, -2.0, -6.0]]], dtype=torch.float64)
        fixture = [LoRAUpdate(torch.ones(1, 1, dtype=torch.float64), value) for value in tensor]
        actual = ties_effective(fixture, density=0.5)
        expected = torch.tensor([[4.0, 5.0, 0.0, -6.0]], dtype=torch.float64)
        record("ties/manual_trim_elect_disjoint_fixture", _relative(actual, expected))
        record("ties/peft_kernel", _relative(actual, core.ties(list(tensor), torch.ones(2, dtype=torch.float64), 0.5, "total")))
        seed = 433
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expected_dare = core.dare_ties(list(tensor), torch.ones(2, dtype=torch.float64), 0.7, "total")
        record("dare_ties/peft_kernel", _relative(dare_ties_effective(fixture, density=0.7, seed=seed), expected_dare))
        parity["ties_dare_ties"] = peft_core_provenance()
    except (RuntimeError, importlib.metadata.PackageNotFoundError) as exc:
        parity["ties_dare_ties"] = {"status": "blocked", "reason": str(exc)}
    profile = profile_factor_only()
    result = {"status": "controlled_algebraic", "dtype": "float64/complex128", "seed": 20261014, "checks": checks, "passed": all(row["passed"] for row in checks) and profile["passed"], "stress_panel": stress, "allocation_profile": profile, "baseline_parity": parity, "interpretation": "small exact matrix checks, no trained-model evaluation or new neural evidence"}
    if include_timings:
        result["cpu_timings"] = isolated_cpu_timings()
    return result


if __name__ == "__main__":
    if len(sys.argv) in (3, 5) and sys.argv[1] == "--timing-worker":
        dimensions = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) == 5 else (128, 96)
        print(json.dumps(_timing_worker(sys.argv[2], *dimensions), allow_nan=False))
    else:
        print(json.dumps(verify_all(), indent=2, allow_nan=False))
