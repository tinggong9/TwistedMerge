"""Preflight and cooperative CPU/accelerator/disk ceilings, without downloads."""
from __future__ import annotations

from pathlib import Path
import time

import psutil
import torch


def bytes_under(path):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def preflight(config, root):
    errors = []
    limits = config.get("resource_limits", {})
    for key in ("cpu_seconds", "disk_bytes"):
        if not isinstance(limits.get(key), (int, float)) or limits[key] <= 0:
            errors.append(f"missing positive resource ceiling: {key}")
    device = torch.device(config.get("device", "cpu"))
    if config.get("phase") == "software_only" and device.type != "cpu":
        errors.append("software fixture must use CPU")
    if device.type == "cuda" and not torch.cuda.is_available():
        errors.append("requested CUDA device is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        errors.append("requested MPS device is unavailable")
    if device.type != "cpu" and not limits.get("pilot_accelerator_hours", 0) > 0:
        errors.append("no accelerator-hour allocation")
    model_path = config.get("model", {}).get("local_path")
    if model_path and (Path(root) / model_path).is_dir():
        weight_bytes = sum(p.stat().st_size for p in (Path(root) / model_path).glob("*.safetensors"))
        # Conservative admission estimate, not a measured memory result.
        estimated = int(weight_bytes * (2 if config.get("dtype", "float32") == "float32" else 1) * 1.5 + 256 * 1024**2)
        available = psutil.virtual_memory().available
        if device.type == "cuda" and torch.cuda.is_available():
            available = torch.cuda.mem_get_info(device)[0]
        if available < estimated:
            errors.append(f"insufficient available memory for conservative loading estimate: available={available}, estimated={estimated}")
    return errors


class BudgetGuard:
    """Checked between operations; individual native kernels cannot be preempted."""
    def __init__(self, configuration, out):
        self.limits = configuration["resource_limits"]
        self.out = Path(out)
        self.start_bytes = bytes_under(out)
        self.start = time.monotonic()
        self.process = psutil.Process()
        times = self.process.cpu_times()
        self.start_cpu = times.user + times.system
        self.device = torch.device(configuration.get("device", "cpu"))

    def check(self):
        times = self.process.cpu_times()
        cpu = times.user + times.system - self.start_cpu
        wall = time.monotonic() - self.start
        disk = bytes_under(self.out) - self.start_bytes
        if cpu > self.limits["cpu_seconds"]:
            raise TimeoutError("CPU execution ceiling exceeded")
        if disk > self.limits["disk_bytes"]:
            raise OSError("new output disk ceiling exceeded")
        accelerator_hours = wall / 3600 if self.device.type != "cpu" else 0
        if accelerator_hours > self.limits.get("pilot_accelerator_hours", 0):
            raise TimeoutError("accelerator-hour ceiling exceeded")
        return {"cpu_seconds": cpu, "wall_seconds": wall, "new_output_bytes": disk,
                "accelerator_hours": accelerator_hours, "accelerator_count_used": int(self.device.type != "cpu"),
                "limitation": "cooperative checks between operations; includes this invocation only, not unobserved prior process work"}
