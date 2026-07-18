"""Channel gauges for ResNet-style Conv/BatchNorm/ReLU networks.

Channel permutations are exact when every edge incident to a representation
uses the same basis and identity residual shortcuts keep the input and output
basis equal.  Positive channel scaling in front of BatchNorm needs more care:
with nonzero epsilon, scaling running statistics alone is not exact.  This
module exposes explicit strategies instead of silently calling them all exact.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ResNetChannelPermutations:
    """Representation bases for a torchvision-style two-conv BasicBlock net."""

    stem: tuple[int, ...]
    stages: Mapping[str, tuple[int, ...]]
    hidden: Mapping[str, tuple[int, ...]]


def _torch():
    import torch

    return torch


def _permutation(values: Sequence[int], width: int, label: str):
    torch = _torch()
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim != 1 or tensor.numel() != width:
        raise ValueError(f"{label} must have length {width}")
    if sorted(tensor.tolist()) != list(range(width)):
        raise ValueError(f"{label} is not a permutation of 0..{width - 1}")
    return tensor


def _channels(module) -> int:
    if hasattr(module, "out_channels"):
        return int(module.out_channels)
    if hasattr(module, "num_features"):
        return int(module.num_features)
    raise TypeError(f"cannot infer channels for {type(module).__name__}")


def _permute_conv_(conv, input_perm, output_perm) -> None:
    torch = _torch()
    if int(conv.groups) != 1:
        raise NotImplementedError("arbitrary channel permutations require groups=1")
    input_perm = _permutation(input_perm, int(conv.in_channels), "conv input permutation")
    output_perm = _permutation(output_perm, int(conv.out_channels), "conv output permutation")
    with torch.no_grad():
        weight = conv.weight.detach().clone()
        conv.weight.copy_(weight.index_select(0, output_perm).index_select(1, input_perm))
        if conv.bias is not None:
            conv.bias.copy_(conv.bias.detach().clone().index_select(0, output_perm))


def _permute_batchnorm_(batchnorm, permutation) -> None:
    torch = _torch()
    permutation = _permutation(permutation, int(batchnorm.num_features), "BatchNorm permutation")
    with torch.no_grad():
        if batchnorm.affine:
            batchnorm.weight.copy_(batchnorm.weight.detach().clone().index_select(0, permutation))
            batchnorm.bias.copy_(batchnorm.bias.detach().clone().index_select(0, permutation))
        if batchnorm.track_running_stats:
            batchnorm.running_mean.copy_(batchnorm.running_mean.detach().clone().index_select(0, permutation))
            batchnorm.running_var.copy_(batchnorm.running_var.detach().clone().index_select(0, permutation))
            # num_batches_tracked is a scalar count and is intentionally unchanged.


def _permute_linear_input_(linear, permutation) -> None:
    torch = _torch()
    permutation = _permutation(permutation, int(linear.in_features), "linear input permutation")
    with torch.no_grad():
        linear.weight.copy_(linear.weight.detach().clone().index_select(1, permutation))


def _projection_modules(downsample):
    torch = _torch()
    convs = [module for module in downsample.modules() if isinstance(module, torch.nn.Conv2d)]
    norms = [module for module in downsample.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if len(convs) != 1 or len(norms) != 1:
        raise NotImplementedError("shortcut projection must contain exactly one Conv2d and one BatchNorm")
    return convs[0], norms[0]


def random_resnet18_permutations(model, seed: int = 0) -> ResNetChannelPermutations:
    """Create compatible random bases for a two-conv ResNet.

    A stage may change basis only at a projection shortcut.  Identity blocks
    inherit their input basis.  Hidden branch bases remain independent.
    """

    rng = np.random.default_rng(seed)
    stem_width = _channels(model.conv1)
    stem = tuple(map(int, rng.permutation(stem_width)))
    stages: dict[str, tuple[int, ...]] = {}
    hidden: dict[str, tuple[int, ...]] = {}
    current = stem
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        if not hasattr(model, stage_name):
            continue
        blocks = getattr(model, stage_name)
        first = blocks[0]
        output_width = _channels(first.conv2)
        if first.downsample is None:
            if output_width != len(current):
                raise ValueError(f"identity stage {stage_name} changes width")
            stage_perm = current
        else:
            stage_perm = tuple(map(int, rng.permutation(output_width)))
        stages[stage_name] = stage_perm
        for index, block in enumerate(blocks):
            hidden[f"{stage_name}.{index}"] = tuple(map(int, rng.permutation(_channels(block.conv1))))
        current = stage_perm
    return ResNetChannelPermutations(stem=stem, stages=stages, hidden=hidden)


def permute_resnet_channels(model, permutations: ResNetChannelPermutations):
    """Return an exact channel-permuted copy of a torchvision-style ResNet-18.

    Supported residual blocks have ``conv1/bn1/conv2/bn2`` and an optional
    Conv+BatchNorm projection in ``downsample``.  Identity shortcuts reject an
    inconsistent requested output basis rather than silently breaking the sum.
    """

    transformed = copy.deepcopy(model)
    input_rgb = tuple(range(int(transformed.conv1.in_channels)))
    stem = tuple(permutations.stem)
    _permute_conv_(transformed.conv1, input_rgb, stem)
    _permute_batchnorm_(transformed.bn1, stem)
    current = stem
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        if not hasattr(transformed, stage_name):
            continue
        if stage_name not in permutations.stages:
            raise ValueError(f"missing stage permutation for {stage_name}")
        stage_perm = tuple(permutations.stages[stage_name])
        blocks = getattr(transformed, stage_name)
        for index, block in enumerate(blocks):
            block_name = f"{stage_name}.{index}"
            if block_name not in permutations.hidden:
                raise ValueError(f"missing hidden permutation for {block_name}")
            hidden = tuple(permutations.hidden[block_name])
            output = stage_perm
            if block.downsample is None and tuple(current) != tuple(output):
                raise ValueError(f"identity shortcut {block_name} requires identical input/output bases")
            _permute_conv_(block.conv1, current, hidden)
            _permute_batchnorm_(block.bn1, hidden)
            _permute_conv_(block.conv2, hidden, output)
            _permute_batchnorm_(block.bn2, output)
            if block.downsample is not None:
                projection, norm = _projection_modules(block.downsample)
                _permute_conv_(projection, current, output)
                _permute_batchnorm_(norm, output)
            current = output
    if not hasattr(transformed, "fc"):
        raise NotImplementedError("ResNet classifier must be named fc")
    _permute_linear_input_(transformed.fc, current)
    return transformed


def _resolve_module(model, dotted_name: str):
    module = model
    for part in dotted_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _positive_scales(values, width: int, label: str):
    torch = _torch()
    scales = torch.as_tensor(values, dtype=torch.float64)
    if scales.ndim != 1 or scales.numel() != width:
        raise ValueError(f"{label} must have length {width}")
    if not bool((scales > 0).all()):
        raise ValueError(f"{label} must be strictly positive")
    return scales


def _scale_conv_output_(conv, scales) -> None:
    torch = _torch()
    scales = scales.to(dtype=conv.weight.dtype, device=conv.weight.device)
    with torch.no_grad():
        conv.weight.mul_(scales[:, None, None, None])
        if conv.bias is not None:
            conv.bias.mul_(scales)


def scale_conv_batchnorm(
    model,
    conv_name: str,
    batchnorm_name: str,
    scales,
    *,
    strategy: str,
):
    """Scale Conv output channels and apply an explicit BatchNorm strategy.

    Strategies:

    ``frozen``
        Keep original running statistics and affine parameters. Not exact.
    ``running``
        Scale running mean/variance, keep affine parameters. Approximate for
        nonzero epsilon.
    ``affine``
        Keep original running statistics and transform affine parameters. This
        is exactly function-preserving in eval mode with frozen statistics,
        although those statistics no longer describe the scaled Conv output.
    ``running_affine``
        Scale running statistics and compensate epsilon in the affine weight.
        Exactly function-preserving in eval mode with frozen transformed stats.
    """

    torch = _torch()
    transformed = copy.deepcopy(model)
    conv = _resolve_module(transformed, conv_name)
    batchnorm = _resolve_module(transformed, batchnorm_name)
    if _channels(conv) != _channels(batchnorm):
        raise ValueError("Conv output and BatchNorm widths differ")
    scales64 = _positive_scales(scales, _channels(conv), "channel scales")
    if not batchnorm.affine or not batchnorm.track_running_stats:
        raise ValueError("scaling strategies require affine BatchNorm with running statistics")
    scales_t = scales64.to(dtype=batchnorm.running_var.dtype, device=batchnorm.running_var.device)
    old_mean = batchnorm.running_mean.detach().clone()
    old_var = batchnorm.running_var.detach().clone()
    old_weight = batchnorm.weight.detach().clone()
    old_bias = batchnorm.bias.detach().clone()
    sigma = torch.sqrt(old_var + float(batchnorm.eps))
    _scale_conv_output_(conv, scales_t)
    strategy = strategy.lower()
    with torch.no_grad():
        if strategy == "frozen":
            pass
        elif strategy == "running":
            batchnorm.running_mean.copy_(scales_t * old_mean)
            batchnorm.running_var.copy_(scales_t.square() * old_var)
        elif strategy == "affine":
            batchnorm.weight.copy_(old_weight / scales_t)
            batchnorm.bias.copy_(old_bias + old_weight * old_mean / sigma * (1.0 / scales_t - 1.0))
        elif strategy == "running_affine":
            scaled_var = scales_t.square() * old_var
            batchnorm.running_mean.copy_(scales_t * old_mean)
            batchnorm.running_var.copy_(scaled_var)
            correction = torch.sqrt(scaled_var + float(batchnorm.eps)) / (scales_t * sigma)
            batchnorm.weight.copy_(old_weight * correction)
            batchnorm.bias.copy_(old_bias)
        else:
            raise ValueError(f"unknown BatchNorm scaling strategy: {strategy}")
    return transformed


def scale_relu_conv_pair(model, first_conv_name: str, second_conv_name: str, scales):
    """Exact positive ReLU gauge for a no-BatchNorm Conv-ReLU-Conv pair."""

    torch = _torch()
    transformed = copy.deepcopy(model)
    first = _resolve_module(transformed, first_conv_name)
    second = _resolve_module(transformed, second_conv_name)
    if int(first.out_channels) != int(second.in_channels):
        raise ValueError("adjacent Conv channel widths differ")
    scales_t = _positive_scales(scales, int(first.out_channels), "channel scales").to(
        dtype=first.weight.dtype,
        device=first.weight.device,
    )
    _scale_conv_output_(first, scales_t)
    with torch.no_grad():
        second.weight.div_(scales_t[None, :, None, None].to(second.weight.device, second.weight.dtype))
    return transformed


def recompute_batchnorm_statistics(model, batches, *, device="cpu", reset: bool = True):
    """Recompute running statistics from input batches without gradient updates."""

    torch = _torch()
    transformed = copy.deepcopy(model).to(device)
    norms = [module for module in transformed.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if reset:
        for module in norms:
            module.reset_running_stats()
            module.momentum = None
    transformed.train()
    with torch.no_grad():
        for batch in batches:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            transformed(images.to(device))
    transformed.eval()
    return transformed


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
