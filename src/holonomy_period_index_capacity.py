"""Controlled finite-Heisenberg carrier layers for Application C."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProjectiveCase:
    name: str
    period: int
    index: int
    generator_pairs: tuple[tuple[str, str, complex], ...]


def shift(dimension: int) -> torch.Tensor:
    matrix = torch.zeros((dimension, dimension), dtype=torch.complex64)
    for column in range(dimension):
        matrix[(column + 1) % dimension, column] = 1.0
    return matrix


def clock(dimension: int) -> torch.Tensor:
    root = torch.exp(torch.tensor(2j * torch.pi / dimension, dtype=torch.complex64))
    return torch.diag(torch.stack([root**power for power in range(dimension)]))


def kron(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.kron(left.contiguous(), right.contiguous())


def base_generators(case_name: str) -> tuple[ProjectiveCase, dict[str, torch.Tensor]]:
    if case_name == "period2_index2":
        x = shift(2)
        z = clock(2)
        case = ProjectiveCase(case_name, 2, 2, (("x", "z", -1.0 + 0.0j),))
        return case, {"i": torch.eye(2, dtype=torch.complex64), "x": x, "z": z}
    if case_name == "period2_index4":
        identity = torch.eye(2, dtype=torch.complex64)
        x = shift(2)
        z = clock(2)
        generators = {
            "i": torch.eye(4, dtype=torch.complex64),
            "x1": kron(x, identity),
            "z1": kron(z, identity),
            "x2": kron(identity, x),
            "z2": kron(identity, z),
        }
        case = ProjectiveCase(
            case_name,
            2,
            4,
            (("x1", "z1", -1.0 + 0.0j), ("x2", "z2", -1.0 + 0.0j)),
        )
        return case, generators
    if case_name == "period3_index3":
        x = shift(3)
        z = clock(3)
        root = complex(torch.exp(torch.tensor(2j * torch.pi / 3)).item())
        case = ProjectiveCase(case_name, 3, 3, (("x", "z", root),))
        return case, {"i": torch.eye(3, dtype=torch.complex64), "x": x, "z": z}
    raise ValueError(f"unknown controlled projective case: {case_name}")


def expand_operator(operator: torch.Tensor, capacity: int) -> torch.Tensor:
    index = operator.shape[0]
    full_blocks, remainder = divmod(capacity, index)
    blocks = [operator.clone() for _ in range(full_blocks)]
    if remainder:
        blocks.append(operator[:remainder, :remainder].clone())
    if not blocks:
        raise ValueError("capacity must be positive")
    return torch.block_diag(*blocks)


def candidate_generators(case_name: str, capacity: int) -> tuple[ProjectiveCase, dict[str, torch.Tensor]]:
    case, generators = base_generators(case_name)
    return case, {name: expand_operator(value, capacity) for name, value in generators.items()}


def chart_operators(case_name: str, capacity: int) -> tuple[ProjectiveCase, list[torch.Tensor]]:
    case, generators = candidate_generators(case_name, capacity)
    if case_name == "period2_index2":
        x, z, identity = generators["x"], generators["z"], generators["i"]
        operators = [identity, x, z, x @ z, z @ x, x @ z @ x, z @ x @ z, x @ z @ x @ z]
    elif case_name == "period2_index4":
        identity = generators["i"]
        x1, z1 = generators["x1"], generators["z1"]
        x2, z2 = generators["x2"], generators["z2"]
        operators = [identity, x1, z1, x2, z2, x1 @ z1, x2 @ z2, x1 @ x2 @ z1 @ z2]
    else:
        identity = generators["i"]
        x, z = generators["x"], generators["z"]
        operators = [identity, x, z, x @ z, x @ x, z @ z, x @ x @ z, x @ z @ z]
    return case, operators


def relation_residual(case: ProjectiveCase, generators: dict[str, torch.Tensor]) -> float:
    residuals = []
    for left_name, right_name, root in case.generator_pairs:
        left = generators[left_name]
        right = generators[right_name]
        numerator = torch.linalg.norm(right @ left - root * left @ right)
        denominator = (torch.linalg.norm(left) * torch.linalg.norm(right)).clamp_min(1e-12)
        residuals.append(float(numerator / denominator))
    return max(residuals)


def unitarity_residual(operators: list[torch.Tensor]) -> float:
    residuals = []
    for operator in operators:
        identity = torch.eye(operator.shape[0], dtype=operator.dtype)
        residuals.append(
            float(torch.linalg.norm(operator.conj().T @ operator - identity) / torch.linalg.norm(identity))
        )
    return max(residuals)


def carrier_vectors(capacity: int, classes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(capacity, classes, generator=generator)
    imaginary = torch.randn(capacity, classes, generator=generator)
    values = torch.complex(real, imaginary)
    return values / torch.linalg.norm(values, dim=0, keepdim=True).clamp_min(1e-12)


def encode_logits(
    logits: torch.Tensor, operator: torch.Tensor, carriers: torch.Tensor
) -> torch.Tensor:
    transformed = operator @ carriers
    return logits.to(torch.complex64).unsqueeze(-1) * transformed.T.unsqueeze(0)


def decode_carrier(encoded: torch.Tensor, carriers: torch.Tensor) -> torch.Tensor:
    # encoded: item, class, capacity; carriers: capacity, class
    numerator = torch.einsum("rc,ncr->nc", carriers.conj(), encoded)
    denominator = (carriers.conj() * carriers).sum(dim=0).real.clamp_min(1e-12)
    return (numerator / denominator).real


def coherent_fusion(
    local_logits: torch.Tensor,
    operators: list[torch.Tensor],
    carriers: torch.Tensor,
    alignment_operators: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Encode actual local logits, align carrier branches, and decode."""

    alignments = operators if alignment_operators is None else alignment_operators
    aligned = []
    for chart in range(8):
        encoded = encode_logits(local_logits[chart], operators[chart], carriers)
        inverse = torch.linalg.pinv(alignments[chart])
        aligned.append(torch.einsum("rs,ncs->ncr", inverse, encoded))
    return decode_carrier(torch.stack(aligned).mean(0), carriers)


def ordinary_fusion(
    local_logits: torch.Tensor, operators: list[torch.Tensor], carriers: torch.Tensor
) -> torch.Tensor:
    encoded = [encode_logits(local_logits[chart], operators[chart], carriers) for chart in range(8)]
    return decode_carrier(torch.stack(encoded).mean(0), carriers)
