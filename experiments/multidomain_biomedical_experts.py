#!/usr/bin/env python3
"""C1: independently trained experts under exact charts and synthetic domains."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.chart_followup_common import D4EquivariantChartCNN  # noqa: E402
from experiments.spatial_output_common import (  # noqa: E402
    DEVICE,
    OUT,
    D4SymmetrizedUNet,
    TinyUNet,
    apply_d4,
    average_state_dict,
    chart_probabilities,
    dataset_checksum,
    dataset_ready,
    expert_original_frame_logits,
    factual_report,
    hard_canonical_retransport,
    inverse_d4,
    load_checkpoint,
    paired_rows,
    predict_logits,
    predict_probability,
    record_command,
    role_split,
    save_checkpoint,
    save_predictions_before_metrics,
    segmentation_metrics,
    soft_canonical_retransport,
    stage_complete,
    train_segmenter,
    update_status,
    utc_now,
    write_csv,
)

SCRIPT = Path(__file__).resolve()
DEST = OUT / "multidomain"
COMMAND = "python experiments/multidomain_biomedical_experts.py"
CONDITIONS = ("seen_domain_seen_chart", "seen_domain_unseen_chart", "unseen_domain_seen_chart", "unseen_domain_unseen_chart", "missing_local_expert", "absent_domain_chart_combination")
METHODS = ("weight_average", "greedy_soup", "generic_moe", "domain_router", "chart_router", "joint_domain_chart_router", "direct_d4_equivariant_unet", "d4_test_time_augmentation", "one_canonical_after_chart_inference", "domain_specific_canonical_experts", "chart_aware_multi_expert_pooling", "canonicalize_pool_retransport", "supplied_domain_chart_oracle", "ensemble")


def synthetic_domain(images: torch.Tensor, domain: int) -> torch.Tensor:
    """Non-group color/stain shifts, explicitly not chart actions."""

    if domain == 0:
        return images
    if domain == 1:
        scale = torch.tensor([1.18, 0.82, 0.92])[None, :, None, None]
        return (images * scale).clamp(0, 1)
    if domain == 2:
        scale = torch.tensor([0.82, 1.15, 1.08])[None, :, None, None]
        return (images * scale).clamp(0, 1)
    if domain == 3:
        matrix = torch.tensor([[0.86, 0.09, 0.05], [0.08, 0.82, 0.10], [0.04, 0.12, 0.84]])
        return torch.einsum("ij,njhw->nihw", matrix, images).clamp(0, 1)
    if domain == 4:
        return (0.55 * synthetic_domain(images, 1) + 0.45 * synthetic_domain(images, 2)).clamp(0, 1)
    raise ValueError(domain)


def domain_probabilities(images: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    features = images.mean((-2, -1))
    distances = ((features[:, None] - prototypes[None]) ** 2).mean(-1)
    return (-40.0 * distances).softmax(1)


def _load_chart_and_direct(seed: int) -> tuple[Any, Any, Any]:
    chart = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_chart_equivariant.pt", D4EquivariantChartCNN(3, width=4))[0]
    direct_base = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_direct_base.pt", TinyUNet(width=4))[0]
    canonical = load_checkpoint(OUT / "checkpoints" / f"seed_{seed}_canonical_0.pt", TinyUNet(width=4))[0]
    return chart, D4SymmetrizedUNet(direct_base).to(DEVICE), canonical


def _train_domains(seed: int, payload: dict[str, Any], smoke: bool) -> tuple[list[TinyUNet], torch.Tensor, list[dict[str, Any]]]:
    models, prototypes, checkpoints = [], [], []
    for domain in range(4):
        indices = torch.arange(domain, len(payload["expert_images"]), 4)
        images = synthetic_domain(payload["expert_images"][indices], domain)
        validation = synthetic_domain(payload["early_images"], domain)
        model, elapsed, history = train_segmenter(TinyUNet(width=4), images, payload["expert_masks"][indices], validation, payload["early_masks"], 360_000_000 + seed * 10 + domain, 1 if smoke else 2)
        models.append(model)
        prototypes.append(images.mean((0, 2, 3)))
        checkpoints.append({"seed": seed, "expert": domain, "domain_type": "synthetic_color_or_stain", **save_checkpoint(OUT / "checkpoints" / f"multidomain_seed_{seed}_expert_{domain}.pt", model, {"seed": seed, "synthetic_domain": domain, "disjoint_indices_modulo_four": domain, "training_time": elapsed, "history": history})})
    return models, torch.stack(prototypes), checkpoints


def _condition(payload: dict[str, Any], condition: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = min(4, len(payload["test_images"]))
    images, masks = payload["test_images"][:count], payload["test_masks"][:count]
    seen_domains = condition in ("seen_domain_seen_chart", "seen_domain_unseen_chart", "missing_local_expert", "absent_domain_chart_combination")
    true_domains = torch.tensor([index % 4 for index in range(count)], dtype=torch.long) if seen_domains else torch.full((count,), 4, dtype=torch.long)
    transformed = torch.cat([synthetic_domain(images[index:index+1], int(true_domains[index])) for index in range(count)])
    seen_chart = condition in ("seen_domain_seen_chart", "unseen_domain_seen_chart")
    charts = torch.zeros(count, dtype=torch.long) if seen_chart else torch.tensor([(index * 2 + 3) % 8 for index in range(count)], dtype=torch.long)
    return apply_d4(transformed, charts), apply_d4(masks, charts), true_domains, charts


def _domain_chart_soft(images: torch.Tensor, experts: list[TinyUNet], domain_probs: torch.Tensor, chart_probs: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((len(images), 1, images.shape[-2], images.shape[-1]))
    for chart in range(8):
        canonical = inverse_d4(images, chart)
        expert_logits = expert_original_frame_logits(canonical, experts)
        pooled = (domain_probs[:, :, None, None, None] * expert_logits).sum(1)
        result += chart_probs[:, chart, None, None, None] * apply_d4(pooled, chart)
    return result


def _soup(experts: list[TinyUNet], payload: dict[str, Any]) -> TinyUNet:
    best, best_dice = experts[0], -math.inf
    for count in range(1, 5):
        model = average_state_dict(experts[:count])
        values = []
        for domain in range(4):
            values.append(segmentation_metrics(predict_probability(model, synthetic_domain(payload["early_images"], domain)), payload["early_masks"].numpy())["dice"])
        if float(np.mean(values)) > best_dice:
            best, best_dice = model, float(np.mean(values))
    return best


def run(smoke: bool = False) -> dict[str, Any]:
    if not dataset_ready() or not (OUT / "checkpoints" / "seed_0_chart_equivariant.pt").exists():
        update_status("C1_multidomain_experts", "blocked", "B1 checkpoints unavailable")
        return {"state": "blocked", "seeds": 0}
    seeds = [0] if smoke else list(range(5))
    summary, runs, checkpoints = [], [], []
    for seed in seeds:
        payload = role_split(seed)
        experts, prototypes, seed_checkpoints = _train_domains(seed, payload, smoke)
        checkpoints.extend(seed_checkpoints)
        chart_model, direct, one_canonical = _load_chart_and_direct(seed)
        tta = D4SymmetrizedUNet(one_canonical).to(DEVICE)
        weight_average, soup = average_state_dict(experts), _soup(experts, payload)
        saved = {}
        for condition in CONDITIONS:
            images, target_masks, true_domains, true_charts = _condition(payload, condition, seed)
            domain_probs = domain_probabilities(images, prototypes)
            if condition == "missing_local_expert":
                domain_probs[:, 0] = 0
                domain_probs /= domain_probs.sum(1, keepdim=True)
            chart_probs = chart_probabilities(chart_model, images)
            hard_domains = domain_probs.argmax(1)
            hard_charts = chart_probs.argmax(1)
            expert_logits = expert_original_frame_logits(images, experts)
            generic = (domain_probs[:, :, None, None, None] * expert_logits).sum(1)
            hard_domain = expert_logits[torch.arange(len(images)), hard_domains]
            chart_one = hard_canonical_retransport(images, one_canonical, hard_charts)
            domain_specific = torch.cat([hard_canonical_retransport(images[index:index+1], experts[int(hard_domains[index])], hard_charts[index:index+1]) for index in range(len(images))])
            chart_pool = torch.stack([hard_canonical_retransport(images, expert, hard_charts) for expert in experts], dim=1)
            chart_aware = (domain_probs[:, :, None, None, None] * chart_pool).sum(1)
            full = _domain_chart_soft(images, experts, domain_probs, chart_probs)
            oracle = torch.cat([hard_canonical_retransport(images[index:index+1], experts[min(int(true_domains[index]), 3)], true_charts[index:index+1]) for index in range(len(images))])
            candidates = {
                "weight_average": predict_logits(weight_average, images),
                "greedy_soup": predict_logits(soup, images),
                "generic_moe": generic,
                "domain_router": hard_domain,
                "chart_router": chart_one,
                "joint_domain_chart_router": chart_aware,
                "direct_d4_equivariant_unet": predict_logits(direct, images, 4),
                "d4_test_time_augmentation": predict_logits(tta, images, 4),
                "one_canonical_after_chart_inference": chart_one,
                "domain_specific_canonical_experts": domain_specific,
                "chart_aware_multi_expert_pooling": chart_aware,
                "canonicalize_pool_retransport": full,
                "supplied_domain_chart_oracle": oracle,
                "ensemble": expert_logits.mean(1),
            }
            predictions = {method: torch.sigmoid(value).numpy() for method, value in candidates.items()}
            saved.update({f"{condition}__{method}": value for method, value in predictions.items()})
            for method in METHODS:
                metrics = segmentation_metrics(predictions[method], target_masks.numpy())
                summary.append({"seed": seed, "condition": condition, "method": method, **metrics, "domain_metadata_type": "synthetic_domain", "domain_accuracy": float((hard_domains == true_domains.clamp_max(3)).float().mean()), "chart_accuracy": float((hard_charts == true_charts).float().mean())})
                for index, name in enumerate(payload["test_names"][:len(images)]):
                    one = segmentation_metrics(predictions[method][index:index+1], target_masks.numpy()[index:index+1])
                    runs.append({"seed": seed, "condition": condition, "example": name, "synthetic_domain": int(true_domains[index]), "chart": int(true_charts[index]), "method": method, **one})
        audit = save_predictions_before_metrics(DEST / "predictions" / f"seed_{seed}.npz", saved, payload["test_masks"][:4].numpy(), 360_100_000 + seed)
        for row in summary:
            if int(row["seed"]) == seed:
                row["prediction_sha256"] = audit["sha256"]
                row["prediction_hashes_unchanged"] = audit["candidate_hashes_unchanged"]
    write_csv(DEST / "runs.csv", runs)
    write_csv(DEST / "summary.csv", summary)
    unseen = [row for row in summary if row["condition"] == "unseen_domain_unseen_chart"]
    paired = paired_rows(unseen, [
        ("structured_vs_generic_moe", "canonicalize_pool_retransport", "generic_moe"),
        ("structured_vs_domain_router", "canonicalize_pool_retransport", "domain_router"),
        ("structured_vs_direct_equivariant", "canonicalize_pool_retransport", "direct_d4_equivariant_unet"),
        ("structured_vs_tta", "canonicalize_pool_retransport", "d4_test_time_augmentation"),
        ("structured_vs_one_canonical", "canonicalize_pool_retransport", "one_canonical_after_chart_inference"),
    ], "dice", 361_000_000)
    write_csv(DEST / "paired.csv", paired)
    gate = bool(all(float(row["ci_lower"]) > 0 for row in paired))
    claims = [{"claim": "multidomain_primary_gate", "passed": gate, "reason": "paired Dice on unseen synthetic-domain/chart combinations"}, {"claim": "real_multicenter_claim", "passed": False, "reason": "Kvasir-SEG has no center/site metadata; domains are synthetic"}]
    write_csv(DEST / "claims.csv", claims)
    write_csv(DEST / "by_domain_chart.csv", [{"seed": row["seed"], "condition": row["condition"], "method": row["method"], "dice": row["dice"], "boundary_dice": row["boundary_dice"]} for row in summary])
    write_csv(OUT / "checkpoint_manifest.csv", checkpoints)
    factual_report(DEST / "report.md", "Synthetic-domain and exact-chart expert merging", [
        f"Seeds executed: {seeds}; conditions: {CONDITIONS}.",
        "Kvasir-SEG provides no center/site/scanner metadata; all four shifts are labeled synthetic domains.",
        "D4 charts were treated as exact group actions; synthetic color/stain domains were used only for routing and conditioning.",
        f"Primary unseen synthetic-domain/chart gate: {gate}.",
    ])
    update_status("C1_multidomain_experts", "completed", f"{len(seeds)} seeds; synthetic-domain gate={gate}")
    stage_complete(DEST / "summary.csv", {"stage": "C1", "state": "completed", "seeds": seeds, "gate": gate, "domain_type": "synthetic"})
    return {"state": "completed", "seeds": len(seeds), "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started_at, started = utc_now(), time.perf_counter()
    revision = dataset_checksum() if dataset_ready() else "unavailable"
    try:
        result = run(args.smoke)
    except Exception as error:
        update_status("C1_multidomain_experts", "failed", str(error))
        record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=1, state="failed", summary=str(error))
        raise
    record_command(command=COMMAND + (" --smoke" if args.smoke else ""), source=SCRIPT, seed_scope="0" if args.smoke else "0:4", dataset_revision=revision, started_at=started_at, runtime=time.perf_counter()-started, exit_code=0, state=str(result["state"]), summary=f"executed {result['seeds']} seeds; gate={result.get('gate')}")


if __name__ == "__main__":
    main()
