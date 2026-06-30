#!/usr/bin/env python
"""Generate the full capacity, symmetry, and claim-boundary audit table."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


YES = "yes"
NO = "no"


@dataclass(frozen=True)
class AuditRow:
    method_id: str
    method_or_diagnostic: str
    family: str
    output_type: str
    parameter_multiplier: str
    inference_multiplier: str
    exact_relu_mlp_symmetry: str
    exact_relu_cnn_channel_symmetry: str
    exact_linear_hidden_block_symmetry: str
    diagnostic_only: str
    validation_selected: str
    official_external_code: str
    extra_capacity: str
    period_index_lift: str
    requires_residual_certification_gate: str
    dataset_architecture_scope: str
    paper_claim_allowed: str
    paper_claim_forbidden: str
    primary_evidence: str


def row(
    method_id: str,
    name: str,
    family: str,
    output_type: str,
    parameter_multiplier: str = "1x",
    inference_multiplier: str = "1x",
    mlp: str = NO,
    cnn: str = NO,
    block: str = NO,
    diagnostic: str = NO,
    validation: str = NO,
    official: str = NO,
    extra: str = NO,
    period: str = NO,
    gate: str = NO,
    scope: str = "",
    allowed: str = "",
    forbidden: str = "",
    evidence: str = "",
) -> AuditRow:
    return AuditRow(
        method_id=method_id,
        method_or_diagnostic=name,
        family=family,
        output_type=output_type,
        parameter_multiplier=parameter_multiplier,
        inference_multiplier=inference_multiplier,
        exact_relu_mlp_symmetry=mlp,
        exact_relu_cnn_channel_symmetry=cnn,
        exact_linear_hidden_block_symmetry=block,
        diagnostic_only=diagnostic,
        validation_selected=validation,
        official_external_code=official,
        extra_capacity=extra,
        period_index_lift=period,
        requires_residual_certification_gate=gate,
        dataset_architecture_scope=scope,
        paper_claim_allowed=allowed,
        paper_claim_forbidden=forbidden,
        primary_evidence=evidence,
    )


def registry() -> list[AuditRow]:
    rows: list[AuditRow] = [
        row(
            "weight_average",
            "Weight average",
            "core baselines",
            "single merged model",
            scope="MNIST/Fashion/CIFAR MLP or no-BatchNorm CNN settings",
            allowed="Capacity-matched baseline.",
            forbidden="Do not call it obstruction-aware or symmetry-corrected.",
            evidence="reports/model_merging_verification_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "git_rebasin_pairwise_alignment",
            "Git-ReBasin-style pairwise alignment",
            "core baselines",
            "single merged model after pairwise permutation alignment",
            mlp=YES,
            cnn=YES,
            scope="Internal faithful-style MLP hidden permutations and CNN channel permutations",
            allowed="Internal pairwise-permutation baseline and diagnostic.",
            forbidden="Do not claim official Git Re-Basin code execution or global cycle consistency.",
            evidence="reports/external_baseline_comparison.md; reports/official_external_baseline_attempt.md",
        ),
        row(
            "c2m3_cycle_synchronization",
            "C2M3-style cycle-consistent synchronization",
            "core baselines",
            "single merged model after synchronized permutations",
            mlp=YES,
            cnn=YES,
            scope="Internal MLP hidden permutations and CNN channel permutations",
            allowed="Internal C2M3-style cycle-consistent baseline.",
            forbidden="Do not claim official C2M3 code execution or broad CIFAR wins.",
            evidence="reports/external_baseline_comparison.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "greedy_soup",
            "Greedy soup / Model Soups-style validation soup",
            "core baselines",
            "single model produced by validation-selected weight averaging",
            validation=YES,
            scope="MNIST/Fashion/CIFAR benchmark candidate pools",
            allowed="Capacity-matched greedy-soup baseline.",
            forbidden="Do not claim official Model Soups code execution unless separately run.",
            evidence="reports/external_baseline_comparison.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "ensemble_upper_bound",
            "Ensemble upper bound",
            "core baselines",
            "ensemble prediction",
            parameter_multiplier="N x",
            inference_multiplier="N x",
            extra=YES,
            scope="All model-merging benchmark families",
            allowed="Upper bound with explicit extra capacity and inference cost.",
            forbidden="Do not compare as capacity-matched single merged model.",
            evidence="reports/model_merging_verification_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "official_external_code_attempt",
            "Official external baseline integration attempt",
            "external baselines",
            "integration status log",
            diagnostic=YES,
            official=YES,
            scope="Official Git Re-Basin, C2M3, Model Soups repositories",
            allowed="License and integration-boundary documentation.",
            forbidden="Do not report official-code benchmark results from this attempt.",
            evidence="external_baselines/OFFICIAL_INTEGRATION.md; reports/official_external_baseline_attempt.md",
        ),
        row(
            "official_nsd_smoke",
            "Official Neural Sheaf Diffusion smoke run",
            "external baselines",
            "external GNN run plus post-hoc cycle diagnostic",
            diagnostic=YES,
            official=YES,
            scope="Tiny WebKB Texas BundleSheaf in separate PyG environment",
            allowed="Optional official NSD integration feasibility and post-hoc cycle diagnostic.",
            forbidden="Do not claim sheaf regularization generally improves GNNs.",
            evidence="external_baselines/NSD_INTEGRATION.md; reports/nsd_official_integration_report.md",
        ),
        row(
            "monomial_raw_scale",
            "Raw positive monomial scale",
            "MLP monomial gauges",
            "single merged model after exact positive ReLU gauge",
            mlp=YES,
            scope="ReLU MLPs without BatchNorm",
            allowed="Exact positive-gauge candidate for ReLU MLP hidden units.",
            forbidden="Do not claim robust greedy-soup improvement.",
            evidence="reports/validated_ladder_merge_report.md; reports/greedy_aware_monomial_report.md",
        ),
        row(
            "monomial_shrinkage_scale",
            "Shrinkage monomial scale",
            "MLP monomial gauges",
            "single merged model after validation-selected shrinkage gauge",
            mlp=YES,
            validation=YES,
            scope="ReLU MLPs without BatchNorm",
            allowed="Exact-gauge C2M3-style refinement where paired evidence supports it.",
            forbidden="Do not claim greedy-soup win in current artifacts.",
            evidence="reports/improved_validated_ladder_merge_report.md; reports/greedy_aware_monomial_report.md",
        ),
        row(
            "monomial_global_scale",
            "Global monomial scale synchronization",
            "MLP monomial gauges",
            "single merged model after global positive-scale synchronization",
            mlp=YES,
            validation=YES,
            scope="ReLU MLPs without BatchNorm",
            allowed="Exact-gauge diagnostic/refinement for MLP permutation alignments.",
            forbidden="Do not call the result a broad natural model-merging solution.",
            evidence="reports/improved_validated_ladder_merge_report.md",
        ),
        row(
            "monomial_optimized_scale",
            "Validation-optimized monomial scale",
            "MLP monomial gauges",
            "single merged model after validation grid selection",
            mlp=YES,
            validation=YES,
            scope="ReLU MLPs without BatchNorm",
            allowed="Validation-selected exact-gauge candidate.",
            forbidden="Do not claim it beats greedy soup overall.",
            evidence="reports/improved_validated_ladder_merge_report.md; reports/greedy_aware_monomial_report.md",
        ),
        row(
            "monomial_scaled_greedy_soup",
            "Monomial-scaled greedy soup",
            "MLP soups",
            "single model soup over exact-gauged candidates",
            mlp=YES,
            validation=YES,
            scope="ReLU MLP candidate pools",
            allowed="Capacity-matched soup candidate pool.",
            forbidden="Do not claim improvement over ordinary greedy soup in current artifacts.",
            evidence="reports/external_baseline_comparison.md; reports/greedy_aware_monomial_report.md",
        ),
        row(
            "monomial_shrinkage_greedy_soup",
            "Shrinkage monomial greedy soup",
            "MLP soups",
            "single model soup over shrinkage-scaled exact-gauge candidates",
            mlp=YES,
            validation=YES,
            scope="ReLU MLP candidate pools",
            allowed="Capacity-matched validation-selected shrinkage-gauge candidate soup.",
            forbidden="Do not claim broad improvement over ordinary greedy soup.",
            evidence="reports/greedy_aware_monomial_report.md",
        ),
        row(
            "monomial_global_greedy_soup",
            "Global-scale monomial greedy soup",
            "MLP soups",
            "single model soup over globally synchronized exact-gauge candidates",
            mlp=YES,
            validation=YES,
            scope="ReLU MLP candidate pools",
            allowed="Capacity-matched validation-selected global-gauge candidate soup.",
            forbidden="Do not claim overall greedy-soup win.",
            evidence="reports/greedy_aware_monomial_report.md",
        ),
        row(
            "monomial_optimized_greedy_soup",
            "Optimized monomial greedy soup",
            "MLP soups",
            "single model soup over validation-grid exact-gauge candidates",
            mlp=YES,
            validation=YES,
            scope="ReLU MLP candidate pools",
            allowed="Capacity-matched validation-selected optimized-gauge candidate soup.",
            forbidden="Do not claim robust greedy-soup improvement in current artifacts.",
            evidence="reports/greedy_aware_monomial_report.md",
        ),
        row(
            "union_candidate_soup",
            "Union candidate soup",
            "MLP soups",
            "single model soup over original, C2M3, and monomial candidates",
            mlp=YES,
            validation=YES,
            scope="MNIST/Fashion MLP candidate pools",
            allowed="Soup-compatible selector boundary experiment.",
            forbidden="Do not claim broad improvement over greedy soup.",
            evidence="reports/greedy_aware_monomial_report.md; reports/external_baseline_comparison.md",
        ),
        row(
            "improved_validated_selector",
            "Improved validated ladder selector",
            "selectors",
            "validation-selected single-model candidate",
            mlp=YES,
            validation=YES,
            scope="MNIST/Fashion MLP exact-gauge candidate pools",
            allowed="Limited internal C2M3-style improvement where paired CI supports it.",
            forbidden="Do not claim external C2M3, greedy-soup, CIFAR, or broad architecture win.",
            evidence="reports/improved_validated_ladder_merge_report.md; reports/external_baseline_comparison.md",
        ),
        row(
            "greedy_aware_selector",
            "Greedy-aware monomial selector",
            "selectors",
            "validation-selected single-model candidate with greedy baseline awareness",
            mlp=YES,
            validation=YES,
            scope="MNIST MLP monomial candidate pools",
            allowed="Selector behavior and regret diagnostics.",
            forbidden="Do not claim overall greedy-soup win.",
            evidence="reports/greedy_aware_monomial_report.md",
        ),
        row(
            "greedy_safe_selector",
            "Greedy-safe selector",
            "selectors",
            "validation-gated selector defaulting to greedy soup",
            validation=YES,
            scope="Fashion/CIFAR CNN and MLP candidate pools",
            allowed="No-harm selector boundary when it matches or avoids harmful departures from greedy soup.",
            forbidden="Do not call exact gauge when it chooses ordinary greedy soup.",
            evidence="reports/fashion_mnist_greedy_safe_selector_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_channel_permutation",
            "CNN channel permutation / C2M3 channel synchronization",
            "CNN channel gauges",
            "single merged no-BatchNorm CNN after channel permutation",
            cnn=YES,
            scope="No-BatchNorm ReLU CNNs with conv and hidden channel gauges",
            allowed="Exact CNN channel-gauge baseline.",
            forbidden="Do not claim BatchNorm gauge or general CNN rotation symmetry.",
            evidence="tests/test_cnn_channel_gauge.py; reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_positive_channel_scale",
            "CNN positive channel scale",
            "CNN channel gauges",
            "single merged no-BatchNorm CNN after exact positive channel scaling",
            cnn=YES,
            scope="No-BatchNorm ReLU CNNs",
            allowed="Exact positive channel-gauge diagnostic.",
            forbidden="Do not claim CIFAR improvement; final CIFAR run is negative/descriptive only.",
            evidence="tests/test_cnn_channel_gauge.py; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_shrinkage_channel_scale",
            "CNN shrinkage channel scale",
            "CNN channel gauges",
            "single merged no-BatchNorm CNN after validation-selected shrinkage scale",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and final bounded CIFAR no-BatchNorm CNNs",
            allowed="Fashion-MNIST limited C2M3 improvement; CIFAR descriptive boundary.",
            forbidden="Do not promote CIFAR result to main broad claim.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_global_channel_scale",
            "CNN global channel-scale synchronization",
            "CNN channel gauges",
            "single merged no-BatchNorm CNN after global channel-scale synchronization",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and final bounded CIFAR no-BatchNorm CNNs",
            allowed="Exact-gauge candidate with validation-only selection.",
            forbidden="Do not claim robust CIFAR improvement.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_optimized_channel_scale",
            "CNN optimized channel scale",
            "CNN channel gauges",
            "single merged no-BatchNorm CNN after validation grid selection",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and final bounded CIFAR no-BatchNorm CNNs",
            allowed="Fashion-MNIST limited improvement over C2M3; CIFAR descriptive only.",
            forbidden="Do not claim greedy-soup or broad CIFAR win.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_positive_channel_scaled_soup",
            "CNN positive channel-scaled greedy soup",
            "CNN soups",
            "single model soup over positive channel-scale candidates",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and CIFAR no-BatchNorm CNNs",
            allowed="Capacity-matched positive channel-scale candidate soup diagnostic.",
            forbidden="Do not claim CIFAR improvement or greedy-soup win from this row.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_shrinkage_channel_scaled_soup",
            "CNN shrinkage channel-scaled greedy soup",
            "CNN soups",
            "single model soup over shrinkage channel-scale candidates",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and CIFAR no-BatchNorm CNNs",
            allowed="Capacity-matched shrinkage candidate soup diagnostic.",
            forbidden="Do not claim robust greedy-soup improvement unless CI lower bound is positive.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_global_channel_scaled_soup",
            "CNN global channel-scaled greedy soup",
            "CNN soups",
            "single model soup over globally synchronized channel-scale candidates",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and CIFAR no-BatchNorm CNNs",
            allowed="Capacity-matched global channel-scale candidate soup diagnostic.",
            forbidden="Do not claim broad vision or CIFAR method win.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_optimized_channel_scaled_soup",
            "CNN optimized channel-scaled greedy soup",
            "CNN soups",
            "single model soup over validation-optimized channel-scale candidates",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and CIFAR no-BatchNorm CNNs",
            allowed="Capacity-matched optimized channel-scale candidate soup diagnostic.",
            forbidden="Do not claim robust greedy-soup improvement unless CI lower bound is positive.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cnn_union_candidate_soup",
            "CNN union channel candidate soup",
            "CNN soups",
            "single model soup over original, C2M3, and channel-scale candidates",
            cnn=YES,
            validation=YES,
            scope="Fashion-MNIST and CIFAR no-BatchNorm CNNs",
            allowed="Descriptive/small limited soup boundary where supported.",
            forbidden="Do not claim broad vision or external baseline win.",
            evidence="reports/fashion_mnist_cnn_channel_gauge_confirmatory_report.md; reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "bridge_rotated_mnist_channel_sync",
            "Rotated-MNIST bridge C2M3 channel synchronization",
            "bridge datasets",
            "single merged no-BatchNorm CNN after channel synchronization",
            cnn=YES,
            scope="Rotated-MNIST bridge dataset only",
            allowed="Bridge-only internal C2M3-style channel synchronization evidence.",
            forbidden="Do not infer CIFAR or general vision performance.",
            evidence="reports/bridge_dataset_channel_gauge_expansion.md",
        ),
        row(
            "bridge_colored_mnist_channel_sync",
            "Colored-MNIST bridge C2M3 channel synchronization",
            "bridge datasets",
            "single merged no-BatchNorm CNN after channel synchronization",
            cnn=YES,
            scope="Colored-MNIST bridge dataset only",
            allowed="Bridge-only internal C2M3-style channel synchronization evidence.",
            forbidden="Do not infer CIFAR, ImageNet, or general vision performance.",
            evidence="reports/bridge_dataset_channel_gauge_expansion.md",
        ),
        row(
            "bridge_channel_scale_candidates",
            "Rotated/colored-MNIST bridge channel-scale candidates",
            "bridge datasets",
            "single merged no-BatchNorm CNN channel-scale candidates",
            cnn=YES,
            validation=YES,
            scope="Rotated-MNIST and colored-MNIST bridge datasets only",
            allowed="Bridge-only validation-selected exact-gauge candidate evidence.",
            forbidden="Do not promote bridge channel scales to broad vision claims.",
            evidence="reports/bridge_dataset_channel_gauge_expansion.md",
        ),
        row(
            "bridge_union_candidate_soup",
            "Rotated/colored-MNIST bridge union candidate soup",
            "bridge datasets",
            "single model soup over bridge C2M3 and channel-scale candidates",
            cnn=YES,
            validation=YES,
            scope="Rotated-MNIST and colored-MNIST bridge datasets only",
            allowed="Bridge-only capacity-matched soup boundary experiment.",
            forbidden="Do not claim external Model Soups or CIFAR performance.",
            evidence="reports/bridge_dataset_channel_gauge_expansion.md",
        ),
        row(
            "bridge_greedy_safe_selector",
            "Rotated/colored-MNIST bridge greedy-safe selector",
            "bridge datasets",
            "validation-gated selector defaulting to bridge greedy soup",
            validation=YES,
            scope="Rotated-MNIST and colored-MNIST bridge datasets only",
            allowed="Bridge-only no-harm selector diagnostic.",
            forbidden="Do not infer CIFAR or general vision performance.",
            evidence="reports/bridge_dataset_channel_gauge_expansion.md",
        ),
        row(
            "cifar_final_c2m3_channel_sync",
            "Final bounded CIFAR C2M3 channel synchronization",
            "CIFAR final",
            "single merged no-BatchNorm CNN after channel synchronization",
            cnn=YES,
            scope="CIFAR-10 no-BatchNorm CNN, 32/64/256, N=3, five seeds",
            allowed="Base accuracy gate passed; bounded appendix baseline.",
            forbidden="Do not claim official C2M3 execution or broad CIFAR win.",
            evidence="reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cifar_final_positive_channel_scale",
            "Final bounded CIFAR positive channel scale",
            "CIFAR final",
            "single merged no-BatchNorm CNN after positive channel scaling",
            cnn=YES,
            scope="CIFAR-10 no-BatchNorm CNN, 32/64/256, N=3, five seeds",
            allowed="Bounded appendix exact-gauge diagnostic.",
            forbidden="Do not claim CIFAR improvement; reported deltas are negative/descriptive.",
            evidence="reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cifar_final_shrinkage_channel_scale",
            "Final bounded CIFAR shrinkage channel scale",
            "CIFAR final",
            "single merged no-BatchNorm CNN after validation-selected shrinkage scale",
            cnn=YES,
            validation=YES,
            scope="CIFAR-10 no-BatchNorm CNN, 32/64/256, N=3, five seeds",
            allowed="Bounded appendix validation-selected exact-gauge candidate.",
            forbidden="Do not claim broad or statistically confirmed CIFAR improvement.",
            evidence="reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cifar_final_global_channel_scale",
            "Final bounded CIFAR global channel scale",
            "CIFAR final",
            "single merged no-BatchNorm CNN after global channel-scale synchronization",
            cnn=YES,
            validation=YES,
            scope="CIFAR-10 no-BatchNorm CNN, 32/64/256, N=3, five seeds",
            allowed="Bounded appendix validation-selected exact-gauge candidate.",
            forbidden="Do not claim broad or statistically confirmed CIFAR improvement.",
            evidence="reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cifar_final_optimized_channel_scale",
            "Final bounded CIFAR optimized channel scale",
            "CIFAR final",
            "single merged no-BatchNorm CNN after validation-optimized channel scaling",
            cnn=YES,
            validation=YES,
            scope="CIFAR-10 no-BatchNorm CNN, 32/64/256, N=3, five seeds",
            allowed="Bounded appendix validation-selected exact-gauge candidate.",
            forbidden="Do not claim broad or statistically confirmed CIFAR improvement.",
            evidence="reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cifar_final_union_candidate_soup",
            "Final bounded CIFAR union candidate soup",
            "CIFAR final",
            "single model soup over CIFAR baseline and exact-gauge candidates",
            cnn=YES,
            validation=YES,
            scope="CIFAR-10 no-BatchNorm CNN, 32/64/256, N=3, five seeds",
            allowed="Bounded appendix soup-compatible candidate diagnostic.",
            forbidden="Do not claim greedy-soup win when confidence interval touches zero.",
            evidence="reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "cifar_final_greedy_safe_selector",
            "Final bounded CIFAR greedy-safe selector",
            "CIFAR final",
            "validation-gated selector defaulting to greedy soup",
            validation=YES,
            scope="CIFAR-10 no-BatchNorm CNN, 32/64/256, N=3, five seeds",
            allowed="Bounded appendix no-harm selector diagnostic.",
            forbidden="Do not promote to a general CIFAR model-merging solution.",
            evidence="reports/cifar_final_channel_gauge_confirmatory_report.md",
        ),
        row(
            "twisted_rank_lift_branch",
            "TwistedMerge q=2 branch lift",
            "twisted/branch lifts",
            "branch prediction / extra-capacity lifted representation",
            parameter_multiplier="q x",
            inference_multiplier="q x or branch ensemble proxy",
            extra=YES,
            gate=YES,
            scope="Controlled finite central mu_2 example",
            allowed="Detects failed gauge synchronization and recovers prediction in controlled branch-lift example.",
            forbidden="Do not call capacity-matched single merged model or nonzero H2 trivialization.",
            evidence="tests/test_twisted_merge_algorithm.py; reports/twisted_merge_algorithm_verification.md",
        ),
        row(
            "twisted_merge_plus_branch_selector",
            "TwistedMerge++ residual classifier/selector",
            "twisted/branch lifts",
            "selector over C2M3, edge-outlier, central, and projective cases",
            diagnostic=YES,
            gate=YES,
            scope="Controlled residual-classification demos",
            allowed="Residual taxonomy and guarded lift activation.",
            forbidden="Do not claim natural MNIST/CIFAR solution.",
            evidence="tests/test_twisted_merge_plus.py; reports/twisted_merge_plus_report.md",
        ),
        row(
            "finite_index_projective_lift",
            "Finite-index projective lift",
            "period-index/projective lifts",
            "rank-d projective/direct-sum lift",
            parameter_multiplier="d x or rank divisible by d",
            inference_multiplier="d x proxy",
            extra=YES,
            period=YES,
            gate=YES,
            scope="Controlled clock-shift finite torsion examples",
            allowed="Period/index threshold and rank obstruction in controlled algebraic setting.",
            forbidden="Do not claim real neural defects have clock-shift form.",
            evidence="reports/finite_index_twist_report.md; reports/twisted_merge_plus_finite_index_report.md",
        ),
        row(
            "period_index_projective_morita_lift",
            "Period-index projective/Morita lift",
            "period-index/projective lifts",
            "rank d^k projective/Morita lift",
            parameter_multiplier="index d^k x",
            inference_multiplier="index d^k x proxy",
            extra=YES,
            period=YES,
            gate=YES,
            scope="Controlled finite Heisenberg k-pair central systems",
            allowed="Certified period/index rank threshold in controlled cases.",
            forbidden="Do not claim same-cover ordinary vector-bundle trivialization.",
            evidence="reports/period_index_central_report.md; reports/twisted_merge_plus_period_index_report.md",
        ),
        row(
            "commutator_matrix_detector",
            "Central commutator-matrix period-index detector",
            "period-index diagnostics",
            "diagnostic detector and rank decision",
            diagnostic=YES,
            period=YES,
            gate=YES,
            scope="Controlled central/projective generators and noncentral controls",
            allowed="Certified period/index detector when centrality and rank gates pass.",
            forbidden="Do not lift unknown-index or noncentral cases.",
            evidence="reports/period_index_commutator_matrix_report.md",
        ),
        row(
            "robust_period_index_detector",
            "Robust/noisy period-index detector",
            "period-index diagnostics",
            "diagnostic detector with calibrated uncertainty/rejection",
            diagnostic=YES,
            period=YES,
            gate=YES,
            scope="Controlled noisy Heisenberg, loop-mining, and noncentral controls",
            allowed="Certified/uncertain/rejected detector state under calibrated thresholds.",
            forbidden="Do not lift uncertain candidates.",
            evidence="reports/robust_period_index_detector_report.md; reports/robust_period_index_calibration_report.md",
        ),
        row(
            "time_frequency_known_chart",
            "Known finite time-frequency chart operators",
            "time-frequency period-index",
            "diagnostic known-operator period/index certification",
            diagnostic=YES,
            period=YES,
            gate=YES,
            scope="Synthetic finite time-frequency signal-domain operators",
            allowed="Natural signal-domain central projective relation in controlled charts.",
            forbidden="Do not claim arbitrary neural hidden layers expose this structure.",
            evidence="reports/time_frequency_period_index_report.md",
        ),
        row(
            "time_frequency_learned_charts",
            "Learned time-frequency chart maps",
            "time-frequency period-index",
            "learned chart diagnostic plus guarded lift decision",
            diagnostic=YES,
            period=YES,
            gate=YES,
            scope="Controlled paired-chart and autoencoder chart recovery",
            allowed="Certified learned chart recovery only when detector passes.",
            forbidden="Do not claim supervised encoder or noisy rejected rows are valid lifts.",
            evidence="reports/time_frequency_learned_chart_report.md",
        ),
        row(
            "denoised_learned_chart_methods",
            "Denoised learned-chart methods",
            "time-frequency period-index",
            "structure-preserving chart denoising diagnostic",
            diagnostic=YES,
            period=YES,
            gate=YES,
            scope="Controlled noisy time-frequency learned charts",
            allowed="Denoising can improve certified recovery under small noise.",
            forbidden="Do not use operator-error reduction alone as lift certificate.",
            evidence="reports/time_frequency_denoised_chart_report.md",
        ),
        row(
            "nearest_heisenberg_projection",
            "Nearest finite-Heisenberg projection",
            "time-frequency period-index",
            "projection diagnostic plus residual-gated certification",
            diagnostic=YES,
            period=YES,
            gate=YES,
            scope="Controlled time-frequency projection benchmark",
            allowed="Projection can extend certified recovery when projection residual gate accepts.",
            forbidden="Do not accept canonical replacement with large residual.",
            evidence="reports/time_frequency_heisenberg_projection_report.md",
        ),
        row(
            "finite_index_residual_mining",
            "Finite-index residual mining",
            "residual mining",
            "diagnostic mining table",
            diagnostic=YES,
            period=YES,
            gate=YES,
            scope="Real MNIST activation-permutation residuals plus clock-shift controls",
            allowed="Negative real-residual evidence and positive-control detection.",
            forbidden="Do not claim real MNIST residuals are finite-index projective classes.",
            evidence="reports/finite_index_residual_mining_report.md",
        ),
        row(
            "structure_group_ladder",
            "Structure-group ladder diagnostics",
            "structure-group diagnostics",
            "diagnostic residual taxonomy over permutation, signed, monomial, GL, and block groups",
            diagnostic=YES,
            gate=YES,
            scope="Real MNIST/Fashion residual diagnostics and controlled cases",
            allowed="Diagnostic taxonomy and negative central/projective evidence.",
            forbidden="Do not call signed/full-GL/block rotations exact ReLU merges.",
            evidence="reports/structure_group_ladder_report.md; reports/block_orthogonal_ladder_report.md",
        ),
        row(
            "block_orthogonal_ladder",
            "Block-orthogonal ladder diagnostics",
            "block diagnostics",
            "diagnostic block-orthogonal residual analysis",
            diagnostic=YES,
            gate=YES,
            scope="Real MNIST ReLU MLP block-size diagnostics plus synthetic controls",
            allowed="Feature-space diagnostic and controlled block-holonomy checks.",
            forbidden="Do not report ReLU block-orthogonal rotations as exact single-model merges.",
            evidence="reports/block_orthogonal_ladder_report.md",
        ),
        row(
            "global_block_synchronization",
            "Global block synchronization",
            "block diagnostics",
            "diagnostic projected block gauges with connection-residual honesty check",
            diagnostic=YES,
            gate=YES,
            scope="Synthetic exact gauges and real MNIST block diagnostics",
            allowed="Projected-cycle diagnostics plus connection residual calibration.",
            forbidden="Do not use post-projection cycle score alone as descent proof.",
            evidence="reports/global_block_synchronization_report.md; reports/optimized_global_block_synchronization_report.md",
        ),
        row(
            "optimized_global_block_synchronization",
            "Connection-residual optimized global block synchronization",
            "block diagnostics",
            "diagnostic block-gauge optimization",
            diagnostic=YES,
            gate=YES,
            scope="Controlled block-gauge grids and real ReLU diagnostic rows",
            allowed="Residual-optimized diagnostics and fake-projection-trap rejection.",
            forbidden="Do not claim natural MNIST/CIFAR merge improvement.",
            evidence="reports/block_gauge_phase_diagram_report.md; reports/block_gauge_branch_closure_report.md",
        ),
        row(
            "learned_block_partitions",
            "Learned block partitions",
            "block diagnostics",
            "diagnostic/validation-selected block partition candidates",
            diagnostic=YES,
            validation=YES,
            gate=YES,
            scope="Controlled learned-block recovery and real MNIST diagnostics",
            allowed="Learned partition recovery on planted controls.",
            forbidden="Do not claim learned blocks improve real MNIST residuals in current run.",
            evidence="reports/global_block_synchronization_report.md; reports/block_gauge_phase_diagram_report.md",
        ),
        row(
            "block_compatible_aligned_average",
            "Block-compatible aligned average",
            "block-compatible exact merge",
            "single merged model in exact linear-hidden architecture",
            block=YES,
            validation=YES,
            scope="Controlled identity/linear-hidden block-compatible architecture",
            allowed="Capacity-matched exact block-gauge aligned average in controlled linear-hidden setting.",
            forbidden="Do not infer natural ReLU or CIFAR block-gauge performance.",
            evidence="reports/block_compatible_learning_report.md",
        ),
        row(
            "relu_block_diagnostic",
            "Real ReLU block diagnostic rows",
            "block diagnostics",
            "diagnostic-only ReLU block residual rows",
            diagnostic=YES,
            gate=YES,
            scope="Real ReLU MLP block diagnostics",
            allowed="Diagnostic-only ReLU block residual reporting.",
            forbidden="Do not report block-orthogonal ReLU merge accuracy.",
            evidence="reports/relu_block_diagnostic_report.md",
        ),
        row(
            "sheaf_gnn_cycle_diagnostic",
            "PyTorch-only sheaf/GNN cycle diagnostic",
            "sheaf/GNN diagnostics",
            "diagnostic GNN/sheaf run and cycle score",
            diagnostic=YES,
            validation=YES,
            scope="Synthetic heterophilic small graphs",
            allowed="Cycle inconsistency can be measured in a synthetic optional diagnostic.",
            forbidden="Do not claim sheaf regularization generally improves GNNs.",
            evidence="reports/sheaf_gnn_optional_report.md",
        ),
        row(
            "unified_quantitative_obstruction_chain",
            "Unified quantitative obstruction chain",
            "meta audit/aggregation",
            "diagnostic aggregate table over existing artifacts",
            diagnostic=YES,
            scope="Existing obstruction, detector, gate, selector, block, time-frequency, and sheaf artifacts",
            allowed="Artifact-scoped chain of residual gates and claim boundaries.",
            forbidden="Do not treat aggregation as new experimental success.",
            evidence="reports/unified_quantitative_obstruction_chain.md",
        ),
    ]
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | str:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return "unknown"


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in rows:
        lines.append("| " + " | ".join(str(item.get(col, "")).replace("|", "/") for col in columns) + " |")
    return "\n".join(lines)


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def write_csv(rows: list[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(item) for item in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def write_markdown(rows: list[AuditRow], path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(item) for item in rows]
    family_counts: dict[str, int] = {}
    for item in data:
        family_counts[item["family"]] = family_counts.get(item["family"], 0) + 1
    count_rows = [{"family": key, "rows": value} for key, value in sorted(family_counts.items())]
    compact_columns = [
        "method_id",
        "family",
        "output_type",
        "parameter_multiplier",
        "inference_multiplier",
        "diagnostic_only",
        "validation_selected",
        "extra_capacity",
        "period_index_lift",
        "paper_claim_allowed",
        "paper_claim_forbidden",
    ]
    full_columns = list(data[0].keys())
    report = f"""# Full Capacity And Claim-Boundary Audit

Generated by `experiments/full_capacity_claim_audit.py`.

## Exact Command

```bash
{command}
```

## Git State At Generation

- HEAD commit at generation: `{git_commit()}`
- Worktree dirty at generation: `{git_dirty()}`

## Purpose

This is the repository-wide capacity, symmetry, and claim-boundary registry. The CSV is the authoritative full-width artifact. The Markdown and LaTeX files are rendered views of the same row set.

Use this table to prevent overclaiming about:

- exact ReLU MLP positive/permutation gauges versus heuristic or diagnostic transforms;
- exact no-BatchNorm CNN channel gauges versus BatchNorm or general rotation claims;
- exact linear-hidden block-compatible merges versus ReLU block diagnostics;
- validation-selected single-model candidates versus ensembles or branch/projective lifts;
- official external code status versus internal faithful-style baselines;
- period-index lifts and projection methods that require residual/certification gates.

## Row Counts

{markdown_table(count_rows, ["family", "rows"])}

## Compact Audit Table

{markdown_table(data, compact_columns)}

## Full Audit Table

{markdown_table(data, full_columns)}
"""
    path.write_text(report, encoding="utf-8")


def write_latex(rows: list[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("method_id", "Method"),
        ("family", "Family"),
        ("output_type", "Output"),
        ("parameter_multiplier", "Param"),
        ("inference_multiplier", "Infer"),
        ("diagnostic_only", "Diag"),
        ("validation_selected", "Val"),
        ("extra_capacity", "Extra"),
        ("period_index_lift", "PI"),
        ("requires_residual_certification_gate", "Gate"),
    ]
    lines = [
        "\\begin{longtable}{p{0.18\\linewidth}p{0.12\\linewidth}p{0.23\\linewidth}lllllll}",
        "\\caption{Full capacity and claim-boundary audit. The CSV contains all claim-boundary text.}\\\\",
        "\\toprule",
        " & ".join(label for _key, label in columns) + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        " & ".join(label for _key, label in columns) + " \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for item in rows:
        data = asdict(item)
        lines.append(" & ".join(latex_escape(data[key]) for key, _label in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def update_claims_audit(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start = "## Full Capacity And Claim-Boundary Audit"
    next_heading = "\n## Not Yet Supported"
    if start in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(next_heading, 1)
        text = before.rstrip() + next_heading + after
    section = f"""{start}

| Claim | Status | Evidence |
| --- | --- | --- |
| `full_capacity_claim_audit_created` / authoritative capacity and symmetry registry exists | Supported | `reports/full_capacity_claim_audit.md` and `reports/csv/full_capacity_claim_audit.csv` enumerate current methods and diagnostics with output type, multipliers, exact-symmetry flags, validation-selection status, official-code status, extra-capacity status, residual gates, scopes, and allowed/forbidden paper claims. |
| `capacity_claim_boundaries_are_explicit` / broad overclaim boundaries are machine-readable | Supported | `reports/csv/full_capacity_claim_audit.csv` records per-row `paper_claim_allowed` and `paper_claim_forbidden` fields, including explicit boundaries for external official code, CIFAR, BatchNorm, ReLU block rotations, period-index lifts, and sheaf/GNN diagnostics. |
"""
    text = text.replace("\n## Not Yet Supported", "\n\n" + section + "\n## Not Yet Supported", 1)
    artifact_marker = "| `reports/csv/model_merging_stats.csv` | Correlations, bootstrap intervals, deltas, and negative-result labels for verification settings. |"
    artifacts = [
        "| `experiments/full_capacity_claim_audit.py` | Generates the repository-wide capacity, exact-symmetry, diagnostic, validation-selection, official-code, extra-capacity, residual-gate, and claim-boundary registry. |",
        "| `reports/full_capacity_claim_audit.md` | Markdown report rendering the full capacity and claim-boundary audit. |",
        "| `reports/csv/full_capacity_claim_audit.csv` | Authoritative full-width capacity and claim-boundary table. |",
        "| `reports/tables/full_capacity_claim_audit.tex` | LaTeX longtable rendering of the audit. |",
    ]
    if "`experiments/full_capacity_claim_audit.py`" not in text:
        text = text.replace(artifact_marker, "\n".join([artifact_marker, *artifacts]), 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    command = " ".join([sys.executable, *sys.argv])
    rows = registry()
    csv_path = args.reports_dir / "csv" / "full_capacity_claim_audit.csv"
    md_path = args.reports_dir / "full_capacity_claim_audit.md"
    tex_path = args.reports_dir / "tables" / "full_capacity_claim_audit.tex"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, command)
    write_latex(rows, tex_path)
    update_claims_audit(args.reports_dir / "claims_audit.md")
    metadata = {
        "rows": len(rows),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
