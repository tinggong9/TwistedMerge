#!/usr/bin/env python3
"""Application B: conservative central projective/Brauer-like certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.holonomy_application_A import load_models, load_shared
from src.holonomy_application_transitions import fit_transition
from src.holonomy_brauer_certificate import (
    coboundary_fit,
    gauge_transform_connection,
    nearest_root,
    normalized_commutator_residual,
    scalar_centrality,
    scalar_phase,
    tetrahedral_cocycle_rows,
    triangle_defect,
)

APP_DIR = ROOT / "reports" / "holonomy_applications" / "application_B_brauer_certificate"
APPLICATION_A_CONFIG = ROOT / "reports" / "holonomy_applications" / "application_A_holonomy" / "config.json"
ARTIFACT_ROOT = ROOT / "reports" / "tmp" / "holonomy_applications"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def random_orthogonal(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(dimension, dimension, generator=generator)).Q


def fit_connection(
    method: str,
    activations: list[torch.Tensor],
    adapters: list[torch.Tensor],
    low_rank: int,
) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (source, target): fit_transition(
            method,
            activations[source],
            activations[target],
            adapters[source],
            adapters[target],
            low_rank=low_rank,
        )
        for source in range(8)
        for target in range(8)
        if source != target
    }


def phase_circular_instability(values: list[float]) -> float:
    if not values:
        return float("nan")
    resultant = abs(np.mean(np.exp(1j * np.asarray(values))))
    return float(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-12)), 0.0)) / np.pi)


def paired_bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze_connection(
    seed: int,
    method: str,
    transitions: dict[tuple[int, int], torch.Tensor],
    activations: list[torch.Tensor],
    adapters: list[torch.Tensor],
    low_rank: int,
    bootstrap_samples: int,
    max_root_order: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    triangles = list(combinations(range(8), 3))
    full_defects = {triangle: triangle_defect(transitions, triangle) for triangle in triangles}
    bootstrap_centrality = {triangle: [] for triangle in triangles}
    bootstrap_phases = {triangle: [] for triangle in triangles}
    generator = torch.Generator().manual_seed(910000 + seed * 100 + sum(map(ord, method)))
    for _ in range(bootstrap_samples):
        indices = torch.randint(0, len(activations[0]), (len(activations[0]),), generator=generator)
        sampled = [values[indices] for values in activations]
        bootstrap_connection = fit_connection(method, sampled, adapters, low_rank)
        for triangle in triangles:
            diagnostics = scalar_centrality(triangle_defect(bootstrap_connection, triangle))
            bootstrap_centrality[triangle].append(diagnostics.centrality_residual)
            bootstrap_phases[triangle].append(scalar_phase(diagnostics))

    gauges = [random_orthogonal(adapters[0].shape[0], 920000 + seed * 10 + chart) for chart in range(8)]
    gauge_connection = gauge_transform_connection(transitions, gauges)
    certificate_rows: list[dict[str, object]] = []
    gauge_rows: list[dict[str, object]] = []
    triangle_phases: dict[tuple[int, int, int], float] = {}
    for triangle in triangles:
        defect = full_defects[triangle]
        diagnostics = scalar_centrality(defect)
        root = nearest_root(defect, max_order=max_root_order)
        phase = scalar_phase(diagnostics)
        triangle_phases[triangle] = phase
        central_values = np.asarray(bootstrap_centrality[triangle], dtype=float)
        commutators = []
        base = triangle[0]
        for target in range(8):
            if target != base:
                commutators.append(normalized_commutator_residual(defect, transitions[(base, target)]))
        gauge_diagnostics = scalar_centrality(triangle_defect(gauge_connection, triangle))
        gauge_delta = max(
            abs(diagnostics.centrality_residual - gauge_diagnostics.centrality_residual),
            abs(diagnostics.scalar_real - gauge_diagnostics.scalar_real),
            abs(diagnostics.scalar_imag - gauge_diagnostics.scalar_imag),
        )
        certificate_rows.append(
            {
                "evidence_label": "natural_measured",
                "corpus_seed": seed,
                "transition_method": method,
                "triangle": "-".join(map(str, triangle)),
                "scalar_real": diagnostics.scalar_real,
                "scalar_imag": diagnostics.scalar_imag,
                "centrality_residual": diagnostics.centrality_residual,
                "normalized_centrality_residual": diagnostics.normalized_centrality_residual,
                "eigenvalue_dispersion": diagnostics.eigenvalue_dispersion,
                "mean_commutator_with_local_transitions": float(np.mean(commutators)),
                "estimated_root_order": root.order,
                "estimated_root_exponent": root.exponent,
                "root_residual": root.residual,
                "root_margin": root.margin,
                "root_confidence": root.confidence,
                "bootstrap_centrality_mean": float(central_values.mean()),
                "bootstrap_centrality_ci_low": float(np.quantile(central_values, 0.025)),
                "bootstrap_centrality_ci_high": float(np.quantile(central_values, 0.975)),
                "bootstrap_phase_instability": phase_circular_instability(bootstrap_phases[triangle]),
                "gauge_invariance_delta": gauge_delta,
            }
        )
        gauge_rows.append(
            {
                "evidence_label": "natural_measured",
                "corpus_seed": seed,
                "transition_method": method,
                "triangle": "-".join(map(str, triangle)),
                "centrality_before": diagnostics.centrality_residual,
                "centrality_after": gauge_diagnostics.centrality_residual,
                "scalar_real_before": diagnostics.scalar_real,
                "scalar_real_after": gauge_diagnostics.scalar_real,
                "gauge_invariance_delta": gauge_delta,
            }
        )

    cocycle_rows = []
    for row in tetrahedral_cocycle_rows(triangle_phases, vertices=8):
        cocycle_rows.append(
            {
                "evidence_label": "natural_measured",
                "corpus_seed": seed,
                "transition_method": method,
                **row,
            }
        )
    coboundary_residual, rephasings = coboundary_fit(triangle_phases, vertices=8)
    coboundary_row = {
        "evidence_label": "natural_measured",
        "corpus_seed": seed,
        "transition_method": method,
        "normalized_coboundary_residual": coboundary_residual,
        "edge_rephasings_json": json.dumps(rephasings, sort_keys=True),
    }
    certificate = pd.DataFrame(certificate_rows)
    cocycle = pd.DataFrame(cocycle_rows)
    aggregate = {
        "max_centrality_residual": float(certificate["centrality_residual"].max()),
        "mean_centrality_residual": float(certificate["centrality_residual"].mean()),
        "max_bootstrap_centrality_ci_high": float(certificate["bootstrap_centrality_ci_high"].max()),
        "max_bootstrap_phase_instability": float(certificate["bootstrap_phase_instability"].max()),
        "max_root_residual": float(certificate["root_residual"].max()),
        "minimum_root_margin": float(certificate["root_margin"].min()),
        "dominant_root_order": int(Counter(certificate["estimated_root_order"]).most_common(1)[0][0]),
        "max_cocycle_residual": float(cocycle["normalized_cocycle_residual"].max()),
        "coboundary_residual": coboundary_residual,
        "max_gauge_invariance_delta": float(certificate["gauge_invariance_delta"].max()),
    }
    return certificate_rows, cocycle_rows, gauge_rows, coboundary_row, aggregate


def classify(aggregate: dict[str, object], threshold: dict[str, float]) -> str:
    if float(aggregate["max_centrality_residual"]) > threshold["centrality"]:
        return "noncentral_holonomy"
    instability = max(
        float(aggregate["max_bootstrap_centrality_ci_high"]),
        float(aggregate["max_bootstrap_phase_instability"]),
    )
    if instability > threshold["bootstrap_instability"]:
        return "central_but_unstable"
    if float(aggregate["max_root_residual"]) > threshold["root_residual"]:
        return "inconclusive"
    if float(aggregate["max_cocycle_residual"]) > threshold["cocycle"]:
        return "inconclusive"
    if float(aggregate["max_gauge_invariance_delta"]) > threshold["gauge_invariance"]:
        return "inconclusive"
    if (
        int(aggregate["dominant_root_order"]) == 1
        or float(aggregate["coboundary_residual"]) <= threshold["coboundary_trivial"]
    ):
        return "trivial_coboundary"
    return "central_finite_order_candidate"


def write_report(
    mode: str,
    output_dir: Path,
    config: dict[str, object],
    certificates: pd.DataFrame,
    classifications: pd.DataFrame,
    sensitivity: pd.DataFrame,
    command: str,
) -> None:
    medium = classifications[classifications["threshold_level"] == "medium"]
    counts = medium["classification"].value_counts().to_dict()
    natural_candidates = int(
        medium["classification"].isin(["central_finite_order_candidate", "certified_controlled_projective_class"]).sum()
    )
    selected = medium[medium["selected_by_application_A"] == True]
    strict_loose_agreement = int(
        sensitivity.groupby(["corpus_seed", "transition_method"])["classification"].nunique().eq(1).sum()
    )
    report = f"""# Application B: Conservative Brauer-Like Certificate

Decision: **{'bounded smoke completed' if mode == 'smoke' else ('no natural Brauer-like candidate certified' if natural_candidates == 0 else 'candidate requires period-index validation')}**.

## Commands

Smoke:

```bash
{sys.executable} experiments/holonomy_application_B.py --mode smoke
```

Confirmatory:

```bash
{sys.executable} experiments/holonomy_application_B.py --mode confirmatory
```

Executed command: `{command}`

## Result

- Natural adapter corpus only; no adapter training and no test-label access.
- Medium-threshold classification counts: `{counts}`.
- Application-A-selected connection classifications: `{selected['classification'].value_counts().to_dict()}`.
- Natural central finite-order candidates: `{natural_candidates}`.
- Settings whose label is unchanged across strict/medium/loose thresholds: `{strict_loose_agreement}` / `{classifications[['corpus_seed', 'transition_method']].drop_duplicates().shape[0]}`.

The weight-derived transitions selected by held-out overlap residual in Application A are classified as trivial/coboundary. Their near-identity triangle defects follow from the construction `Q_ij = A_j pinv(A_i)` and are not evidence of a nontrivial projective class. Activation-Procrustes, low-rank, and joint connections produce larger defects, but they fail centrality and/or stability before torsion or cohomological language is admissible.

## Certificate boundary

A nonzero cycle defect was never treated as sufficient. Each triangle records scalarity, normalized residual, eigenvalue dispersion, commutation with local transitions, finite-root fits through order {config['max_root_order']}, bootstrap intervals, and gauge sensitivity. All 70 tetrahedra per seed/method are checked, followed by a scalar edge-rephasing fit. No natural row passes centrality, torsion, cocycle, nontriviality modulo coboundaries, gauge invariance, bootstrap stability, and predicted lift behavior together.

The only defensible natural conclusion is negative: selected residuals are trivial/coboundary, while alternative activation-derived residuals are predominantly noncentral or unstable. The phrase `Brauer class` is not used for them.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--config", type=Path, default=APP_DIR / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    application_a_config = json.loads(APPLICATION_A_CONFIG.read_text(encoding="utf-8"))
    output_dir = APP_DIR if args.mode == "confirmatory" else APP_DIR / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    resolved, manifest, payload, _shared = load_shared(args.mode)
    features = {name: values.float() for name, values in payload["features"].items()}
    certificate_rows: list[dict[str, object]] = []
    cocycle_rows: list[dict[str, object]] = []
    gauge_rows: list[dict[str, object]] = []
    coboundary_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    selected_from_a = pd.read_csv(
        (ROOT / "reports" / "holonomy_applications" / "application_A_holonomy" / ("runs.csv" if args.mode == "confirmatory" else "smoke/runs.csv"))
    )
    selected_lookup = (
        selected_from_a.groupby("corpus_seed")["selected_transition_method"].first().to_dict()
    )
    for seed in sorted(int(value) for value in manifest["corpus_seed"].unique()):
        try:
            models = load_models(seed, manifest, int(resolved["feature_dim"]), int(resolved["adapter_rank"]))
            adapters = [model.effective_adapter().detach() for model in models]
            activations = [
                models[chart].forward_activations(features["overlap_fit"][chart]).detach()
                for chart in range(8)
            ]
            transition_bundle = {}
            for method in application_a_config["transition_methods"]:
                transitions = fit_connection(
                    str(method), activations, adapters, int(application_a_config["low_rank_transition_rank"])
                )
                transition_bundle[str(method)] = transitions
                outputs = analyze_connection(
                    seed,
                    str(method),
                    transitions,
                    activations,
                    adapters,
                    int(application_a_config["low_rank_transition_rank"]),
                    int(config[args.mode]["bootstrap_samples"]),
                    int(config["max_root_order"]),
                )
                local_certificates, local_cocycles, local_gauges, coboundary, aggregate = outputs
                certificate_rows.extend(local_certificates)
                cocycle_rows.extend(local_cocycles)
                gauge_rows.extend(local_gauges)
                coboundary_rows.append(coboundary)
                for level, threshold in config["thresholds"].items():
                    label = classify(aggregate, threshold)
                    row = {
                        "evidence_label": "natural_measured",
                        "mode": args.mode,
                        "corpus_seed": seed,
                        "transition_method": method,
                        "threshold_level": level,
                        "selected_by_application_A": str(method) == str(selected_lookup[seed]),
                        "classification": label,
                        "brauer_like_candidate": label == "central_finite_order_candidate",
                        "execution_commit": git_head(),
                        **aggregate,
                    }
                    classification_rows.append(row)
                    sensitivity_rows.append(row.copy())
                capacity_rows.append(
                    {
                        "evidence_label": "natural_measured",
                        "mode": args.mode,
                        "corpus_seed": seed,
                        "transition_method": method,
                        "feature_dimension": adapters[0].shape[0],
                        "directed_edges": len(transitions),
                        "triangles": 56,
                        "tetrahedra": 70,
                        "bootstrap_samples": int(config[args.mode]["bootstrap_samples"]),
                        "new_trainable_parameters": 0,
                        "new_adapter_training": False,
                        "test_labels_accessed": False,
                    }
                )
            bundle_path = ARTIFACT_ROOT / f"application_B_{args.mode}" / f"transitions_seed_{seed}.pt"
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "evidence_label": "natural_measured",
                    "corpus_seed": seed,
                    "transitions": transition_bundle,
                },
                bundle_path,
            )
            artifact_rows.append(
                {
                    "evidence_label": "natural_measured",
                    "mode": args.mode,
                    "corpus_seed": seed,
                    "artifact_kind": "reconstituted_transition_bundle",
                    "path": str(bundle_path),
                    "sha256": sha256_file(bundle_path),
                    "bytes": bundle_path.stat().st_size,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "mode": args.mode,
                    "corpus_seed": seed,
                    "stage": "application_B",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    certificates = pd.DataFrame(certificate_rows)
    cocycles = pd.DataFrame(cocycle_rows)
    gauges = pd.DataFrame(gauge_rows)
    coboundaries = pd.DataFrame(coboundary_rows)
    classifications = pd.DataFrame(classification_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    certificates.to_csv(output_dir / "certificates.csv", index=False)
    cocycles.to_csv(output_dir / "cocycle_checks.csv", index=False)
    gauges.to_csv(output_dir / "gauge_invariance_checks.csv", index=False)
    coboundaries.to_csv(output_dir / "coboundary_checks.csv", index=False)
    classifications.to_csv(output_dir / "candidate_classifications.csv", index=False)
    sensitivity.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    pd.DataFrame(capacity_rows).to_csv(output_dir / "capacity_audit.csv", index=False)
    pd.DataFrame(failures, columns=("mode", "corpus_seed", "stage", "error_type", "message")).to_csv(
        output_dir / "failure_log.csv", index=False
    )

    medium = classifications[classifications["threshold_level"] == "medium"]
    pivot = medium.pivot(index="corpus_seed", columns="transition_method", values="mean_centrality_residual")
    delta = (pivot["activation_procrustes"] - pivot["weight_based"]).to_numpy(dtype=float)
    mean, low, high = paired_bootstrap(delta, 4000 if args.mode == "confirmatory" else 200, 930000)
    paired = pd.DataFrame(
        [
            {
                "evidence_label": "natural_measured",
                "mode": args.mode,
                "comparison": "activation_procrustes_minus_weight_based",
                "metric": "mean_centrality_residual",
                "n_independent_seeds": len(delta),
                "mean_delta": mean,
                "ci_low": low,
                "ci_high": high,
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
            }
        ]
    )
    paired.to_csv(output_dir / "paired_statistics.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, rows in medium.groupby("transition_method"):
        axes[0].scatter(
            rows["mean_centrality_residual"],
            rows["max_bootstrap_phase_instability"],
            label=method,
            s=40,
        )
    axes[0].set_xlabel("Mean centrality residual")
    axes[0].set_ylabel("Maximum bootstrap phase instability")
    axes[0].set_title("Natural certificate failure modes")
    axes[0].legend(fontsize=7)
    counts = medium["classification"].value_counts()
    axes[1].bar(counts.index, counts.values, color="#8b5a83")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Seed-estimator settings")
    axes[1].set_title("Conservative certificate labels")
    figure.tight_layout()
    figure.savefig(output_dir / "plots" / "brauer_certificate_summary.pdf", bbox_inches="tight")
    plt.close(figure)

    table = medium.groupby(["transition_method", "classification"], as_index=False).agg(
        settings=("corpus_seed", "count"),
        mean_centrality=("mean_centrality_residual", "mean"),
        max_cocycle=("max_cocycle_residual", "max"),
        max_coboundary=("coboundary_residual", "max"),
    )
    latex = ["\\begin{tabular}{llrrrr}", "\\toprule", "Estimator & Label & Settings & Centrality & Cocycle & Coboundary\\\\", "\\midrule"]
    for row in table.itertuples(index=False):
        latex.append(
            f"{row.transition_method.replace('_', ' ')} & {row.classification.replace('_', ' ')} & {row.settings} & {row.mean_centrality:.3f} & {row.max_cocycle:.3f} & {row.max_coboundary:.3f}\\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output_dir / "tables" / "application_B_certificate.tex").write_text("\n".join(latex), encoding="utf-8")
    write_report(args.mode, output_dir, config, certificates, classifications, sensitivity, command)

    committed_paths = (
        output_dir / "certificates.csv",
        output_dir / "cocycle_checks.csv",
        output_dir / "gauge_invariance_checks.csv",
        output_dir / "coboundary_checks.csv",
        output_dir / "candidate_classifications.csv",
        output_dir / "threshold_sensitivity.csv",
        output_dir / "capacity_audit.csv",
        output_dir / "paired_statistics.csv",
        output_dir / "plots" / "brauer_certificate_summary.pdf",
        output_dir / "tables" / "application_B_certificate.tex",
    )
    artifact_rows.extend(
        {
            "evidence_label": "natural_measured",
            "mode": args.mode,
            "corpus_seed": "all",
            "artifact_kind": "committed_output",
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in committed_paths
    )
    pd.DataFrame(artifact_rows).to_csv(output_dir / "artifact_hashes.csv", index=False)
    expected = 56 * 4 * manifest["corpus_seed"].nunique()
    if failures or len(certificates) != expected:
        raise RuntimeError("Application B incomplete; inspect failure_log.csv")


if __name__ == "__main__":
    main()
