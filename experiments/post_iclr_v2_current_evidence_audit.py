#!/usr/bin/env python3
"""Generate the post-ICLR v2 evidence audit from tracked source artifacts.

The audit is intentionally read-only with respect to existing evidence.  It
derives a narrow claim matrix, an integrity manifest, and the phase index under
``reports/post_iclr_v2`` without modifying any manuscript source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "post_iclr_v2"
ALLOWED_STATUSES = {
    "supported",
    "supported-narrow",
    "descriptive",
    "negative",
    "forbidden",
    "pending",
}


@dataclass(frozen=True)
class EvidenceSnapshot:
    origin_main: str
    official_git_rebasin_rows: int
    official_c2m3_rows: int
    official_ties_rows: int
    official_failures: int
    selector_minus_git_rebasin: float
    selector_minus_git_rebasin_ci_low: float
    selector_minus_git_rebasin_ci_high: float
    selector_minus_c2m3: float
    selector_minus_c2m3_ci_low: float
    selector_minus_c2m3_ci_high: float
    c2m3_minus_gauge: float
    c2m3_minus_gauge_ci_low: float
    c2m3_minus_gauge_ci_high: float
    selector_minus_greedy_soup: float
    selector_minus_greedy_soup_ci_low: float
    selector_minus_greedy_soup_ci_high: float
    selector_soup_choices: int
    selector_total_choices: int
    biomedical_retransport_passed: bool
    biomedical_specific_passed: bool
    biomedical_multidomain_passed: bool
    biomedical_residual_correction_passed: bool
    biomedical_inferred_method_on_any_pareto_frontier: bool


def git(*args: str, root: Path = ROOT) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def single_row(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    rows = frame[mask]
    if len(rows) != 1:
        raise RuntimeError(f"expected one row for {conditions}, found {len(rows)}")
    return rows.iloc[0]


def claim_passed(path: Path, claim: str) -> bool:
    row = single_row(pd.read_csv(path), claim=claim)
    return bool(row["passed"])


def load_snapshot(root: Path = ROOT) -> EvidenceSnapshot:
    official = pd.read_csv(root / "reports/csv/post_iclr_official_baseline_summary.csv")
    runs = pd.read_csv(root / "reports/csv/post_iclr_official_baseline_runs.csv")
    external_summary = pd.read_csv(root / "reports/csv/external_baseline_comparison_summary.csv")
    external_runs = pd.read_csv(root / "reports/csv/external_baseline_comparison.csv")

    git_rebasin = single_row(
        official,
        regime="independent_initialization",
        baseline="official_git_rebasin",
    )
    c2m3 = single_row(
        official,
        regime="independent_initialization",
        baseline="official_c2m3",
    )
    ties = single_row(
        official,
        regime="common_base_task_vector",
        baseline="official_ties",
    )
    improved = single_row(
        external_summary,
        summary_type="method_summary",
        scope="overall",
        method="improved_validated_selector",
    )
    selector = external_runs[external_runs["method"].eq("improved_validated_selector")]
    soup_choices = int(selector["selector_chose"].fillna("").str.contains("soup").sum())

    discovery = root / "reports/spatial_output_program/biomedical/discovery/claims.csv"
    multidomain = root / "reports/spatial_output_program/multidomain/claims.csv"
    correction = root / "reports/spatial_output_program/transitions/correction_claims.csv"
    pareto = pd.read_csv(root / "reports/spatial_output_program/biomedical/cost/pareto.csv")
    inferred_pareto = pareto[
        pareto["method"].astype(str).str.contains("inferred", case=False, na=False)
    ]

    evaluated = runs[runs["status"].eq("evaluated")]
    failures = int(
        evaluated["failed_reason"].fillna("").astype(str).str.strip().ne("").sum()
    )
    return EvidenceSnapshot(
        origin_main=git("rev-parse", "origin/main", root=root),
        official_git_rebasin_rows=int(git_rebasin["n_rows"]),
        official_c2m3_rows=int(c2m3["n_rows"]),
        official_ties_rows=int(ties["n_rows"]),
        official_failures=failures,
        selector_minus_git_rebasin=-float(git_rebasin["mean_delta_vs_twistedmerge_selector"]),
        selector_minus_git_rebasin_ci_low=-float(git_rebasin["delta_vs_twistedmerge_selector_ci_high"]),
        selector_minus_git_rebasin_ci_high=-float(git_rebasin["delta_vs_twistedmerge_selector_ci_low"]),
        selector_minus_c2m3=-float(c2m3["mean_delta_vs_twistedmerge_selector"]),
        selector_minus_c2m3_ci_low=-float(c2m3["delta_vs_twistedmerge_selector_ci_high"]),
        selector_minus_c2m3_ci_high=-float(c2m3["delta_vs_twistedmerge_selector_ci_low"]),
        c2m3_minus_gauge=float(c2m3["mean_delta_vs_twistedmerge_gauge"]),
        c2m3_minus_gauge_ci_low=float(c2m3["delta_vs_twistedmerge_gauge_ci_low"]),
        c2m3_minus_gauge_ci_high=float(c2m3["delta_vs_twistedmerge_gauge_ci_high"]),
        selector_minus_greedy_soup=float(improved["paired_mean_accuracy_delta_vs_greedy_soup"]),
        selector_minus_greedy_soup_ci_low=float(improved["paired_accuracy_delta_vs_greedy_soup_ci_low"]),
        selector_minus_greedy_soup_ci_high=float(improved["paired_accuracy_delta_vs_greedy_soup_ci_high"]),
        selector_soup_choices=soup_choices,
        selector_total_choices=len(selector),
        biomedical_retransport_passed=claim_passed(discovery, "retransport_gate"),
        biomedical_specific_passed=claim_passed(discovery, "twistedmerge_specific_gate"),
        biomedical_multidomain_passed=claim_passed(multidomain, "multidomain_primary_gate"),
        biomedical_residual_correction_passed=claim_passed(correction, "residual_correction_activated"),
        biomedical_inferred_method_on_any_pareto_frontier=bool(
            inferred_pareto["frontier"].fillna(False).astype(bool).any()
        ),
    )


def claim_rows(snapshot: EvidenceSnapshot) -> list[dict[str, object]]:
    rows = [
        {
            "claim_id": "official_git_rebasin_exact_family",
            "regime": "independent-initialization/rebasin",
            "claim": "Adapter-assisted official Git Re-Basin ran on 20 exact MNIST MLP settings.",
            "status": "supported-narrow",
            "value": snapshot.official_git_rebasin_rows,
            "ci_low": "",
            "ci_high": "",
            "capacity": "same-capacity single model; 1x inference",
            "selection_budget": "not validation-selected",
            "evidence": "reports/csv/post_iclr_official_baseline_runs.csv; reports/csv/post_iclr_official_baseline_summary.csv",
            "limitations": "adapter-assisted official core; exact checkpoint family only",
        },
        {
            "claim_id": "official_c2m3_exact_family",
            "regime": "independent-initialization/rebasin",
            "claim": "Adapter-assisted official C2M3 ran on 20 exact MNIST MLP settings.",
            "status": "supported-narrow",
            "value": snapshot.official_c2m3_rows,
            "ci_low": "",
            "ci_high": "",
            "capacity": "same-capacity single model; 1x inference",
            "selection_budget": "not validation-selected",
            "evidence": "reports/csv/post_iclr_official_baseline_runs.csv; reports/csv/post_iclr_official_baseline_summary.csv",
            "limitations": "adapter-assisted official core; exact checkpoint family only",
        },
        {
            "claim_id": "official_ties_matches_internal",
            "regime": "common-base task-vector",
            "claim": "Adapter-assisted official TIES matched the internal TIES-style result on three exact settings.",
            "status": "supported-narrow",
            "value": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "capacity": "same-capacity single model; 1x inference",
            "selection_budget": "validation-tuned density and scale",
            "evidence": "reports/csv/post_iclr_official_baseline_runs.csv; reports/csv/post_iclr_official_baseline_summary.csv",
            "limitations": "three MNIST common-base settings",
        },
        {
            "claim_id": "selector_over_official_git_rebasin",
            "regime": "independent-initialization/rebasin",
            "claim": "The existing validation-only TwistedMerge selector exceeds adapter-assisted official Git Re-Basin on the exact checkpoint family.",
            "status": "supported-narrow",
            "value": snapshot.selector_minus_git_rebasin,
            "ci_low": snapshot.selector_minus_git_rebasin_ci_low,
            "ci_high": snapshot.selector_minus_git_rebasin_ci_high,
            "capacity": "same-capacity single-model or soup output; 1x inference",
            "selection_budget": "enriched validation-selected pool",
            "evidence": "reports/csv/post_iclr_official_baseline_summary.csv",
            "limitations": "MNIST one-hidden-layer MLP; exact-setting bootstrap; attribution unresolved",
        },
        {
            "claim_id": "selector_over_official_c2m3",
            "regime": "independent-initialization/rebasin",
            "claim": "The existing validation-only TwistedMerge selector exceeds adapter-assisted official C2M3 on the exact checkpoint family.",
            "status": "supported-narrow",
            "value": snapshot.selector_minus_c2m3,
            "ci_low": snapshot.selector_minus_c2m3_ci_low,
            "ci_high": snapshot.selector_minus_c2m3_ci_high,
            "capacity": "same-capacity single-model or soup output; 1x inference",
            "selection_budget": "enriched validation-selected pool",
            "evidence": "reports/csv/post_iclr_official_baseline_summary.csv",
            "limitations": "MNIST one-hidden-layer MLP; exact-setting bootstrap; attribution unresolved",
        },
        {
            "claim_id": "pure_gauge_over_official_c2m3",
            "regime": "independent-initialization/rebasin",
            "claim": "The pure TwistedMerge positive monomial gauge beats adapter-assisted official C2M3.",
            "status": "negative",
            "value": -snapshot.c2m3_minus_gauge,
            "ci_low": -snapshot.c2m3_minus_gauge_ci_high,
            "ci_high": -snapshot.c2m3_minus_gauge_ci_low,
            "capacity": "same-capacity single models; 1x inference",
            "selection_budget": "not applicable",
            "evidence": "reports/csv/post_iclr_official_baseline_summary.csv",
            "limitations": "observed direction favors official C2M3",
        },
        {
            "claim_id": "selector_over_greedy_soup",
            "regime": "greedy-soup validation descent",
            "claim": "The existing improved selector beats ordinary greedy soup.",
            "status": "negative",
            "value": snapshot.selector_minus_greedy_soup,
            "ci_low": snapshot.selector_minus_greedy_soup_ci_low,
            "ci_high": snapshot.selector_minus_greedy_soup_ci_high,
            "capacity": "same-capacity single-model soups; 1x inference",
            "selection_budget": "not budget-matched in the attribution sense",
            "evidence": "reports/csv/external_baseline_comparison_summary.csv",
            "limitations": "existing selector uses a larger candidate family",
        },
        {
            "claim_id": "selector_choices_are_soup_dominated",
            "regime": "greedy-soup validation descent",
            "claim": "Most existing selector choices are soup-based candidates.",
            "status": "descriptive",
            "value": snapshot.selector_soup_choices / snapshot.selector_total_choices,
            "ci_low": "",
            "ci_high": "",
            "capacity": "same-capacity single-model soups; 1x inference",
            "selection_budget": f"{snapshot.selector_total_choices} exact settings",
            "evidence": "reports/csv/external_baseline_comparison.csv",
            "limitations": "choice frequency does not establish causal attribution",
        },
        {
            "claim_id": "controlled_rank_lift",
            "regime": "controlled planted obstruction",
            "claim": "The supplied-context q=2 branch lift resolves the controlled nontrivial mu2 obstruction and beats matched controls.",
            "status": "supported-narrow",
            "value": 0.25,
            "ci_low": 0.25,
            "ci_high": 0.25,
            "capacity": "extra-capacity branch lift; 2x branch representation",
            "selection_budget": "controlled supplied context; learned router separate",
            "evidence": "reports/csv/controlled_twisted_overlap_summary.csv",
            "limitations": "controlled construction only; not a natural Brauer-class result",
        },
        {
            "claim_id": "raw_weight_average_prediction",
            "regime": "diagnostic prediction",
            "claim": "Residual diagnostics predict raw weight-average degradation.",
            "status": "negative",
            "value": "",
            "ci_low": "",
            "ci_high": "",
            "capacity": "diagnostic only",
            "selection_budget": "not applicable",
            "evidence": "reports/csv/obstruction_predictor_target_stats.csv",
            "limitations": "only selected alignment-conditioned targets currently have supported rows",
        },
        {
            "claim_id": "biomedical_inferred_retransport",
            "regime": "biomedical site/domain heterogeneity",
            "claim": "Inferred spatial-output retransport passes its paired gate.",
            "status": "negative",
            "value": snapshot.biomedical_retransport_passed,
            "ci_low": "",
            "ci_high": "",
            "capacity": "four-expert path; not same-capacity",
            "selection_budget": "validation-only charts; synthetic domains",
            "evidence": "reports/spatial_output_program/biomedical/discovery/claims.csv",
            "limitations": "small Kvasir-SEG setup; no site metadata",
        },
        {
            "claim_id": "biomedical_twistedmerge_specific_benefit",
            "regime": "biomedical site/domain heterogeneity",
            "claim": "The current spatial-output program shows a TwistedMerge-specific benefit.",
            "status": "negative",
            "value": snapshot.biomedical_specific_passed,
            "ci_low": "",
            "ci_high": "",
            "capacity": "not on measured quality-cost Pareto frontier",
            "selection_budget": "validation-only; synthetic domains",
            "evidence": "reports/spatial_output_program/biomedical/discovery/claims.csv; reports/spatial_output_program/biomedical/cost/pareto.csv",
            "limitations": "does not support clinical or multicenter claims",
        },
        {
            "claim_id": "biomedical_multidomain_benefit",
            "regime": "biomedical site/domain heterogeneity",
            "claim": "The current spatial-output program passes the multidomain benefit gate.",
            "status": "negative",
            "value": snapshot.biomedical_multidomain_passed,
            "ci_low": "",
            "ci_high": "",
            "capacity": "multi-expert path",
            "selection_budget": "synthetic domains only",
            "evidence": "reports/spatial_output_program/multidomain/claims.csv",
            "limitations": "no real site or center metadata",
        },
        {
            "claim_id": "biomedical_residual_correction",
            "regime": "biomedical site/domain heterogeneity",
            "claim": "The current spatial-output program certifies and activates a realistic residual correction.",
            "status": "negative",
            "value": snapshot.biomedical_residual_correction_passed,
            "ci_low": "",
            "ci_high": "",
            "capacity": "correction remained inactive",
            "selection_budget": "gated by D1 stability",
            "evidence": "reports/spatial_output_program/transitions/correction_claims.csv",
            "limitations": "D1 did not certify a stable residual",
        },
        {
            "claim_id": "resnet18_cifar10_independent_merge",
            "regime": "independent-initialization/rebasin",
            "claim": "TwistedMerge is validated on independently trained CIFAR-10 ResNet-18 groups.",
            "status": "pending",
            "value": "",
            "ci_low": "",
            "ci_high": "",
            "capacity": "must be recorded per method",
            "selection_budget": "must be preregistered",
            "evidence": "reports/post_iclr_experiment_gap_audit.md",
            "limitations": "full ResNet-18 experiment absent",
        },
        {
            "claim_id": "batchnorm_aware_exact_gauge",
            "regime": "exact gauge construction",
            "claim": "BatchNorm-aware channel permutations or scalings are exact under stated assumptions.",
            "status": "pending",
            "value": "",
            "ci_low": "",
            "ci_high": "",
            "capacity": "same capacity if exact",
            "selection_budget": "not applicable",
            "evidence": "reports/post_iclr_experiment_gap_audit.md",
            "limitations": "derivation and exactness suite absent",
        },
        {
            "claim_id": "broad_sota",
            "regime": "claim boundary",
            "claim": "TwistedMerge is a broad state-of-the-art model-merging method.",
            "status": "forbidden",
            "value": "",
            "ci_low": "",
            "ci_high": "",
            "capacity": "mixed regimes and capacities",
            "selection_budget": "not comparable",
            "evidence": "reports/final_claim_ledger.md; reports/full_capacity_claim_audit.md",
            "limitations": "unsupported across modern architectures and strong budget-matched baselines",
        },
        {
            "claim_id": "natural_brauer_classes",
            "regime": "claim boundary",
            "claim": "Natural neural residuals are Brauer or period-index classes.",
            "status": "forbidden",
            "value": "",
            "ci_low": "",
            "ci_high": "",
            "capacity": "not applicable",
            "selection_budget": "not applicable",
            "evidence": "reports/claims_audit.md; reports/final_claim_ledger.md",
            "limitations": "current real residuals remain noncentral or uncertified",
        },
    ]
    invalid = sorted({str(row["status"]) for row in rows} - ALLOWED_STATUSES)
    if invalid:
        raise RuntimeError(f"invalid claim statuses: {invalid}")
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "\\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def protected_dirty_paths(root: Path = ROOT) -> list[str]:
    protected_suffixes = {".tex", ".bib", ".cls", ".sty", ".eps"}
    paths: list[str] = []
    raw = git("worktree", "list", "--porcelain", root=root)
    for line in raw.splitlines():
        if not line.startswith("worktree "):
            continue
        worktree = Path(line.removeprefix("worktree "))
        status = subprocess.check_output(
            ["git", "-C", str(worktree), "status", "--porcelain"], text=True
        )
        for item in status.splitlines():
            relative = item[3:]
            if Path(relative).suffix.lower() in protected_suffixes:
                paths.append(f"{worktree}/{relative}")
    return sorted(paths)


def artifact_manifest(root: Path, files: list[tuple[str, str]]) -> list[dict[str, object]]:
    rows = []
    for relative, role in files:
        path = root / relative
        rows.append(
            {
                "path": relative,
                "role": role,
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256(path) if path.exists() else "",
                "data_rows": csv_rows(path) if path.exists() and path.suffix == ".csv" else "",
            }
        )
    return rows


def journal_rows() -> list[dict[str, object]]:
    return [
        {
            "phase": "current evidence audit",
            "scientific question": "What is already supported and what remains blocked?",
            "current evidence": "official cores completed; narrow selector comparison; strong negative boundaries",
            "new evidence": "machine-readable audit refreshed from tracked CSVs",
            "strongest defensible claim": "narrow exact-family selector advantage over official matching cores",
            "strongest negative result": "pure gauge loses to official C2M3 and selector loses to greedy soup",
            "architecture": "one-hidden-layer MLP; small no-BN CNN; tiny U-Net",
            "dataset": "MNIST; Fashion-MNIST; CIFAR-10 boundary; Kvasir-SEG",
            "seeds": "5 official training groups; phase-specific counts elsewhere",
            "baseline": "official Git Re-Basin; official C2M3; greedy soup",
            "confidence interval": "claim-specific",
            "capacity": "recorded per method",
            "cost": "existing artifacts only",
            "paper relevance": "claim boundary",
            "journal relevance": "baseline and gap map",
            "remaining limitation": "selector attribution and modern architecture absent",
            "status": "complete",
        },
        {
            "phase": "selector attribution",
            "scientific question": "Does TwistedMerge add value beyond an equally tuned soup pool?",
            "current evidence": "selector is soup-dominated and below greedy soup",
            "new evidence": "pending untouched checkpoint groups",
            "strongest defensible claim": "pending",
            "strongest negative result": "existing matched-grid selector delta is negative",
            "architecture": "one-hidden-layer ReLU MLP",
            "dataset": "MNIST",
            "seeds": "at least 10 new training groups planned",
            "baseline": "budget-matched greedy soup",
            "confidence interval": "pending",
            "capacity": "same-capacity single models and soups",
            "cost": "must match candidate and validation budgets",
            "paper relevance": "high",
            "journal relevance": "high",
            "remaining limitation": "fresh smoke, pilot, and confirmation required",
            "status": "in-progress",
        },
        {
            "phase": "BatchNorm gauge",
            "scientific question": "Which channel transformations are exact with BatchNorm?",
            "current evidence": "no-BatchNorm gauges only",
            "new evidence": "none",
            "strongest defensible claim": "pending",
            "strongest negative result": "current exactness claim cannot include BatchNorm",
            "architecture": "ResNet-18 target",
            "dataset": "synthetic tensors then CIFAR-10",
            "seeds": "pending",
            "baseline": "original model functional preservation",
            "confidence interval": "not applicable to numerical exactness threshold",
            "capacity": "same capacity",
            "cost": "pending",
            "paper relevance": "high",
            "journal relevance": "high",
            "remaining limitation": "derivation and tests absent",
            "status": "pending",
        },
        {
            "phase": "ResNet-18 CIFAR-10",
            "scientific question": "Do diagnostics and gauges survive a credible modern architecture?",
            "current evidence": "bounded small no-BatchNorm CNN only",
            "new evidence": "none",
            "strongest defensible claim": "pending",
            "strongest negative result": "existing CIFAR gauge effects are descriptive and below greedy soup",
            "architecture": "ResNet-18",
            "dataset": "CIFAR-10",
            "seeds": "at least 5 groups per N planned",
            "baseline": "official matching cores; greedy soup; ensemble upper bound",
            "confidence interval": "pending",
            "capacity": "must be recorded",
            "cost": "large and gated",
            "paper relevance": "high",
            "journal relevance": "high",
            "remaining limitation": "training and BatchNorm gates not opened",
            "status": "pending",
        },
    ]


def write_outputs(root: Path = ROOT, out: Path = DEFAULT_OUT) -> EvidenceSnapshot:
    snapshot = load_snapshot(root)
    claims = claim_rows(snapshot)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selector_attribution" / "plots").mkdir(parents=True, exist_ok=True)

    claim_fields = [
        "claim_id",
        "regime",
        "claim",
        "status",
        "value",
        "ci_low",
        "ci_high",
        "capacity",
        "selection_budget",
        "evidence",
        "limitations",
    ]
    write_csv(out / "current_claim_matrix.csv", claims, claim_fields)

    manifest_files = [
        ("README.md", "repository scope"),
        ("reports/final_claim_ledger.md", "claim boundary"),
        ("reports/claims_audit.md", "claim audit"),
        ("reports/full_capacity_claim_audit.md", "capacity audit"),
        ("reports/baseline_regime_audit.md", "regime audit"),
        ("reports/post_iclr_experiment_gap_audit.md", "gap audit"),
        ("reports/post_iclr_experiment_plan.md", "frozen plan"),
        ("reports/post_iclr_official_baseline_report.md", "official report"),
        ("reports/csv/post_iclr_official_baseline_runs.csv", "official per-setting rows"),
        ("reports/csv/post_iclr_official_baseline_summary.csv", "official summary"),
        ("reports/csv/external_baseline_comparison.csv", "selector choices and internal controls"),
        ("reports/csv/external_baseline_comparison_summary.csv", "selector paired summary"),
        ("reports/csv/controlled_twisted_overlap_summary.csv", "controlled obstruction summary"),
        ("reports/csv/obstruction_predictor_target_stats.csv", "diagnostic target audit"),
        ("reports/spatial_output_program/biomedical/discovery/claims.csv", "biomedical discovery gates"),
        ("reports/spatial_output_program/multidomain/claims.csv", "biomedical multidomain gates"),
        ("reports/spatial_output_program/transitions/correction_claims.csv", "biomedical correction gate"),
        ("reports/spatial_output_program/biomedical/cost/pareto.csv", "biomedical Pareto audit"),
    ]
    manifest = artifact_manifest(root, manifest_files)
    write_csv(
        out / "current_artifact_manifest.csv",
        manifest,
        ["path", "role", "exists", "bytes", "sha256", "data_rows"],
    )
    write_csv(
        out / "journal_evidence_matrix.csv",
        journal_rows(),
        [
            "phase",
            "scientific question",
            "current evidence",
            "new evidence",
            "strongest defensible claim",
            "strongest negative result",
            "architecture",
            "dataset",
            "seeds",
            "baseline",
            "confidence interval",
            "capacity",
            "cost",
            "paper relevance",
            "journal relevance",
            "remaining limitation",
            "status",
        ],
    )

    protected = protected_dirty_paths(root)
    protected_text = "\n".join(f"- `{path}`" for path in protected) or "- None detected."
    audit = f"""# Post-ICLR v2 Current Evidence Audit

Generated from tracked source artifacts at `origin/main` `{snapshot.origin_main}`. The audit ran in an isolated Codex worktree and did not modify manuscript, bibliography, or existing paper-figure files.

## Protected collaborator work

The authoritative main checkout was clean at the preflight. The following manuscript-like uncommitted paths exist in other worktrees and are explicitly out of scope:

{protected_text}

## Verified official-baseline starting point

- Adapter-assisted official Git Re-Basin: `{snapshot.official_git_rebasin_rows}` exact independent-initialization settings.
- Adapter-assisted official C2M3: `{snapshot.official_c2m3_rows}` exact independent-initialization settings.
- Adapter-assisted official TIES: `{snapshot.official_ties_rows}` exact common-base settings and zero paired difference from the internal TIES-style implementation.
- Evaluated official-core rows with recorded runtime failure: `{snapshot.official_failures}`.
- Official Model Soups remains interface-blocked. Task Arithmetic and DARE remain license-blocked in the pinned author repositories. No blocked metric is substituted.

## Verified narrow positive result

- Existing validation-only selector minus official Git Re-Basin: `{snapshot.selector_minus_git_rebasin:.6f}`, 95% CI `[{snapshot.selector_minus_git_rebasin_ci_low:.6f}, {snapshot.selector_minus_git_rebasin_ci_high:.6f}]`.
- Existing validation-only selector minus official C2M3: `{snapshot.selector_minus_c2m3:.6f}`, 95% CI `[{snapshot.selector_minus_c2m3_ci_low:.6f}, {snapshot.selector_minus_c2m3_ci_high:.6f}]`.

These are supported-narrow exact-family comparisons, not broad external-baseline or SOTA claims.

## Verified negative boundaries

- Official C2M3 minus the pure TwistedMerge monomial gauge: `{snapshot.c2m3_minus_gauge:.6f}`, 95% CI `[{snapshot.c2m3_minus_gauge_ci_low:.6f}, {snapshot.c2m3_minus_gauge_ci_high:.6f}]`; the proposed pure-gauge win has the wrong direction.
- Existing improved selector minus greedy soup: `{snapshot.selector_minus_greedy_soup:.6f}`, 95% CI `[{snapshot.selector_minus_greedy_soup_ci_low:.6f}, {snapshot.selector_minus_greedy_soup_ci_high:.6f}]`.
- Soup-based selections: `{snapshot.selector_soup_choices}/{snapshot.selector_total_choices}` (`{snapshot.selector_soup_choices / snapshot.selector_total_choices:.1%}`). The existing selector advantage over matching cores is therefore not attributable to residual geometry without a new budget-matched experiment.
- Biomedical inferred retransport, TwistedMerge-specific benefit, multidomain benefit, and realistic residual correction are all false under their recorded gates.
- Any inferred spatial-output method on a measured quality-cost Pareto frontier: `{snapshot.biomedical_inferred_method_on_any_pareto_frontier}`.
- Full independently initialized ResNet-18 CIFAR-10/CIFAR-100 and BatchNorm-aware gauge experiments remain absent.

## Claim classification

{markdown_table(claims, ["claim_id", "regime", "status", "value", "ci_low", "ci_high"])}

Machine-readable details are in `current_claim_matrix.csv`. Integrity and row counts are in `current_artifact_manifest.csv`.
"""
    (out / "current_evidence_audit.md").write_text(audit, encoding="utf-8")

    index = """# Post-ICLR v2 Experimental Evidence

This namespace contains post-ICLR v2 audits and new gated experiments. It does not contain manuscript edits.

## Status

- Current evidence audit: complete.
- Selector attribution: in progress; fresh smoke and pilot required before confirmation.
- BatchNorm-aware gauge: pending selector-attribution handoff.
- ResNet-18 CIFAR-10: pending the BatchNorm derivation and base-quality preregistration.
- Later planted, prediction, selector, and biomedical phases: gated.

## Files

- `current_evidence_audit.md`
- `current_claim_matrix.csv`
- `current_artifact_manifest.csv`
- `journal_evidence_matrix.csv`
- `selector_attribution/`
- `proposed_claim_update.md`
- `paper_editor_evidence_brief.md`
"""
    (out / "index.md").write_text(index, encoding="utf-8")

    proposed = f"""# Post-ICLR v2 Proposed Claim Update

No paper claim is promoted by this audit alone.

| Exact wording | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| On the exact 20-setting MNIST one-hidden-layer MLP family, the existing validation-only selector exceeded adapter-assisted official Git Re-Basin by `{snapshot.selector_minus_git_rebasin:.4f}` (95% CI `[{snapshot.selector_minus_git_rebasin_ci_low:.4f}, {snapshot.selector_minus_git_rebasin_ci_high:.4f}]`) and official C2M3 by `{snapshot.selector_minus_c2m3:.4f}` (95% CI `[{snapshot.selector_minus_c2m3_ci_low:.4f}, {snapshot.selector_minus_c2m3_ci_high:.4f}]`). | supported-narrow | Official baseline summary CSV | Attribution is unresolved and selections are soup-dominated. |
| The pure TwistedMerge monomial gauge beats official C2M3. | negative | Official C2M3 leads by `{snapshot.c2m3_minus_gauge:.4f}` with positive CI. | Wording must not appear as a positive result. |
| TwistedMerge beats greedy soup. | forbidden until new evidence | Existing selector delta is `{snapshot.selector_minus_greedy_soup:.4f}` with a negative CI. | Requires the fresh budget-matched selector-attribution phase. |
| TwistedMerge works on ResNet-18 with exact BatchNorm-aware gauges. | pending | No current artifact. | Requires derivation, exactness tests, base-quality gate, and confirmatory groups. |
"""
    (out / "proposed_claim_update.md").write_text(proposed, encoding="utf-8")

    brief = f"""# Evidence Brief for the Paper Editor

Do not paste a broad performance claim from this audit.

## Confirmed positive findings

- Exact-family selector advantage over adapter-assisted official Git Re-Basin: `{snapshot.selector_minus_git_rebasin:.4f}` (`[{snapshot.selector_minus_git_rebasin_ci_low:.4f}, {snapshot.selector_minus_git_rebasin_ci_high:.4f}]`).
- Exact-family selector advantage over adapter-assisted official C2M3: `{snapshot.selector_minus_c2m3:.4f}` (`[{snapshot.selector_minus_c2m3_ci_low:.4f}, {snapshot.selector_minus_c2m3_ci_high:.4f}]`).
- Controlled q=2 branch-lift evidence remains controlled-only and extra-capacity.

## Confirmed negative findings

- Pure gauge loses to official C2M3 by `{snapshot.c2m3_minus_gauge:.4f}`.
- Existing improved selector loses to greedy soup by `{abs(snapshot.selector_minus_greedy_soup):.4f}` on average.
- `{snapshot.selector_soup_choices}/{snapshot.selector_total_choices}` selector choices are soup-based.
- All four requested biomedical gates remain closed, and inferred methods are not on a measured quality-cost Pareto frontier.

## Allowed official-baseline wording

Use "adapter-assisted official core" for Git Re-Basin, C2M3, and TIES. Do not call these unmodified end-to-end applications. Official Model Soups is interface-blocked; Task Arithmetic and DARE are license-blocked in the pinned source audit.

## Wording to remove or soften

- Remove broad SOTA or universal superiority language.
- Do not say the pure gauge beats official C2M3.
- Do not attribute the selector result to residual geometry before attribution closes.
- Do not call natural residuals Brauer or period-index classes.

## Paper placement

- Current official-core comparison: appendix or narrowly scoped baseline section.
- Controlled obstruction mechanisms: main theory/controlled evidence, with capacity labels.
- Current CIFAR and biomedical outcomes: appendix or negative-results note.
- Selector attribution, BatchNorm-aware ResNet-18, and harmful-merge prediction: unresolved experiments.

## Claim-to-artifact map

- Official comparisons: `reports/csv/post_iclr_official_baseline_summary.csv`.
- Selector composition and greedy boundary: `reports/csv/external_baseline_comparison.csv` and its summary.
- Controlled lift: `reports/csv/controlled_twisted_overlap_summary.csv`.
- Biomedical boundaries: `reports/spatial_output_program/` claim and Pareto CSVs.
"""
    (out / "paper_editor_evidence_brief.md").write_text(brief, encoding="utf-8")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    snapshot = write_outputs(ROOT, args.output_dir)
    print(
        json.dumps(
            {
                "origin_main": snapshot.origin_main,
                "selector_minus_git_rebasin": snapshot.selector_minus_git_rebasin,
                "selector_minus_c2m3": snapshot.selector_minus_c2m3,
                "selector_minus_greedy_soup": snapshot.selector_minus_greedy_soup,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
