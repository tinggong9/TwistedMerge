#!/usr/bin/env python3
"""Generate conservative verification-pipeline and claim-audit artifacts.

This script is intentionally small and deterministic. It writes the fixed-setting
audit layer used to keep paper claims separated from smoke tests, descriptive
implementation checks, and full repeated-seed empirical support.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CSV_DIR = REPORTS / "csv"
TABLE_DIR = REPORTS / "tables"

CLAIM_FIELDS = [
    "claim_id",
    "claim",
    "status",
    "evidence",
    "safe_wording",
    "unsupported_wording",
    "fake_mnist_support_allowed",
    "next_gate",
]

CLAIMS = [
    {
        "claim_id": "controlled_twisted_overlap_rank_lift",
        "claim": "Controlled twisted-overlap rank lift",
        "status": "supported controlled",
        "evidence": (
            "experiments/controlled_twisted_overlap_benchmark.py; "
            "reports/controlled_twisted_overlap_report.md; "
            "reports/csv/controlled_twisted_overlap_summary.csv"
        ),
        "safe_wording": (
            "In the controlled central-twist benchmark, rank-lifted branches "
            "are supported as controlled obstruction-structured evidence."
        ),
        "unsupported_wording": (
            "This does not show that real neural residuals are Brauer classes "
            "or that rank lift is a capacity-matched single merged model."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": "Keep capacity and branch-count caveats in any paper text.",
    },
    {
        "claim_id": "training_quality_sweep",
        "claim": "Training quality sweep",
        "status": "supported design choice",
        "evidence": (
            "experiments/train_quality_sweep.py; "
            "reports/training_quality_sweep_report.md; "
            "reports/csv/training_quality_sweep.csv"
        ),
        "safe_wording": (
            "The training-quality sweep supports choosing model-quality "
            "settings before the confirmatory verification run."
        ),
        "unsupported_wording": (
            "Do not use the sweep as evidence for obstruction prediction or "
            "merge-method superiority."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": "Use only as a design-choice artifact.",
    },
    {
        "claim_id": "real_fixed_setting_obstruction_prediction",
        "claim": "Real fixed-setting obstruction prediction",
        "status": "not yet supported unless full runs pass gates",
        "evidence": (
            "experiments/model_merging_fixed_setting_verification.py; "
            "reports/obstruction_predictor_target_report.md; "
            "reports/csv/obstruction_predictor_target_stats.csv"
        ),
        "safe_wording": (
            "The fixed-setting script is the confirmatory real verification "
            "entry point, but real obstruction-prediction claims remain gated "
            "until full observed repeated-seed runs pass the predefined "
            "statistical criteria."
        ),
        "unsupported_wording": (
            "Do not claim raw weight-average degradation prediction, broad "
            "real-model prediction, or positive empirical support from "
            "fake-MNIST smoke rows."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": (
            "Run experiments/model_merging_fixed_setting_verification.py on "
            "observed real datasets with enough repeated seeds and CIs."
        ),
    },
    {
        "claim_id": "monomial_gauge_functional_preservation",
        "claim": "Monomial gauge functional preservation",
        "status": "supported implementation",
        "evidence": (
            "src/monomial_gauge_alignment.py; "
            "tests/test_monomial_gauge_alignment.py; "
            "reports/monomial_gauge_alignment_report.md"
        ),
        "safe_wording": (
            "Positive monomial ReLU MLP gauges are implemented and tested as "
            "function-preserving transformations."
        ),
        "unsupported_wording": (
            "Do not turn exact functional preservation into a performance or "
            "generalization claim."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": "Require repeated-seed performance runs for accuracy claims.",
    },
    {
        "claim_id": "monomial_gauge_performance",
        "claim": "Monomial gauge performance",
        "status": "not yet supported",
        "evidence": (
            "reports/monomial_gauge_alignment_report.md is "
            "implementation/descriptive until full repeated-seed runs exist."
        ),
        "safe_wording": (
            "Monomial gauge performance remains an open empirical question in "
            "this audit layer."
        ),
        "unsupported_wording": (
            "Do not claim monomial gauges improve merge accuracy from "
            "implementation checks alone."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": "Add full repeated-seed validation/test comparisons.",
    },
    {
        "claim_id": "greedy_soup_win",
        "claim": "Greedy soup win",
        "status": "not supported",
        "evidence": (
            "reports/external_baseline_comparison.md and later audit reports "
            "treat greedy soup as a strong boundary baseline."
        ),
        "safe_wording": (
            "Greedy soup remains a strong boundary baseline that exact-gauge "
            "methods do not robustly beat under the current evidence."
        ),
        "unsupported_wording": (
            "Do not claim TwistedMerge beats greedy soup unless paired CIs "
            "directly support that exact comparison."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": "Require paired validation/test deltas with positive CI lower bounds.",
    },
    {
        "claim_id": "official_external_baseline_win",
        "claim": "Official external baseline win",
        "status": "not supported",
        "evidence": (
            "external_baselines/OFFICIAL_INTEGRATION.md; "
            "reports/official_external_baseline_attempt.md"
        ),
        "safe_wording": (
            "Official external-code integration was attempted and documented, "
            "but no official baseline win is claimed."
        ),
        "unsupported_wording": (
            "Do not say TwistedMerge beats official Git-ReBasin, C2M3, Model "
            "Soups, or NSD baselines unless official-code runs produce those "
            "metrics."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": "Run official code on the exact checkpoint split before any official-win claim.",
    },
    {
        "claim_id": "real_brauer_projective_residual",
        "claim": "Real Brauer/projective residual claim",
        "status": "not supported",
        "evidence": (
            "reports/claims_audit.md; residual taxonomy reports; "
            "period-index detector reports"
        ),
        "safe_wording": (
            "Real residuals remain non-Brauer under tested diagnostics; "
            "controlled period-index examples support the mathematics."
        ),
        "unsupported_wording": (
            "Do not call real MNIST, Fashion-MNIST, CIFAR, CNN, or block "
            "residuals Brauer/period-index classes under the current evidence."
        ),
        "fake_mnist_support_allowed": "no",
        "next_gate": "Require accepted central/projective detector rows on real residuals.",
    },
]

PIPELINE_ROWS = [
    {
        "artifact": "experiments/model_merging_benchmark.py --mode verification",
        "status": "historical/descriptive",
        "role": (
            "Retained for continuity with earlier benchmark plumbing. It is "
            "not the next confirmatory real verification run."
        ),
    },
    {
        "artifact": "experiments/model_merging_fixed_setting_verification.py",
        "status": "confirmatory real verification script",
        "role": (
            "Use this script for the next confirmatory real run measuring "
            "obstruction predictors, merge degradation targets, rank-lift "
            "comparisons, and confidence intervals."
        ),
    },
    {
        "artifact": "experiments/controlled_twisted_overlap_benchmark.py",
        "status": "confirmatory controlled central-twist benchmark",
        "role": (
            "Use this for controlled central-twist obstruction and rank-lift "
            "claims, not for natural real-model Brauer claims."
        ),
    },
    {
        "artifact": "experiments/train_quality_sweep.py",
        "status": "only for choosing model-quality settings",
        "role": (
            "This supports choosing train/width/epoch settings before the "
            "fixed-setting run; it is not a merge-claim experiment."
        ),
    },
    {
        "artifact": "reports/monomial_gauge_alignment_report.md",
        "status": "implementation/descriptive until full repeated-seed runs exist",
        "role": (
            "This supports exact ReLU-compatible gauge implementation and "
            "functional preservation, not performance."
        ),
    },
]

SAFE_ABSTRACT_WORDING = [
    (
        "We study model-merging residuals as descent defects and separate "
        "controlled obstruction evidence from real-model diagnostic evidence."
    ),
    (
        "Controlled central-twist experiments support rank-lifted branch "
        "constructions in a synthetic setting, while real fixed-setting "
        "obstruction-prediction claims remain gated by repeated-seed "
        "verification."
    ),
    (
        "Positive monomial ReLU gauges are implemented as exact "
        "function-preserving symmetries, but their performance advantage is "
        "not claimed without full repeated-seed support."
    ),
    (
        "Greedy soup and official external baselines remain claim boundaries; "
        "no official external-code win or robust greedy-soup win is claimed."
    ),
    (
        "Real residuals are reported as non-Brauer under tested diagnostics, "
        "while period-index and projective lifts are claimed only for "
        "controlled certified settings."
    ),
]

FORBIDDEN_WORDING = [
    "TwistedMerge broadly beats greedy soup.",
    "TwistedMerge beats official external baselines.",
    "Fake-MNIST smoke runs support real empirical claims.",
    "Raw weight-average degradation is predicted unless the full fixed-setting gates pass.",
    "Monomial gauges improve performance based only on implementation checks.",
    "Real MNIST/Fashion-MNIST/CIFAR residuals are Brauer or period-index classes.",
    "The historical model_merging_benchmark.py verification mode is the confirmatory real run.",
]


def md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def markdown_table(rows: list[dict[str, str]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(md_cell(str(row[field])) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def latex_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{p{0.24\linewidth}p{0.20\linewidth}p{0.25\linewidth}p{0.23\linewidth}}",
        r"\toprule",
        r"Claim & Status & Safe wording & Forbidden wording \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["claim"]),
                    latex_escape(row["status"]),
                    latex_escape(row["safe_wording"]),
                    latex_escape(row["unsupported_wording"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_claim_csv() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with (CSV_DIR / "claims_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLAIM_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in CLAIMS:
            writer.writerow(row)


def write_claim_tex() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "claims_audit.tex").write_text(latex_table(CLAIMS), encoding="utf-8")


def claim_audit_section() -> str:
    section_lines = [
        "<!-- claim_audit-claim-audit:start -->",
        "## fixed-setting Verification Pipeline And Claim Boundary Audit",
        "",
        (
            "This section is generated by "
            "`experiments/generate_claim_audit.py`. It is deliberately "
            "conservative: fake-MNIST smoke rows are diagnostic only and "
            "never count as empirical support."
        ),
        "",
        "### Required Claim Statuses",
        "",
        markdown_table(CLAIMS, ["claim_id", "status", "evidence", "safe_wording", "unsupported_wording"]),
        "",
        "### Evidence-Constrained Summary Wording",
        "",
    ]
    section_lines.extend(f"- {item}" for item in SAFE_ABSTRACT_WORDING)
    section_lines.extend(["", "### Unsupported Extrapolations", ""])
    section_lines.extend(f"- {item}" for item in FORBIDDEN_WORDING)
    section_lines.extend(
        [
            "",
            "### Next Confirmatory Real Run",
            "",
            (
                "Use `experiments/model_merging_fixed_setting_verification.py` "
                "for the next confirmatory real verification run. Keep "
                "`experiments/model_merging_benchmark.py --mode verification` "
                "as historical/descriptive context only."
            ),
            "",
            "<!-- claim_audit-claim-audit:end -->",
            "",
        ]
    )
    return "\n".join(section_lines)


def update_claims_audit_md() -> None:
    path = REPORTS / "claims_audit.md"
    original = path.read_text(encoding="utf-8") if path.exists() else "# Claims Audit\n"
    section = claim_audit_section()
    pattern = re.compile(
        r"<!-- claim_audit-claim-audit:start -->.*?<!-- claim_audit-claim-audit:end -->\n?",
        re.DOTALL,
    )
    if pattern.search(original):
        updated = pattern.sub(section, original)
    else:
        updated = original.rstrip() + "\n\n" + section
    path.write_text(updated, encoding="utf-8")


def write_pipeline_status() -> None:
    lines = [
        "# Verification Pipeline Status",
        "",
        (
            "This report fixes the current pipeline roles before additional "
            "experiments are run. It names exactly which script should be used "
            "for the next confirmatory real run."
        ),
        "",
        "## Script Roles",
        "",
        markdown_table(PIPELINE_ROWS, ["artifact", "status", "role"]),
        "",
        "## Next Confirmatory Real Run",
        "",
        (
            "The next confirmatory real run should use "
            "`experiments/model_merging_fixed_setting_verification.py`. That "
            "script is the current real-model verification entry point for "
            "obstruction predictors, alignment-conditioned targets, ordinary "
            "merge degradation, cycle-consistent merge, rank-lift comparisons, "
            "ensemble bounds, and bootstrap confidence intervals."
        ),
        "",
        "## Claim Boundary Notes",
        "",
        (
            "- `experiments/model_merging_benchmark.py --mode verification` is "
            "historical/descriptive and should not be cited as the final "
            "confirmatory real verification run."
        ),
        (
            "- `experiments/model_merging_fixed_setting_verification.py` is the "
            "confirmatory real verification script."
        ),
        (
            "- `experiments/controlled_twisted_overlap_benchmark.py` is the "
            "confirmatory controlled central-twist benchmark."
        ),
        (
            "- `experiments/train_quality_sweep.py` is only for choosing "
            "model-quality settings."
        ),
        (
            "- `reports/monomial_gauge_alignment_report.md` is "
            "implementation/descriptive until full repeated-seed runs exist."
        ),
        (
            "- Fake-MNIST smoke runs are diagnostic only and never empirical "
            "support for paper claims."
        ),
        "",
    ]
    (REPORTS / "verification_pipeline_status.md").write_text("\n".join(lines), encoding="utf-8")


def write_evidence_summary() -> None:
    lines = [
        "# Paper Evidence Summary",
        "",
        (
            "This summary separates implemented artifacts, completed runs, and "
            "claims supported by data. It is synchronized with "
            "`experiments/generate_claim_audit.py`."
        ),
        "",
        "## Implemented",
        "",
        "- `experiments/model_merging_fixed_setting_verification.py` is implemented as the confirmatory real verification script.",
        "- `experiments/controlled_twisted_overlap_benchmark.py` is implemented as the confirmatory controlled central-twist benchmark.",
        "- `experiments/train_quality_sweep.py` is implemented for model-quality setting selection.",
        "- `src/monomial_gauge_alignment.py` and `tests/test_monomial_gauge_alignment.py` implement and test exact ReLU-compatible monomial gauges.",
        "- `experiments/generate_claim_audit.py` generates the fixed-setting audit section plus CSV/TeX/status artifacts.",
        "",
        "## Run Or Descriptive",
        "",
        "- Controlled twisted-overlap outputs are present and support controlled rank-lift wording only.",
        "- Training-quality sweep outputs are present and support design-choice wording only.",
        "- Historical `experiments/model_merging_benchmark.py --mode verification` outputs are descriptive context only.",
        "- `reports/monomial_gauge_alignment_report.md` is implementation/descriptive until full repeated-seed runs exist.",
        "- Official external-code integration was attempted and documented, but no official external baseline metrics were produced.",
        "",
        "## Supported By Data Or Tests",
        "",
        markdown_table(CLAIMS, ["claim_id", "status", "safe_wording"]),
        "",
        "## Unsupported Or Gated",
        "",
        "- Real fixed-setting obstruction prediction is not yet supported unless full observed repeated-seed runs pass gates.",
        "- Monomial gauge performance is not yet supported.",
        "- A greedy soup win is not supported.",
        "- An official external baseline win is not supported.",
        "- Real Brauer/projective residual claims are not supported.",
        "- Fake-MNIST smoke rows are diagnostic only and never empirical support.",
        "",
    ]
    (REPORTS / "paper_evidence_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_claim_csv()
    write_claim_tex()
    update_claims_audit_md()
    write_pipeline_status()
    write_evidence_summary()
    print("Generated fixed-setting claim-audit artifacts.")


if __name__ == "__main__":
    main()
