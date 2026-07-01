# Reviewer Contingency Checklist

Generated for Prompt 40. This is a reviewer-response planning artifact, not paper prose. No experiments were run for this checklist.

## Scope

- Purpose: decide which reviewer criticisms are already covered by existing artifacts and which would require new data.
- Rule: only mark an experiment essential when the paper would otherwise need a stronger claim than the current claim audit allows.
- Default posture: keep the current conservative boundaries unless a reviewer explicitly asks for stronger evidence.

## Checklist

| ID | Possible reviewer criticism | Current evidence already available | More data needed? | Minimal experiment if needed | Optional or essential |
| --- | --- | --- | --- | --- | --- |
| R1 | Why not compare to SLERP/TIES/DARE? | `reports/baseline_regime_audit.md` separates same-base task-vector methods from independent-seed rebasin methods. `reports/same_base_task_vector_report.md` runs Model Soups, SLERP, Task Arithmetic, TIES, and DARE on 6 completed same-base settings with 20 seeds each. `reports/descent_envelope_selector_report.md` treats SLERP/TIES/DARE/Task Arithmetic/DARE as validation-only generated candidates. | No for the basic criticism. Yes only if reviewers demand broader task-vector coverage beyond the current MNIST/Fashion-MNIST settings. | Extend `experiments/same_base_task_vector_benchmark.py` to one additional task family or dataset, keeping common base checkpoints, validation-only hyperparameter selection, and paired bootstrap CIs. | Optional for current boundaries; essential only for a broader same-base generality claim. |
| R2 | Why does greedy soup remain strong? | `reports/greedy_soup_trajectory_report.md` directly logs every accepted and rejected candidate-soup validation metric for checkpointed fixed-setting runs, with zero validation monotonicity violations and no test selection. `reports/expanded_candidate_soup_report.md` shows expanded gauge candidates do not improve over ordinary greedy soup under paired CIs. `reports/soup_acceptance_obstruction_diagnostics.md` models greedy-soup acceptance as something to explain, not beat. `reports/descent_envelope_selector_report.md` shows enriched same-base candidate pools can improve validation descent in exact settings while keeping greedy as the baseline boundary. | No for explaining greedy soup as a strong empirical validation-descent baseline. Yes only if making a new claim that another method robustly beats greedy soup. | For any new candidate family, rerun full greedy-soup trajectory logging with candidate-soup validation accuracy/loss before each accept/reject decision and paired CIs versus ordinary greedy soup. | Optional for explanation; essential for any soup-beating claim. |
| R3 | Are real residuals actually Brauer classes? | `reports/claims_audit.md` marks real Brauer/projective residual claims as not supported. Controlled period-index and central-twist artifacts support the mathematical story only in controlled settings, while real MNIST/Fashion/CIFAR residuals remain non-Brauer under tested diagnostics. `reports/controlled_twisted_overlap_report.md` explicitly separates controlled obstruction evidence from real-model evidence and keeps random noncentral rows as negative controls. | No if the paper keeps the current boundary that real residuals are diagnostics, not certified Brauer classes. Yes if the paper claims real residuals are Brauer/projective/period-index classes. | Add a certified real-residual detector with calibrated false-lift control, period/index divisibility gates, and preregistered thresholds; rerun on the quality-gated real fixed-setting checkpoints. | Optional for current diagnostic wording; essential for any real-Brauer claim. |
| R4 | Is rank-lift extra capacity? | `reports/claims_audit.md` says rank-lift branch evidence is separated from branch-capacity matched controls. `reports/controlled_twisted_overlap_report.md` includes random branch, validation-selected branch, C2M3-cluster branch, no-twist branch, parameter-matched wide, wrong-twist, wrong-context, learned-router, and distilled-single controls. The same report states that failed distillation leaves the branch result extra-capacity/charted. | No for labeling q=2 rank-lift as branch/extra-capacity. Yes only if the paper wants a same-capacity rank-lift superiority claim. | Strengthen `experiments/controlled_twisted_overlap_benchmark.py` with an additional successful same-capacity distillation or stricter parameter-matched single-model control, then require paired CIs against branch-capacity controls. | Optional for current controlled branch claim; essential for any same-capacity rank-lift claim. |
| R5 | Is monomial gauge exact or heuristic? | `reports/monomial_gauge_alignment_report.md` states that positive monomial ReLU gauges are exact reparameterizations before averaging, single same-capacity, not branch/rank-lift methods. Functional preservation tables report zero preservation error and zero prediction disagreement across reported rows. `src/monomial_gauge_alignment.py` and `tests/test_monomial_gauge_alignment.py` are listed in `reports/claims_audit.md` for implementation support. | No for exact positive ReLU-gauge preservation. Yes only for a performance-improvement claim or for a new architecture not covered by tests. | Add architecture-specific exactness tests for any new model family; for performance, run a 20+ seed paired monomial benchmark with no test leakage and CIs versus C2M3 and greedy soup. | Optional for exactness in current architectures; essential for performance or new-architecture claims. |
| R6 | Why separate same-base and independent-seed settings? | `reports/baseline_regime_audit.md` gives the regime split: SLERP/Task Arithmetic/TIES/DARE need same-base task-vector coordinates, while Git-ReBasin/C2M3/monomial gauges address independent-seed coordinate mismatch. `reports/same_base_task_vector_report.md` records same-base results and marks Git-ReBasin/C2M3 as secondary not-run diagnostics. `reports/external_baseline_comparison.md` and fixed-setting reports cover the independent-seed side. `reports/claims_audit.md` marks the separation as supported descriptive. | No for the methodological separation. Yes only if reviewers request a paired demonstration across both regimes with matched architecture and datasets. | Run two matched tables with identical architecture/data budgets: common-base fine-tune for task-vector methods, and independent-seed training for rebasin/gauge methods. Do not collapse them into one leaderboard. | Optional for clarity; essential only if making a unified cross-regime ranking claim. |

## Immediate Triage

| Action | Decision |
| --- | --- |
| Add new experiments before submission | Not required by this checklist. |
| Strengthen claims about real Brauer classes | Do not do this without essential new certified real-residual evidence. |
| Strengthen claims about same-capacity rank-lift | Do not do this without successful same-capacity or stricter parameter-matched evidence. |
| Compare SLERP/TIES/DARE in independent-seed rows | Do not do this; use same-base task-vector rows. |
| Claim monomial performance from exactness tests | Do not do this; exactness and performance are separate. |
| Treat greedy soup as weak | Do not do this; current artifacts make it the empirical validation-descent boundary. |

## Existing Artifact Index

- `reports/baseline_regime_audit.md`
- `reports/same_base_task_vector_report.md`
- `reports/descent_envelope_selector_report.md`
- `reports/greedy_soup_trajectory_report.md`
- `reports/expanded_candidate_soup_report.md`
- `reports/soup_acceptance_obstruction_diagnostics.md`
- `reports/controlled_twisted_overlap_report.md`
- `reports/monomial_gauge_alignment_report.md`
- `reports/external_baseline_comparison.md`
- `reports/claims_audit.md`
