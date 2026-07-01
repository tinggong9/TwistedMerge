# Final Claim Ledger

Generated for Prompt 38. This ledger freezes current claim boundaries without adding paper prose.

## Ledger Rules

- Categories are `supported`, `supported-narrow`, `descriptive`, `negative`, and `forbidden`.
- Evidence families distinguish `independent-seed/rebasin`, `same-base task-vector`, `greedy-soup empirical descent`, and `controlled twisted obstruction`; controlled period-index rows are marked separately.
- Real Brauer/real period-index claims are not supported unless explicitly controlled-only.
- Official external-baseline wins are forbidden because official-code runs did not succeed on the exact checkpoint set.

## Supported Claims

| claim_id | family | claim | evidence | row_count | boundary |
| --- | --- | --- | --- | --- | --- |
| `monomial_function_preservation` | independent-seed/rebasin | Positive monomial ReLU gauges preserve model functions before averaging in the implemented MLP gauge layer. | reports/monomial_gauge_alignment_report.md; reports/csv/monomial_fixed_setting_runs.csv | monomial run rows recorded in CSV; report lines inventoried in freeze manifest | Do not convert exact gauge preservation into a broad performance claim. |
| `baseline_regime_separation` | claim-boundary/integration | The baseline-regime audit separates independent-seed/rebasin methods from same-base task-vector methods and records which comparisons are fair. | reports/baseline_regime_audit.md; reports/csv/baseline_regime_audit.csv | 10 audit rows | Do not judge Task Arithmetic/TIES/DARE from independent-seed rows or compare rank-lift directly to same-capacity task-vector baselines. |

## Supported-Narrow Claims

| claim_id | family | claim | evidence | row_count | boundary |
| --- | --- | --- | --- | --- | --- |
| `controlled_mu2_h2_rank_lift` | controlled twisted obstruction | In the controlled nontrivial mu2 obstruction benchmark, supplied-context q=2 branch lift outperforms ordinary/C2M3 and branch-capacity controls. | reports/controlled_twisted_overlap_report.md; reports/csv/controlled_twisted_overlap_summary.csv | 156 summary rows | Do not claim real MNIST/Fashion/CIFAR residuals are Brauer classes from this controlled result. |
| `controlled_coboundary_sync` | controlled twisted obstruction | Controlled coboundary central-sign rows support cycle-consistent synchronization as the resolved-residual case. | reports/controlled_twisted_overlap_report.md; reports/csv/controlled_twisted_overlap_summary.csv | 156 summary rows | Do not generalize to arbitrary real residuals. |
| `controlled_period_index_thresholds` | controlled period-index/time-frequency | Controlled finite-index, period-index, robust detector, and time-frequency chart artifacts support period/index rank-threshold behavior in controlled systems. | reports/period_index_central_report.md; reports/robust_period_index_calibration_report.md; reports/time_frequency_heisenberg_projection_report.md | period_index summary 7; robust calibration summary 560; Heisenberg projection summary 1220 | Do not label real residuals as period-index/Brauer unless real detector rows certify them. |
| `monomial_vs_c2m3_limited` | independent-seed/rebasin | Same-capacity monomial rows pass a positive paired-CI gate versus internal C2M3 in at least one MNIST/Fashion fixed setting. | reports/monomial_gauge_alignment_report.md; reports/csv/monomial_paired_deltas.csv | paired-delta rows recorded in CSV; exact count in freeze manifest | Do not claim monomial gauges broadly beat greedy soup or official baselines. |
| `fixed_setting_alignment_conditioned_prediction` | independent-seed/rebasin | The obstruction predictor report supports alignment-conditioned targets, including Git-ReBasin/C2M3 degradation and rank-lift delta versus C2M3, in selected observed rows. | reports/obstruction_predictor_target_report.md; reports/csv/obstruction_predictor_target_stats.csv | 1536 stats rows | Do not claim raw weight-average degradation prediction. |
| `same_base_task_vector_exact_settings` | same-base task-vector | Same-base task-vector artifacts support exact-setting validation-selected improvements for Task Arithmetic/DARE/TIES-style families over the original greedy soup in several common-base settings. | reports/same_base_task_vector_report.md; reports/same_base_task_vector_extended.md; reports/csv/same_base_task_vector_extended_summary.csv | same-base summary 80; extended summary 106 | Do not use this to judge independent random-initialization rebasin methods. |
| `descent_envelope_enriched_pool` | same-base task-vector | The descent-envelope selector supports exact-setting enriched generated-candidate validation descent over the original greedy soup in the tested same-base pool. | reports/descent_envelope_selector_report.md; reports/csv/descent_envelope_summary.csv | 35 summary rows | Do not cite as broad model-merging superiority. |
| `greedy_soup_checkpointed_trajectory` | greedy-soup empirical descent | Checkpointed activation-setting greedy-soup trajectories directly support stepwise validation-descent accounting with no validation monotonicity or rejection-rule violations. | reports/greedy_soup_trajectory_report.md; reports/csv/greedy_soup_trajectory_summary.csv | 9 summary rows; 840 trajectory rows | Do not extend to rows where rejected candidate metrics were not logged. |

## Descriptive Claims

| claim_id | family | claim | evidence | row_count | boundary |
| --- | --- | --- | --- | --- | --- |
| `greedy_soup_final_only_descent` | greedy-soup empirical descent | The reconstructed greedy-soup descent audit supports final-output validation safety, while rejected-candidate margins remain algorithm-implied rather than empirically logged. | reports/greedy_soup_descent_audit.md; reports/csv/greedy_soup_descent_summary.csv | 16 summary rows; 1680 audit rows | Do not claim every rejected averaged candidate has an observed non-positive margin from this audit. |
| `task_vector_interference_diagnostic` | same-base task-vector | Task-vector interference diagnostics are descriptive same-base diagnostics and are separated from independent-seed cycle obstruction. | reports/task_vector_interference_report.md; reports/csv/task_vector_interference_summary.csv | 690 summary rows | Do not certify Brauer/projective obstruction from same-base interference rows. |
| `slerp_barrier_geometry_boundary` | barrier/path geometry | SLERP is treated as a path-geometry baseline; in the bounded run it does not lower validation max-loss barriers on average. | reports/slerp_barrier_geometry_report.md; reports/csv/slerp_barrier_geometry_summary.csv | 18 summary rows | Do not claim a broad SLERP win or failure outside the reported regime. |

## Negative Claims

| claim_id | family | claim | evidence | row_count | boundary |
| --- | --- | --- | --- | --- | --- |
| `same_base_union_selector_negative` | same-base task-vector | The same-base union candidate selector does not pass the positive paired-CI gate versus the best existing method. | reports/same_base_union_candidate_selector.md; reports/csv/same_base_union_candidate_selector_summary.csv | 49 summary rows | Do not claim the union selector improves over the best existing same-base method. |
| `official_integration_attempt_only` | claim-boundary/integration | Official external Git-ReBasin, C2M3, and Model Soups integrations were attempted/documented but did not produce official-code results on the exact checkpoint set. | reports/official_external_baseline_attempt.md; external_baselines/OFFICIAL_INTEGRATION.md | report line count in freeze manifest | Do not claim TwistedMerge beats official external baselines. |
| `real_brauer_period_index_not_supported` | independent-seed/rebasin | Real MNIST/Fashion/CIFAR residuals remain non-Brauer or uncertified under the tested diagnostics. | reports/claims_audit.md; residual taxonomy reports; reports/csv/finite_index_residual_mining_summary.csv | finite-index residual mining summary 18 rows plus audit inventory | Do not call real residuals Brauer/period-index classes. |
| `raw_weight_average_prediction_not_supported` | independent-seed/rebasin | Raw weight-average degradation prediction by the obstruction score is not supported by the current predictor-target report. | reports/obstruction_predictor_target_report.md; reports/csv/obstruction_predictor_target_stats.csv | 1536 stats rows | Do not claim raw weight-average prediction. |
| `greedy_soup_win_not_supported` | independent-seed/rebasin | A broad TwistedMerge win over greedy soup is not supported; greedy soup remains a strong boundary baseline. | reports/claims_audit.md; reports/monomial_gauge_alignment_report.md; reports/external_baseline_comparison.md | audit row counts in freeze manifest | Do not say the method beats greedy soup unless an exact paired CI supports that comparison. |
| `cifar_general_win_not_supported` | independent-seed/rebasin | CIFAR exact-gauge effects remain descriptive/bounded; broad CIFAR or general vision wins are not supported. | reports/cifar_final_channel_gauge_confirmatory_report.md; reports/cifar_rescue_or_no_go_report.md | CIFAR final summary 32; rescue summary 8 | Do not claim robust CIFAR exact-gauge win or broad vision result. |

## Forbidden Claims

| claim_id | family | claim | evidence | row_count | boundary |
| --- | --- | --- | --- | --- | --- |
| `forbid_official_external_win` | claim-boundary/integration | TwistedMerge beats official Git-ReBasin, official C2M3, official Model Soups, or official NSD baselines. | reports/official_external_baseline_attempt.md; reports/nsd_official_integration_report.md | report line counts in freeze manifest | No official external-baseline win claim. |
| `forbid_real_brauer_positive` | independent-seed/rebasin | Real MNIST/Fashion/CIFAR residuals are Brauer, real-period-index, or central projective classes. | reports/claims_audit.md; reports/final_evidence_freeze_manifest.md | audit inventory | Controlled-only period-index evidence must remain labeled controlled-only. |
| `forbid_rank_lift_single_model` | controlled twisted obstruction | Rank-lifted branch models are capacity-matched single merged models. | reports/controlled_twisted_overlap_report.md; reports/baseline_regime_audit.md; reports/csv/full_capacity_claim_audit.csv | full capacity audit 61 rows | Always label branch/rank lift capacity and inference multipliers. |
| `forbid_broad_sota` | all families | The current artifacts establish broad SOTA model-merging performance or broad generalization. | reports/claims_audit.md; reports/paper_level_decision_after_35791f7.md; reports/full_capacity_claim_audit.md | claims audit 8 CSV rows; full capacity audit 61 rows | Keep claims exact-setting, controlled, or descriptive as recorded. |

