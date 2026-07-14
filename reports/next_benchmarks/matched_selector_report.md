# Matched Candidate-Budget and Selector Ablation Report

Primary practical-selector decision: **unsupported** versus ordinary greedy soup under the tracked executed grid.

## Exact command

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/matched_selector_budget_benchmark.py
```

- Git commit at aggregation: `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf`
- Primary grid: MNIST, one-hidden-layer ReLU MLP, `n_models=3,4`, widths `16,32,64`, seeds `1800:1819`.
- All 120 primary settings are present in the tracked executed-model source.
- This run is a clean aggregation and matched-selection audit of those executed Torch rows; it is **not fresh inference from every checkpoint on the current commit**. That limitation prevents these numbers from entering the clean release manifest until a full checkpoint rerun is made.
- Test accuracy is used only for final evaluation and the explicitly labeled regret audit, never candidate selection.

## Main summary

| method | n_settings | n_seeds | mean_test_accuracy | mean_validation_accuracy | mean_candidate_count | mean_selected_candidate_count | coverage_status | fresh_inference_on_current_commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c2m3_greedy_soup | 120 | 20 | 0.856868 | 0.84775 | 1 | 1.05 | complete_primary_grid | False |
| c2m3_synchronization_alone | 120 | 20 | 0.811894 | 0.801317 | 1 | 1 | complete_primary_grid | False |
| git_rebasin_style_pairwise | 20 | 5 | 0.845975 | 0.82885 | 1 | 1 | partial_grid | False |
| global_monomial | 120 | 20 | 0.82689 | 0.819875 | 1 | 1 | complete_primary_grid | False |
| improved_twistedmerge_exact_gauge_soup_selector | 120 | 20 | 0.855735 | 0.849383 | 1 | 1 | complete_primary_grid | False |
| monomial_greedy_soup | 120 | 20 | 0.85698 | 0.847867 | 1 | 1.06667 | complete_primary_grid | False |
| optimized_monomial | 120 | 20 | 0.820628 | 0.814342 | 1 | 1 | complete_primary_grid | False |
| ordinary_greedy_soup | 120 | 20 | 0.857245 | 0.8476 | 1 | 1 | complete_primary_grid | False |
| ordinary_weight_average | 120 | 20 | 0.666268 | 0.661167 | 1 | 1 | complete_primary_grid | False |
| randomly_augmented_candidate_union | 120 | 20 | 0.856416 | 0.8491 | 12 | 1 | complete_primary_grid | False |
| raw_monomial_alignment_alone | 120 | 20 | 0.819743 | 0.80985 | 1 | 1 | complete_primary_grid | False |
| shrinkage_monomial | 120 | 20 | 0.826762 | 0.820408 | 1 | 1 | complete_primary_grid | False |
| union_candidate_soup | 120 | 20 | 0.856587 | 0.848958 | 14 | 2.49167 | complete_primary_grid | False |

## Paired statistics

| comparison | n_pairs | paired_mean_accuracy_delta | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_twistedmerge_exact_gauge_soup_selector_vs_ordinary_greedy_soup | 120 | -0.00151 | -0.00226352 | -0.000846604 | 20 | 49 | 51 |
| union_candidate_soup_vs_ordinary_greedy_soup | 120 | -0.0006575 | -0.00109337 | -0.000249958 | 19 | 58 | 43 |
| randomly_augmented_candidate_union_vs_ordinary_greedy_soup | 120 | -0.000829167 | -0.00137008 | -0.000333146 | 20 | 56 | 44 |
| raw_monomial_alignment_alone_vs_c2m3_synchronization_alone | 120 | 0.00784833 | 0.00455652 | 0.0112309 | 89 | 1 | 30 |
| shrinkage_monomial_vs_raw_monomial_alignment_alone | 120 | 0.00702 | 0.00434706 | 0.0100093 | 85 | 6 | 29 |
| global_monomial_vs_raw_monomial_alignment_alone | 120 | 0.0071475 | 0.00462408 | 0.0102936 | 85 | 4 | 31 |
| optimized_monomial_vs_raw_monomial_alignment_alone | 120 | 0.000885 | -0.00302029 | 0.00479598 | 56 | 0 | 64 |

## Candidate-selection audit

| method | n_settings | mean_regret | central_lift_rate | nonabelian_lift_rate |
| --- | --- | --- | --- | --- |
| improved_twistedmerge_exact_gauge_soup_selector | 120 | 0.00178167 | 0 | 0 |
| randomly_augmented_candidate_union | 120 | 0.00110083 | 0 | 0 |

The practical selector selected **no central lift and no nonabelian branch lift**. Its available choices were exact-gauge or soup candidates. The obstruction-gated branch candidate was never activated because no setting supplied a valid certificate.

## Exact blockers

| method | evaluation_status | blocker |
| --- | --- | --- |
| always_on_branch_candidate | not_run_exact_blocker | No certified central or nonabelian branch tensor exists for any primary-grid setting; constructing one without a certificate would violate the benchmark gate. |
| obstruction_gated_branch_candidate | not_run_exact_blocker | No certified central or nonabelian branch tensor exists for any primary-grid setting; constructing one without a certificate would violate the benchmark gate. |

`git_rebasin_style_pairwise` has only the 20-setting tracked external subset and is labeled partial coverage. The always-on branch control is not fabricated: no executed certified branch tensor exists for the 120-setting source grid. This is a benchmark limitation, not a positive result.
