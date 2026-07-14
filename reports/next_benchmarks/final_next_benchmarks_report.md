# Final Next-Benchmarks Report

## Executive decisions

| benchmark | decision |
| --- | --- |
| Stage 0 evidence audit | Existing target-injected S3/D4 accuracy artifacts are invalid and quarantined. |
| Executed two-loop holonomy | B. Structural noncommuting holonomy supported, but accuracy advantage unsupported. |
| Context-router generalization | Learned practical router `unsupported`; supplied-context oracle retained separately. |
| Controlled central reproduction | Supported as an executed controlled construction. |
| Finite-Heisenberg period-index | Supported as a checked structural representation-theoretic construction. |
| Matched selector budget | Unsupported versus ordinary greedy soup; aggregation is not fresh inference from the evidence commit. |
| Held-out diagnostic prediction | Unsupported under the preregistered held-out gate. |
| Pretrained merging | Not run at full required scale; one-seed ResNet-18 smoke completed. |

Evidence commit: `0a41f76d3c8a77acc3a47514c2639b81fbc5b280`. Complete tests: `355 passed, 5 subtests passed in 18.94s`.

## Exact commands and output paths

| benchmark | exact_command | git_commit | raw_csv | summary_csv | plot_path | latex_table_path |
| --- | --- | --- | --- | --- | --- | --- |
| controlled_mu2_reproduction | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/central_reproduction_next.py --seeds 0:29 --widths 32,64 --samples-per-chart 500 --samples-per-overlap 2000 | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/central_mu2_runs.csv | reports/next_benchmarks/central_mu2_summary.csv | not_applicable | reports/next_benchmarks/tables/central_mu2.tex |
| period_index_reproduction | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/central_reproduction_next.py --seeds 0:29 --widths 32,64 --samples-per-chart 500 --samples-per-overlap 2000 | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/period_index_rank_outcomes.csv | reports/next_benchmarks/period_index_summary.csv | not_applicable | reports/next_benchmarks/tables/period_index.tex |
| executed_two_loop_holonomy | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/executed_two_loop_holonomy.py --mode full | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/two_loop_holonomy_residuals.csv | reports/next_benchmarks/two_loop_holonomy_summary.csv | reports/next_benchmarks/plots/two_loop_holonomy_residuals.pdf | reports/next_benchmarks/tables/two_loop_holonomy_residuals.tex |
| executed_two_loop_holonomy | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/executed_two_loop_holonomy.py --mode full | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/two_loop_holonomy_runs.csv | reports/next_benchmarks/two_loop_holonomy_paired_stats.csv | reports/next_benchmarks/plots/two_loop_holonomy_accuracy.pdf | reports/next_benchmarks/tables/two_loop_holonomy_accuracy.tex |
| context_router_generalization | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/context_router_generalization.py --groups S3,D4 --seeds 0:19 --n-validation-per-context 200 --n-test-per-context 300 | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/context_router_runs.csv | reports/next_benchmarks/context_router_summary.csv | reports/next_benchmarks/plots/context_router_generalization.pdf | reports/next_benchmarks/tables/context_router.tex |
| matched_selector_budget | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/matched_selector_budget_benchmark.py | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/matched_selector_runs.csv | reports/next_benchmarks/matched_selector_paired_stats.csv | reports/next_benchmarks/plots/matched_selector_accuracy.pdf | reports/next_benchmarks/tables/matched_selector_main.tex |
| heldout_diagnostic_prediction | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/heldout_diagnostic_prediction.py --bootstrap-samples 2000 | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/diagnostic_prediction_runs.csv | reports/next_benchmarks/diagnostic_prediction_summary.csv | reports/next_benchmarks/plots/diagnostic_prediction.pdf | reports/next_benchmarks/tables/diagnostic_prediction.tex |
| pretrained_resnet18_smoke | /Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/pretrained_merge_smoke.py --seed 0 --train-samples 512 --validation-samples 256 --test-samples 512 --head-epochs 30 | 0a41f76d3c8a77acc3a47514c2639b81fbc5b280 | reports/next_benchmarks/pretrained_merge_runs.csv | reports/next_benchmarks/pretrained_merge_summary.csv | not_applicable | reports/next_benchmarks/tables/pretrained_merge.tex |

## Compact numerical tables

Controlled nontrivial mu2, width 32:

| family | width | method | n_seeds | mean_test_accuracy | mean_test_loss | parameter_multiplier | branch_count | inference_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mu2_nontrivial_h2 | 32 | ordinary_weight_average | 30 | 0.500096 | 0.693147 | 1 | 1 | 1 |
| mu2_nontrivial_h2 | 32 | random_branch_control | 30 | 0.45 | 0.84749 | 2 | 2 | 2 |
| mu2_nontrivial_h2 | 32 | supplied_context_q2_branch_predictor | 30 | 1 | 0.406317 | 2 | 2 | 2 |

Finite-Heisenberg period-index:

| case_id | d | k | scalar_commutator_order | certified_representation_threshold | minimal_successful_rank | matrix_relation_residual |
| --- | --- | --- | --- | --- | --- | --- |
| d2_k1 | 2 | 1 | 2 | 2 | 2 | 1.73191e-16 |
| d2_k2 | 2 | 2 | 2 | 4 | 4 | 1.73191e-16 |
| d2_k3 | 2 | 3 | 2 | 8 | 8 | 1.73191e-16 |
| d3_k1 | 3 | 1 | 3 | 3 | 3 | 3.04047e-16 |
| d3_k2 | 3 | 2 | 3 | 9 | 9 | 3.04047e-16 |
| d4_k1 | 4 | 1 | 4 | 4 | 4 | 1.22465e-16 |
| d4_k2 | 4 | 2 | 4 | 16 | 16 | 1.22465e-16 |

Practical-selector primary comparison:

| comparison | n_pairs | paired_mean_accuracy_delta | ci_low | ci_high | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| improved_twistedmerge_exact_gauge_soup_selector_vs_ordinary_greedy_soup | 120 | -0.00151 | -0.00226352 | -0.000846604 | 20 | 49 | 51 |

Pretrained smoke:

| method | average_accuracy | worst_task_accuracy | calibration_ece | forgetting_interference |
| --- | --- | --- | --- | --- |
| weight_average | 0.597656 | 0.517787 | 0.0546974 | 0.157934 |
| greedy_soup | 0.578125 | 0.545082 | 0.0391626 | 0.17703 |
| task_arithmetic | 0.597656 | 0.517787 | 0.0546974 | 0.157934 |
| ties | 0.541016 | 0.438735 | 0.231467 | 0.214354 |
| dare | 0.521484 | 0.395257 | 0.0773216 | 0.233128 |
| slerp | 0.597656 | 0.517787 | 0.112756 | 0.157934 |
| twistedmerge_exact_gauge_soup_selector | 0.597656 | 0.517787 | 0.0546974 | 0.157934 |

## Capacity, inference, and selection accounting

- Full capacity tables: `central_mu2_capacity.csv`, `two_loop_holonomy_capacity.csv`, and `matched_selector_capacity.csv`.
- The two-loop branch regular lift stores one learned model (`1x` learned parameters) but executes `8` branches at up to the recorded inference multiplier.
- The central supplied-context q=2 predictor records `2x` branch capacity/inference; the ensemble reference records `4x`.
- The matched practical selector activated central lifts `0` times and nonabelian branch lifts `0` times; it selected only exact-gauge or soup candidates.
- Selector regret is audit-only and was not used for selection.

## Leakage and structural certificates

- Two-loop saved-logit label permutation: passed.
- Context-router saved-logit label permutation: passed.
- Central mu2 saved-logit label permutation: passed.
- Two-loop generator recovery, noncommutation, local equivalence, regular-action multiplication, and both pooling certificates: passed across the full grid.
- Wrong-generator and random-action controls failed the complete structural certificate as required.

## Manuscript claim actions

- Retain, with controlled scope: executed central mu2 supplied-context result and checked finite-Heisenberg period-index theorem.
- Retain only structurally: two-loop S3/D4 noncommuting holonomy and invariant-pooling certificates.
- Weaken: any practical-selector statement to a negative result; the tracked selector was `-0.001510` versus greedy soup with CI `[-0.002264, -0.000847]`.
- Delete: all empirical accuracy claims sourced from the deprecated target-injected controlled-nonabelian artifacts.
- Do not add: a learned practical-router claim, a promoted natural-data diagnostic claim, or a full pretrained-model-merging claim.

## Recommended safe wording

- Controlled mu2: "In an exact planted mu2 overlap construction, executed supplied-context q=2 branch prediction resolves the nontrivial central class; this is a controlled oracle-context result, not a learned-router or natural-data claim."
- Period-index: "For the checked finite-Heisenberg k-pair systems, the scalar commutator has order d and the certified projective representation threshold is d^k; direct sums realize its multiples."
- Noncentral/noncommuting holonomy: "Executed S3/D4 models certify two noncommuting loop holonomies and invariant pooling, but the lift ties controls and supports no accuracy-advantage claim."
- Practical selector: "On the tracked executed MNIST grid, the selector did not beat ordinary greedy soup; no central or nonabelian branch candidate was selected."
- Learned router: "The supplied-context oracle is valid in the controlled construction, while the learned feature router is unsupported on held-out group words."
- Natural diagnostic: "The preregistered natural-data diagnostic did not add held-out predictive value beyond ordinary validation baselines."

## LaTeX files ready to paste

- `reports/next_benchmarks/tables/central_mu2.tex`
- `reports/next_benchmarks/tables/period_index.tex`
- `reports/next_benchmarks/tables/two_loop_holonomy_residuals.tex`
- `reports/next_benchmarks/tables/context_router.tex` (negative/diagnostic table)
- `reports/next_benchmarks/tables/matched_selector_main.tex` (limited, non-fresh aggregation)
- `reports/next_benchmarks/tables/diagnostic_prediction.tex` (negative result)
- `reports/next_benchmarks/tables/pretrained_merge.tex` (smoke only)

The manuscript itself was not edited.
