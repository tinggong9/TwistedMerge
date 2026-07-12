# Sequential Quotient Lift Report

## Exact Command

```bash
.venv/bin/python experiments/sequential_quotient_lift_benchmark.py --reports-dir reports
```

## Evidence Decision

D. The genuine consecutive lift could not be implemented or evaluated completely; list exact blockers.

## Implementation Status

- Controlled prediction-level consecutive quotient lifting is implemented and measured.
- Exact quotient discovery uses multiplication-table homomorphisms for small exact groups and the ambient permutation sign character for truncated permutation groups.
- Natural MNIST q-driven prediction tensors were not available or constructed in this run.
- Real `twisted_rank_lift_2` remains a disagreement-clustering branch ensemble baseline, not a certified quotient-driven lift.
- No real q=4/q=8/depth>1 lift was executed.
- No Brauer/H2 language is justified by these real data.

## Controlled Group Diagnostics

| group_name | stage_depth | quotient_order | homomorphism_residual | kernel_order | kernel_normal | quotient_certified | bootstrap_stability | bootstrap_method | residual_before | residual_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | 1 | 2 | 0 | 1 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| C2xC2 | 1 | 2 | 0 | 2 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| C2xC2 | 2 | 2 | 0 | 1 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| C4 | 1 | 2 | 0 | 2 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| C4 | 2 | 2 | 0 | 1 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| D4 | 1 | 2 | 0 | 4 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| D4 | 2 | 2 | 0 | 2 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| D4 | 3 | 2 | 0 | 1 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| S3 | 1 | 2 | 0 | 3 | True | True | 1 | fixed_group_chain_signature | 0.5 | 0 |
| S3 | 2 | 3 | 0 | 1 | True | True | 1 | fixed_group_chain_signature | 0.666667 | 0 |

## Controlled Accuracy Summary

| group_name | method | n | mean_validation_accuracy | mean_test_accuracy |
| --- | --- | --- | --- | --- |
| C2 | c2m3_cluster_branch_control | 90 | 0.416269 | 0.416394 |
| C2 | c2m3_synchronized | 90 | 0.307667 | 0.307494 |
| C2 | full_ensemble_upper_bound | 90 | 0.816074 | 0.81795 |
| C2 | greedy_soup | 90 | 0.362463 | 0.361694 |
| C2 | old_twisted_rank_lift_2_disagreement_cluster | 90 | 0.360333 | 0.362633 |
| C2 | one_shot_regular_lift | 90 | 0.54225 | 0.543578 |
| C2 | parameter_matched_wide_model | 90 | 0.442213 | 0.440889 |
| C2 | random_same_branch_count_control | 90 | 0.393009 | 0.392039 |
| C2 | reversed_quotient_order_control | 90 | 0.337556 | 0.339183 |
| C2 | sequential_depth_1_fourier_or_equivariant_pooling | 90 | 0.518907 | 0.518889 |
| C2 | sequential_depth_1_uniform_pooling | 90 | 0.469093 | 0.468056 |
| C2 | sequential_depth_1_validation_router | 90 | 0.539 | 0.539478 |
| C2 | sequential_quotient_lift_validation_router | 90 | 0.540556 | 0.538394 |
| C2 | sequential_validation_selected_depth | 90 | 0.539 | 0.539478 |
| C2 | uniform_pool_sign_destroyed_control | 90 | 0.307046 | 0.305006 |
| C2 | validation_branch_ensemble_control | 90 | 0.438009 | 0.437628 |
| C2 | weight_average | 90 | 0.196556 | 0.196794 |
| C2 | wrong_quotient_control | 90 | 0.327333 | 0.328894 |
| C2 | wrong_quotient_order_control | 90 | 0.318574 | 0.31595 |
| C2xC2 | c2m3_cluster_branch_control | 90 | 0.421315 | 0.422517 |
| C2xC2 | c2m3_synchronized | 90 | 0.307417 | 0.307722 |
| C2xC2 | full_ensemble_upper_bound | 90 | 0.817741 | 0.815789 |
| C2xC2 | greedy_soup | 90 | 0.361963 | 0.363833 |
| C2xC2 | old_twisted_rank_lift_2_disagreement_cluster | 90 | 0.365991 | 0.367289 |
| C2xC2 | one_shot_regular_lift | 90 | 0.661185 | 0.6611 |
| C2xC2 | parameter_matched_wide_model | 90 | 0.441074 | 0.444244 |
| C2xC2 | random_same_branch_count_control | 90 | 0.398074 | 0.400656 |
| C2xC2 | reversed_quotient_order_control | 90 | 0.346 | 0.342517 |
| C2xC2 | sequential_depth_1_fourier_or_equivariant_pooling | 90 | 0.519444 | 0.5196 |
| C2xC2 | sequential_depth_1_uniform_pooling | 90 | 0.472454 | 0.470267 |
| C2xC2 | sequential_depth_1_validation_router | 90 | 0.540389 | 0.539028 |
| C2xC2 | sequential_depth_2_fourier_or_equivariant_pooling | 90 | 0.692593 | 0.694933 |
| C2xC2 | sequential_depth_2_uniform_pooling | 90 | 0.631259 | 0.63195 |
| C2xC2 | sequential_depth_2_validation_router | 90 | 0.722454 | 0.7225 |
| C2xC2 | sequential_quotient_lift_validation_router | 90 | 0.720556 | 0.722822 |
| C2xC2 | sequential_validation_selected_depth | 90 | 0.722454 | 0.7225 |
| C2xC2 | uniform_pool_sign_destroyed_control | 90 | 0.305093 | 0.305472 |
| C2xC2 | validation_branch_ensemble_control | 90 | 0.443176 | 0.445417 |
| C2xC2 | weight_average | 90 | 0.197472 | 0.197022 |
| C2xC2 | wrong_quotient_control | 90 | 0.334194 | 0.33395 |
| C2xC2 | wrong_quotient_order_control | 90 | 0.322167 | 0.326261 |
| C4 | c2m3_cluster_branch_control | 90 | 0.421204 | 0.4228 |
| C4 | c2m3_synchronized | 90 | 0.305611 | 0.306983 |
| C4 | full_ensemble_upper_bound | 90 | 0.815898 | 0.814894 |
| C4 | greedy_soup | 90 | 0.363796 | 0.362783 |
| C4 | old_twisted_rank_lift_2_disagreement_cluster | 90 | 0.368269 | 0.36685 |
| C4 | one_shot_regular_lift | 90 | 0.661963 | 0.659911 |
| C4 | parameter_matched_wide_model | 90 | 0.444519 | 0.442661 |
| C4 | random_same_branch_count_control | 90 | 0.399861 | 0.400061 |
| C4 | reversed_quotient_order_control | 90 | 0.342648 | 0.344639 |
| C4 | sequential_depth_1_fourier_or_equivariant_pooling | 90 | 0.519796 | 0.517389 |
| C4 | sequential_depth_1_uniform_pooling | 90 | 0.471852 | 0.468628 |
| C4 | sequential_depth_1_validation_router | 90 | 0.538491 | 0.5386 |
| C4 | sequential_depth_2_fourier_or_equivariant_pooling | 90 | 0.692 | 0.69405 |
| C4 | sequential_depth_2_uniform_pooling | 90 | 0.63263 | 0.63295 |
| C4 | sequential_depth_2_validation_router | 90 | 0.723963 | 0.721572 |
| C4 | sequential_quotient_lift_validation_router | 90 | 0.722639 | 0.722372 |
| C4 | sequential_validation_selected_depth | 90 | 0.723963 | 0.721572 |
| C4 | uniform_pool_sign_destroyed_control | 90 | 0.306028 | 0.306367 |
| C4 | validation_branch_ensemble_control | 90 | 0.444 | 0.442367 |
| C4 | weight_average | 90 | 0.198444 | 0.198706 |
| C4 | wrong_quotient_control | 90 | 0.334009 | 0.3352 |
| C4 | wrong_quotient_order_control | 90 | 0.324769 | 0.325261 |
| D4 | c2m3_cluster_branch_control | 90 | 0.429574 | 0.430244 |
| D4 | c2m3_synchronized | 90 | 0.307333 | 0.307178 |
| D4 | full_ensemble_upper_bound | 90 | 0.816102 | 0.817722 |
| D4 | greedy_soup | 90 | 0.363463 | 0.361439 |
| D4 | old_twisted_rank_lift_2_disagreement_cluster | 90 | 0.375287 | 0.372989 |
| D4 | one_shot_regular_lift | 90 | 0.76638 | 0.762172 |
| D4 | parameter_matched_wide_model | 90 | 0.442269 | 0.443783 |
| D4 | random_same_branch_count_control | 90 | 0.407444 | 0.408456 |
| D4 | reversed_quotient_order_control | 90 | 0.350843 | 0.350206 |
| D4 | sequential_depth_1_fourier_or_equivariant_pooling | 90 | 0.520824 | 0.519689 |
| D4 | sequential_depth_1_uniform_pooling | 90 | 0.471083 | 0.470717 |
| D4 | sequential_depth_1_validation_router | 90 | 0.540639 | 0.53935 |
| D4 | sequential_depth_2_fourier_or_equivariant_pooling | 90 | 0.693287 | 0.69485 |
| D4 | sequential_depth_2_uniform_pooling | 90 | 0.634296 | 0.630028 |
| D4 | sequential_depth_2_validation_router | 90 | 0.721546 | 0.721961 |
| D4 | sequential_depth_3_fourier_or_equivariant_pooling | 90 | 0.833324 | 0.831978 |
| D4 | sequential_depth_3_uniform_pooling | 90 | 0.771176 | 0.772644 |

## Paired Stats

| source | group_name | noise_level | comparison | n_paired_seeds | mean_delta | ci_low | ci_high | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.235133 | 0.230163 | 0.239684 | supported_controlled |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.177233 | 0.172008 | 0.18313 | supported_controlled |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.151167 | 0.146249 | 0.155575 | supported_controlled |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.101567 | 0.0957333 | 0.108142 | supported_controlled |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.122317 | 0.115955 | 0.128695 | supported_controlled |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.213833 | 0.208241 | 0.218851 | supported_controlled |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.20165 | 0.196047 | 0.206725 | supported_controlled |
| controlled | C2 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | -0.00251667 | -0.00739708 | 0.00272542 | unsupported_or_descriptive |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.22805 | 0.222541 | 0.233544 | supported_controlled |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.175317 | 0.168733 | 0.18105 | supported_controlled |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.141883 | 0.13583 | 0.148432 | supported_controlled |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.09875 | 0.0947817 | 0.103659 | supported_controlled |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.120983 | 0.115758 | 0.12626 | supported_controlled |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.209017 | 0.204707 | 0.21391 | supported_controlled |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.199367 | 0.194125 | 0.204026 | supported_controlled |
| controlled | C2 | 0.25 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | -0.00783333 | -0.0127254 | -0.00265 | unsupported_or_descriptive |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.229517 | 0.225463 | 0.233942 | supported_controlled |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.17755 | 0.173104 | 0.182528 | supported_controlled |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.146017 | 0.141282 | 0.150425 | supported_controlled |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.101983 | 0.0969562 | 0.106875 | supported_controlled |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.1227 | 0.118531 | 0.127617 | supported_controlled |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.20565 | 0.200475 | 0.210995 | supported_controlled |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.196617 | 0.192573 | 0.2006 | supported_controlled |
| controlled | C2 | 0.5 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | -0.0052 | -0.0105192 | -0.000572083 | unsupported_or_descriptive |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.412883 | 0.407482 | 0.418479 | supported_controlled |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.355867 | 0.350831 | 0.36045 | supported_controlled |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.315117 | 0.308797 | 0.320977 | supported_controlled |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.272967 | 0.269014 | 0.277375 | supported_controlled |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.29475 | 0.290049 | 0.30025 | supported_controlled |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.386267 | 0.37994 | 0.392889 | supported_controlled |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.377667 | 0.372107 | 0.382984 | supported_controlled |
| controlled | C2xC2 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | 0.0585667 | 0.0514963 | 0.0645996 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.415933 | 0.410924 | 0.42063 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.358 | 0.35295 | 0.362559 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.325067 | 0.320618 | 0.328869 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.276717 | 0.271273 | 0.281659 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.302417 | 0.298016 | 0.306875 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.392483 | 0.387348 | 0.398009 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.381467 | 0.377562 | 0.385917 | supported_controlled |
| controlled | C2xC2 | 0.25 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | 0.06075 | 0.0568388 | 0.0652683 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.416483 | 0.410824 | 0.422342 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.3631 | 0.358141 | 0.368069 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.326317 | 0.321074 | 0.331417 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.282533 | 0.276875 | 0.288863 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.30375 | 0.299364 | 0.30876 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.387867 | 0.382449 | 0.39311 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.381783 | 0.37765 | 0.386554 | supported_controlled |
| controlled | C2xC2 | 0.5 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | 0.06585 | 0.0607913 | 0.0718279 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.413583 | 0.409366 | 0.417242 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.352117 | 0.347449 | 0.357192 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.318633 | 0.31494 | 0.322379 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.269733 | 0.264748 | 0.27492 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.290767 | 0.285732 | 0.295592 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.383367 | 0.379066 | 0.387484 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.372117 | 0.368108 | 0.376083 | supported_controlled |
| controlled | C4 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | 0.0586667 | 0.0553833 | 0.0618104 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.413767 | 0.408399 | 0.419694 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.3588 | 0.35264 | 0.364611 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.32255 | 0.316248 | 0.32827 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.28105 | 0.277125 | 0.284712 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.300567 | 0.295683 | 0.305137 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.387633 | 0.383037 | 0.392378 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.378817 | 0.37225 | 0.384985 | supported_controlled |
| controlled | C4 | 0.25 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | 0.06315 | 0.0588746 | 0.0677088 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.418817 | 0.415041 | 0.422634 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.36785 | 0.364141 | 0.371942 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.32575 | 0.321454 | 0.330392 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.289233 | 0.28454 | 0.293967 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.307383 | 0.302488 | 0.31237 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.390517 | 0.386583 | 0.3946 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.382267 | 0.378391 | 0.386103 | supported_controlled |
| controlled | C4 | 0.5 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | 0.0655667 | 0.0608229 | 0.0701588 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - c2m3_synchronized | 30 | 0.53865 | 0.534858 | 0.542978 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - greedy_soup | 30 | 0.48415 | 0.480275 | 0.487817 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - random_same_branch_count_control | 30 | 0.434417 | 0.429916 | 0.438959 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - validation_branch_ensemble_control | 30 | 0.388217 | 0.382766 | 0.393036 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - c2m3_cluster_branch_control | 30 | 0.411 | 0.406049 | 0.416267 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - wrong_quotient_control | 30 | 0.499617 | 0.49555 | 0.503913 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - reversed_quotient_order_control | 30 | 0.4903 | 0.485786 | 0.494667 | supported_controlled |
| controlled | D4 | 0 | sequential_quotient_lift_validation_router - one_shot_regular_lift | 30 | 0.0879167 | 0.0829063 | 0.0926779 | supported_controlled |

## Natural MNIST Artifact Scan

The requested natural experiment asks for N=6/N=8 with at least 30 seeds.  Current fixed-setting artifacts contain only N=3/N=4 maps, so the requested full natural run was not silently reduced.  The scan records available smaller artifacts as diagnostics only.

_No rows._

| run_id | method | lift_implemented | claim_boundary |
| --- | --- | --- | --- |
|  | sequential_quotient_lift_validation_router | False | requested_n_models_[6, 8]_missing_available_[3, 4] |

## Controls

| source | group_name | noise_level | method | n | mean_validation_accuracy | mean_test_accuracy | mean_branch_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| controlled | C2 | 0 | c2m3_cluster_branch_control | 30 | 0.453972 | 0.455583 | 2 |
| controlled | C2 | 0 | one_shot_regular_lift | 30 | 0.575472 | 0.580417 | 2 |
| controlled | C2 | 0 | parameter_matched_wide_model | 30 | 0.47875 | 0.480217 | 1 |
| controlled | C2 | 0 | random_same_branch_count_control | 30 | 0.428 | 0.426733 | 2 |
| controlled | C2 | 0 | reversed_quotient_order_control | 30 | 0.374056 | 0.37625 | 2 |
| controlled | C2 | 0 | uniform_pool_sign_destroyed_control | 30 | 0.339139 | 0.340967 | 1 |
| controlled | C2 | 0 | validation_branch_ensemble_control | 30 | 0.475389 | 0.476333 | 2 |
| controlled | C2 | 0 | wrong_quotient_control | 30 | 0.364361 | 0.364067 | 2 |
| controlled | C2 | 0 | wrong_quotient_order_control | 30 | 0.354056 | 0.352917 | 2 |
| controlled | C2 | 0.25 | c2m3_cluster_branch_control | 30 | 0.416417 | 0.41485 | 2 |
| controlled | C2 | 0.25 | one_shot_regular_lift | 30 | 0.542861 | 0.543667 | 2 |
| controlled | C2 | 0.25 | parameter_matched_wide_model | 30 | 0.443361 | 0.439433 | 1 |
| controlled | C2 | 0.25 | random_same_branch_count_control | 30 | 0.394639 | 0.39395 | 2 |
| controlled | C2 | 0.25 | reversed_quotient_order_control | 30 | 0.336167 | 0.336467 | 2 |
| controlled | C2 | 0.25 | uniform_pool_sign_destroyed_control | 30 | 0.304222 | 0.302283 | 1 |
| controlled | C2 | 0.25 | validation_branch_ensemble_control | 30 | 0.440472 | 0.437083 | 2 |
| controlled | C2 | 0.25 | wrong_quotient_control | 30 | 0.326194 | 0.326817 | 2 |
| controlled | C2 | 0.25 | wrong_quotient_order_control | 30 | 0.318583 | 0.313283 | 2 |
| controlled | C2 | 0.5 | c2m3_cluster_branch_control | 30 | 0.378417 | 0.37875 | 2 |
| controlled | C2 | 0.5 | one_shot_regular_lift | 30 | 0.508417 | 0.50665 | 2 |
| controlled | C2 | 0.5 | parameter_matched_wide_model | 30 | 0.404528 | 0.403017 | 1 |
| controlled | C2 | 0.5 | random_same_branch_count_control | 30 | 0.356389 | 0.355433 | 2 |
| controlled | C2 | 0.5 | reversed_quotient_order_control | 30 | 0.302444 | 0.304833 | 2 |
| controlled | C2 | 0.5 | uniform_pool_sign_destroyed_control | 30 | 0.277778 | 0.271767 | 1 |
| controlled | C2 | 0.5 | validation_branch_ensemble_control | 30 | 0.398167 | 0.399467 | 2 |
| controlled | C2 | 0.5 | wrong_quotient_control | 30 | 0.291444 | 0.2958 | 2 |
| controlled | C2 | 0.5 | wrong_quotient_order_control | 30 | 0.283083 | 0.28165 | 2 |
| controlled | C2xC2 | 0 | c2m3_cluster_branch_control | 30 | 0.459417 | 0.4609 | 4 |
| controlled | C2xC2 | 0 | one_shot_regular_lift | 30 | 0.699028 | 0.697083 | 4 |
| controlled | C2xC2 | 0 | parameter_matched_wide_model | 30 | 0.478861 | 0.478933 | 1 |
| controlled | C2xC2 | 0 | random_same_branch_count_control | 30 | 0.43675 | 0.440533 | 4 |
| controlled | C2xC2 | 0 | reversed_quotient_order_control | 30 | 0.382778 | 0.377983 | 4 |
| controlled | C2xC2 | 0 | uniform_pool_sign_destroyed_control | 30 | 0.33875 | 0.3418 | 1 |
| controlled | C2xC2 | 0 | validation_branch_ensemble_control | 30 | 0.480389 | 0.482683 | 4 |
| controlled | C2xC2 | 0 | wrong_quotient_control | 30 | 0.371861 | 0.369383 | 4 |
| controlled | C2xC2 | 0 | wrong_quotient_order_control | 30 | 0.355222 | 0.362117 | 4 |
| controlled | C2xC2 | 0.25 | c2m3_cluster_branch_control | 30 | 0.422111 | 0.421283 | 4 |
| controlled | C2xC2 | 0.25 | one_shot_regular_lift | 30 | 0.659167 | 0.66295 | 4 |
| controlled | C2xC2 | 0.25 | parameter_matched_wide_model | 30 | 0.442056 | 0.447167 | 1 |
| controlled | C2xC2 | 0.25 | random_same_branch_count_control | 30 | 0.398861 | 0.398633 | 4 |
| controlled | C2xC2 | 0.25 | reversed_quotient_order_control | 30 | 0.347694 | 0.342233 | 4 |
| controlled | C2xC2 | 0.25 | uniform_pool_sign_destroyed_control | 30 | 0.306278 | 0.3046 | 1 |
| controlled | C2xC2 | 0.25 | validation_branch_ensemble_control | 30 | 0.442778 | 0.446983 | 4 |
| controlled | C2xC2 | 0.25 | wrong_quotient_control | 30 | 0.331556 | 0.331217 | 4 |
| controlled | C2xC2 | 0.25 | wrong_quotient_order_control | 30 | 0.318611 | 0.326167 | 4 |
| controlled | C2xC2 | 0.5 | c2m3_cluster_branch_control | 30 | 0.382417 | 0.385367 | 4 |
| controlled | C2xC2 | 0.5 | one_shot_regular_lift | 30 | 0.625361 | 0.623267 | 4 |
| controlled | C2xC2 | 0.5 | parameter_matched_wide_model | 30 | 0.402306 | 0.406633 | 1 |
| controlled | C2xC2 | 0.5 | random_same_branch_count_control | 30 | 0.358611 | 0.3628 | 4 |
| controlled | C2xC2 | 0.5 | reversed_quotient_order_control | 30 | 0.307528 | 0.307333 | 4 |
| controlled | C2xC2 | 0.5 | uniform_pool_sign_destroyed_control | 30 | 0.27025 | 0.270017 | 1 |
| controlled | C2xC2 | 0.5 | validation_branch_ensemble_control | 30 | 0.406361 | 0.406583 | 4 |
| controlled | C2xC2 | 0.5 | wrong_quotient_control | 30 | 0.299167 | 0.30125 | 4 |
| controlled | C2xC2 | 0.5 | wrong_quotient_order_control | 30 | 0.292667 | 0.2905 | 4 |
| controlled | C4 | 0 | c2m3_cluster_branch_control | 30 | 0.461444 | 0.462683 | 4 |
| controlled | C4 | 0 | one_shot_regular_lift | 30 | 0.692472 | 0.694783 | 4 |
| controlled | C4 | 0 | parameter_matched_wide_model | 30 | 0.478444 | 0.478433 | 1 |
| controlled | C4 | 0 | random_same_branch_count_control | 30 | 0.438083 | 0.434817 | 4 |
| controlled | C4 | 0 | reversed_quotient_order_control | 30 | 0.378306 | 0.381333 | 4 |
| controlled | C4 | 0 | uniform_pool_sign_destroyed_control | 30 | 0.339667 | 0.338233 | 1 |
| controlled | C4 | 0 | validation_branch_ensemble_control | 30 | 0.482167 | 0.483717 | 4 |
| controlled | C4 | 0 | wrong_quotient_control | 30 | 0.369639 | 0.370083 | 4 |
| controlled | C4 | 0 | wrong_quotient_order_control | 30 | 0.36225 | 0.360383 | 4 |
| controlled | C4 | 0.25 | c2m3_cluster_branch_control | 30 | 0.422583 | 0.4223 | 4 |
| controlled | C4 | 0.25 | one_shot_regular_lift | 30 | 0.664694 | 0.659717 | 4 |
| controlled | C4 | 0.25 | parameter_matched_wide_model | 30 | 0.446278 | 0.4412 | 1 |
| controlled | C4 | 0.25 | random_same_branch_count_control | 30 | 0.401806 | 0.400317 | 4 |
| controlled | C4 | 0.25 | reversed_quotient_order_control | 30 | 0.34125 | 0.34405 | 4 |
| controlled | C4 | 0.25 | uniform_pool_sign_destroyed_control | 30 | 0.304667 | 0.30475 | 1 |
| controlled | C4 | 0.25 | validation_branch_ensemble_control | 30 | 0.442639 | 0.441817 | 4 |
| controlled | C4 | 0.25 | wrong_quotient_control | 30 | 0.335194 | 0.335233 | 4 |
| controlled | C4 | 0.25 | wrong_quotient_order_control | 30 | 0.323111 | 0.323967 | 4 |
| controlled | C4 | 0.5 | c2m3_cluster_branch_control | 30 | 0.379583 | 0.383417 | 4 |
| controlled | C4 | 0.5 | one_shot_regular_lift | 30 | 0.628722 | 0.625233 | 4 |
| controlled | C4 | 0.5 | parameter_matched_wide_model | 30 | 0.408833 | 0.40835 | 1 |
| controlled | C4 | 0.5 | random_same_branch_count_control | 30 | 0.359694 | 0.36505 | 4 |
| controlled | C4 | 0.5 | reversed_quotient_order_control | 30 | 0.308389 | 0.308533 | 4 |
| controlled | C4 | 0.5 | uniform_pool_sign_destroyed_control | 30 | 0.27375 | 0.276117 | 1 |
| controlled | C4 | 0.5 | validation_branch_ensemble_control | 30 | 0.407194 | 0.401567 | 4 |
| controlled | C4 | 0.5 | wrong_quotient_control | 30 | 0.297194 | 0.300283 | 4 |

## Required Questions

1. Was the previous negative result mainly mathematical, implementation-related, or both?  Both: the old real q=2 baseline was implementation-limited, and current natural artifacts do not provide stable certified consecutive quotients plus quotient-routed prediction tensors.
2. Did the old q=2 improvement survive all same-branch and wrong-quotient controls?  No. Existing reports and this audit treat it as not surviving all controls.
3. Was any real depth greater than 1 actually executed?  No.
4. Did sequential lifting outperform a one-shot lift of equal branch capacity?  Controlled rows report this comparison; natural rows do not evaluate it.
5. Did it beat C2M3?  Controlled rows report this comparison; natural quotient-driven lift was not evaluated.
6. Did it beat greedy soup?  Controlled rows report this comparison separately; natural quotient-driven lift was not evaluated.
7. Is any Brauer/H2 language justified by the real data?  No.

## Blockers

- No current N=6/N=8 natural MNIST triangle-map artifacts were found.
- No real quotient-routed prediction tensors were constructed for natural MNIST.
- No parameter-level quotient lift was implemented.
- Existing q=2 branch rows are disagreement-cluster branch ensembles, not quotient-sheet transports.

Final decision: D. The genuine consecutive lift could not be implemented or evaluated completely; list exact blockers.
