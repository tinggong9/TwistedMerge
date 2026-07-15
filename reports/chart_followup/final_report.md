# Focused chart follow-up program

Mode: full. Overall status: completed. Execution commit: `d57d4fcf04141e3c44001bf9c7e576d9c80007ee`.

## Stage status and protocol coverage

- `ablation`: completed; 170 run rows, 10 seeds, 17 methods; artifacts: `reports/chart_followup/ablation/`.
- `zeroshot`: completed; 220 run rows, 10 seeds, 11 methods; artifacts: `reports/chart_followup/zeroshot/`.
- `cifar`: completed; 60 run rows, 5 seeds, 12 methods; artifacts: `reports/chart_followup/cifar/`.
- `cost`: completed; 480 run rows, 10 seeds, 12 methods; artifacts: `reports/chart_followup/cost/`.
- `compression`: completed; 1800 run rows, 5 seeds, 8 methods; artifacts: `reports/chart_followup/compression/`.
- `sample_efficiency`: completed; 180 run rows, 5 seeds, 6 methods; artifacts: `reports/chart_followup/sample_efficiency/`.

## Exact commands and execution commits

- `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/chart_component_ablation.py` — commit `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`, seed scope `20:29`, exit 0, runtime 301.2850077501498 s, state completed.
- `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/strict_zeroshot_chart_generalization.py` — commit `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`, seed scope `30:39`, exit 0, runtime 398.25850004097447 s, state completed.
- `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/cifar10_chart_retransport.py` — commit `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`, seed scope `0:4 discovery; 5:9 conditional confirmation`, exit 0, runtime 242.9249251249712 s, state completed.
- `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/fashion_complete_cost_audit.py` — commit `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`, seed scope `20:29 checkpoints`, exit 0, runtime 2106.736336583039 s, state completed.
- `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/compression_claim_reaudit.py` — commit `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`, seed scope `retained executed-artifact ledger`, exit 0, runtime 49.086294000037014 s, state completed.
- `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/chart_sample_efficiency.py` — commit `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`, seed scope `40:44`, exit 0, runtime 202.66765429102816 s, state completed.
- `/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python -m pytest -q tests` — commit `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`, seed scope `not_applicable`, exit 0, runtime 30.885524000041187 s, state completed.

## Numerical results

### ablation

| method | task_accuracy | ece | complete_latency_ms_batch128 | stored_bytes |
| --- | --- | --- | --- | --- |
| single_canonical_raw | 0.2245 | 0.453084287704747 | 1.885858247987926 | 58024 |
| d4_test_time_augmentation | 0.22945000000000002 | 0.2097905794392799 | 23.767104279249907 | 58024 |
| direct_d4_equivariant_task_classifier | 0.6061 | 0.05890103754878777 | 8.795412769541144 | 46920 |
| ordinary_chart_soft_moe | 0.26965000000000006 | 0.09842598963899948 | 8.490241505205631 | 290816 |
| d4_chart_soft_moe | 0.5553 | 0.07496757436559115 | 16.552224825136364 | 243700 |
| d4_chart_hard_branch_selection | 0.5478 | 0.04731475071718296 | 15.86058761458844 | 243700 |
| inverse_transform_one_canonical_expert | 0.65425 | 0.030642559453730633 | 26.709220837801695 | 69628 |
| inverse_transform_four_expert_average | 0.6638 | 0.03609431575439857 | 130.03351262304932 | 243700 |
| canonicalize_pool_retransport | 0.34064999999999995 | 0.25219682815718103 | 140.44116660952568 | 243700 |
| uncertainty_weighted_retransport | 0.35300000000000004 | 0.20104376662396511 | 172.44499986991286 | 243700 |
| abstaining_retransport | 0.34525 | 0.22220610666678225 | 174.693808471784 | 243700 |
| supplied_chart_oracle | 0.73295 | 0.031889177182267386 | 138.75896248500794 | 232096 |
| random_chart_control | 0.26409999999999995 | 0.41253433669008316 | 137.7415747847408 | 232096 |
| wrong_group_action_control | 0.09015 | 0.6399607645568429 | 138.36702487897128 | 243700 |
| wrong_multiplication_order_control | 0.5151 | 0.08346802180686771 | 131.1396831413731 | 243700 |
| ensemble_reference | 0.22535 | 0.44529179520975964 | 8.082541450858116 | 232096 |
| generic_low_rank_context_adapter | 0.31325000000000003 | 0.12098397180011697 | 4.140108567662537 | 121608 |

### zeroshot

| method | element_role | task_accuracy | chart_accuracy | ece |
| --- | --- | --- | --- | --- |
| d4_equivariant_chart_classifier | seen | 0.7120999999999998 | 0.7248000025749206 | 0.03175478807329844 |
| d4_equivariant_chart_classifier | unseen | 0.7152000000000001 | 0.7245000004768372 | 0.03520392329613136 |
| capacity_matched_ordinary_chart_classifier | seen | 0.7173999999999999 | 0.7867000043392182 | 0.032955721165716836 |
| capacity_matched_ordinary_chart_classifier | unseen | 0.24089999999999998 | 0.0 | 0.31079304635611865 |
| augmentation_trained_ordinary_chart_classifier | seen | 0.7165999999999999 | 0.7543000042438507 | 0.034034715715589164 |
| augmentation_trained_ordinary_chart_classifier | unseen | 0.24950000000000006 | 0.0 | 0.2964982360852744 |
| learned_multiplication_table_model | seen | 0.7117 | 0.7282999992370606 | 0.03115707106287462 |
| learned_multiplication_table_model | unseen | 0.7175 | 0.7232999920845031 | 0.03126756685579837 |
| d4_equivariant_task_classifier | seen | 0.6313 |  | 0.08019659370002982 |
| d4_equivariant_task_classifier | unseen | 0.6396000000000001 |  | 0.08620514788431625 |
| structured_hard_retransport | seen | 0.7101000000000001 | 0.7248000025749206 | 0.03706167937122376 |
| structured_hard_retransport | unseen | 0.7123999999999999 | 0.7245000004768372 | 0.03774865208632427 |
| structured_soft_retransport | seen | 0.7120999999999998 | 0.7248000025749206 | 0.03175478807329844 |
| structured_soft_retransport | unseen | 0.7152000000000001 | 0.7245000004768372 | 0.03520392329613136 |
| d4_test_time_augmentation | seen | 0.23339999999999997 |  | 0.19841954436544268 |
| d4_test_time_augmentation | unseen | 0.23560000000000003 |  | 0.19435506074696707 |
| supplied_chart_oracle | seen | 0.7708999999999999 | 1.0 | 0.028997455538889848 |
| supplied_chart_oracle | unseen | 0.7687000000000002 | 1.0 | 0.028539807011209894 |
| random_action_control | seen | 0.25599999999999995 | 0.12249999940395355 | 0.44766976773474954 |
| random_action_control | unseen | 0.2577 | 0.1268999993801117 | 0.4422269295183218 |
| wrong_action_control | seen | 0.07799999999999999 | 0.019299999997019767 | 0.6969740708116173 |
| wrong_action_control | unseen | 0.0823 | 0.013799999933689832 | 0.6666545000815158 |

### cifar

| phase | method | task_accuracy | ece | stored_bytes |
| --- | --- | --- | --- | --- |
| discovery | context_blind_expert_average | 0.286 | 0.05339380318852838 | 235552 |
| discovery | generic_moe | 0.2819 | 0.04538743205466244 | 266240 |
| discovery | generic_low_rank_context_adapter | 0.2383 | 0.041338054546877105 | 94296 |
| discovery | d4_equivariant_chart_soft_routing | 0.3311 | 0.053866293128040435 | 247876 |
| discovery | inferred_canonicalize_pool_retransport | 0.2061 | 0.028431149123413397 | 247876 |
| discovery | d4_equivariant_task_classifier | 0.293 | 0.04969184434893802 | 47640 |
| discovery | d4_test_time_augmentation | 0.25570000000000004 | 0.04181862824627651 | 58888 |
| discovery | one_canonical_after_inferred_inverse | 0.3021 | 0.04834019986914236 | 71212 |
| discovery | supplied_chart_oracle | 0.38089999999999996 | 0.06223874632302665 | 235552 |
| discovery | random_action_control | 0.27120000000000005 | 0.03495591182855398 | 235552 |
| discovery | wrong_action_control | 0.25179999999999997 | 0.04363245857575381 | 247876 |
| discovery | ensemble_reference | 0.26189999999999997 | 0.04308341598146721 | 235552 |

### cost

| method | task_accuracy | complete_path_latency_ms_batch128 | stored_bytes | chart_training_examples |
| --- | --- | --- | --- | --- |
| single_ordinary_cnn | 0.2245 | 2.014499972574413 | 58024 | 0 |
| direct_d4_equivariant_task_cnn | 0.6061 | 8.351145486813039 | 46920 | 0 |
| d4_test_time_augmentation | 0.22945000000000002 | 34.10962491761893 | 58024 | 0 |
| generic_low_rank_context_adapter | 0.31325000000000003 | 6.325395777821541 | 121608 | 1000 |
| generic_moe | 0.26965000000000006 | 16.38337504118681 | 290816 | 1000 |
| d4_chart_soft_routing | 0.5553 | 17.71586446557194 | 243700 | 1000 |
| hard_branch_selection | 0.5478 | 17.8658957593143 | 243700 | 1000 |
| canonicalize_pool_retransport | 0.34064999999999995 | 143.64716678392142 | 243700 | 1000 |
| uncertainty_weighted_retransport | 0.35300000000000004 | 177.5324479676783 | 243700 | 1000 |
| abstaining_retransport | 0.34525 | 177.1927605732344 | 243700 | 1000 |
| ensemble | 0.22535 | 12.527187529485673 | 232096 | 0 |
| supplied_chart_oracle_diagnostic | 0.73295 | 135.69156249286607 | 232096 | 0 |

### sample_efficiency

| chart_label_budget | method | mean_task_accuracy | mean_worst_condition_task_accuracy | mean_chart_accuracy | mean_ece |
| --- | --- | --- | --- | --- | --- |
| 32 | d4_equivariant_chart_cnn | 0.38210000000000005 | 0.2719999998807907 | 0.32049999833106996 | 0.1350503153695273 |
| 32 | capacity_matched_ordinary_chart_cnn | 0.2386 | 0.16959999799728392 | 0.12400000095367432 | 0.16577564784947837 |
| 32 | augmentation_trained_ordinary_chart_cnn | 0.2399 | 0.1727999985218048 | 0.09230000078678131 | 0.16507117988282863 |
| 32 | generic_low_rank_adapter | 0.0985 | 0.06560000106692314 | 0.12400000095367432 | 0.03210391850959416 |
| 32 | structured_retransport | 0.28809999999999997 | 0.231999996304512 | 0.32049999833106996 | 0.16217069764574815 |
| 32 | direct_d4_equivariant_task_cnn | 0.6104 | 0.4984000027179718 |  | 0.07321870101562267 |
| 64 | d4_equivariant_chart_cnn | 0.3849 | 0.27760000228881837 | 0.2831000030040741 | 0.1235167742727779 |
| 64 | capacity_matched_ordinary_chart_cnn | 0.2387 | 0.17439999878406526 | 0.1237999975681305 | 0.1661974694151085 |
| 64 | augmentation_trained_ordinary_chart_cnn | 0.2387 | 0.17039999961853028 | 0.12469999939203262 | 0.1657779931469612 |
| 64 | generic_low_rank_adapter | 0.1247 | 0.09519999846816063 | 0.1237999975681305 | 0.02080678632881465 |
| 64 | structured_retransport | 0.28850000000000003 | 0.2280000001192093 | 0.2831000030040741 | 0.15247880273104547 |
| 64 | direct_d4_equivariant_task_cnn | 0.6104 | 0.4984000027179718 |  | 0.07321870101562267 |
| 128 | d4_equivariant_chart_cnn | 0.4729 | 0.31759999990463256 | 0.3005000054836273 | 0.0906600125265377 |
| 128 | capacity_matched_ordinary_chart_cnn | 0.2974 | 0.19679999947547913 | 0.17740000039339066 | 0.13773693871400555 |
| 128 | augmentation_trained_ordinary_chart_cnn | 0.29769999999999996 | 0.20160000026226044 | 0.20590000301599504 | 0.1354032250313824 |
| 128 | generic_low_rank_adapter | 0.26139999999999997 | 0.20640000104904174 | 0.17740000039339066 | 0.06121653821591143 |
| 128 | structured_retransport | 0.2992 | 0.2287999987602234 | 0.3005000054836273 | 0.19298501309023197 |
| 128 | direct_d4_equivariant_task_cnn | 0.6104 | 0.4984000027179718 |  | 0.07321870101562267 |
| 256 | d4_equivariant_chart_cnn | 0.5484 | 0.3424000024795532 | 0.4916000127792358 | 0.06669330138935504 |
| 256 | capacity_matched_ordinary_chart_cnn | 0.45170000000000005 | 0.31360000371932983 | 0.318900004029274 | 0.08784098266108767 |
| 256 | augmentation_trained_ordinary_chart_cnn | 0.41080000000000005 | 0.2888000011444092 | 0.27520000040531156 | 0.11124842669836459 |
| 256 | generic_low_rank_adapter | 0.3806 | 0.32799999713897704 | 0.318900004029274 | 0.050671863821126505 |
| 256 | structured_retransport | 0.2837 | 0.21200000047683715 | 0.4916000127792358 | 0.25554279091153975 |
| 256 | direct_d4_equivariant_task_cnn | 0.6104 | 0.4984000027179718 |  | 0.07321870101562267 |
| 512 | d4_equivariant_chart_cnn | 0.6307 | 0.36880000233650206 | 0.598800003528595 | 0.04194449967442038 |
| 512 | capacity_matched_ordinary_chart_cnn | 0.5362 | 0.3288000047206879 | 0.4577000021934509 | 0.061592059464523084 |
| 512 | augmentation_trained_ordinary_chart_cnn | 0.5141 | 0.3199999928474426 | 0.3988999903202057 | 0.06839336244865524 |
| 512 | generic_low_rank_adapter | 0.5071 | 0.4296000063419342 | 0.4577000021934509 | 0.043912304227092046 |
| 512 | structured_retransport | 0.3511 | 0.2615999966859818 | 0.598800003528595 | 0.22905807749074447 |
| 512 | direct_d4_equivariant_task_cnn | 0.6104 | 0.4984000027179718 |  | 0.07321870101562267 |
| 1000 | d4_equivariant_chart_cnn | 0.6617 | 0.38640000820159914 | 0.6663999915122986 | 0.03809987533564534 |
| 1000 | capacity_matched_ordinary_chart_cnn | 0.5954 | 0.3440000057220459 | 0.5348000049591064 | 0.048702297980866546 |
| 1000 | augmentation_trained_ordinary_chart_cnn | 0.5618000000000001 | 0.33440000414848325 | 0.4640999972820282 | 0.04990059742815879 |
| 1000 | generic_low_rank_adapter | 0.5938 | 0.49440000057220457 | 0.5348000049591064 | 0.03527622812559827 |
| 1000 | structured_retransport | 0.37560000000000004 | 0.2839999973773956 | 0.6663999915122986 | 0.22131208126715413 |
| 1000 | direct_d4_equivariant_task_cnn | 0.6104 | 0.4984000027179718 |  | 0.07321870101562267 |

Compression: 0/72 grouped storage claims confirmed; 0/72 latency claims evaluable; 0/72 Pareto claims confirmed.

## Paired confidence intervals and component attribution

- `ablation/soft_moe_minus_hard_same_probabilities`: mean delta 0.007499999999999995, 95% paired bootstrap CI [0.002399999999999991, 0.012149999999999994], collections 10.
- `ablation/retransport_minus_hard_same_probabilities`: mean delta -0.20715000000000003, 95% paired bootstrap CI [-0.251, -0.17179875], collections 10.
- `ablation/four_experts_minus_one_after_canonicalization`: mean delta 0.009550000000000025, 95% paired bootstrap CI [-0.0010512500000000155, 0.019850000000000045], collections 10.
- `ablation/retransport_minus_d4_tta`: mean delta 0.11120000000000001, 95% paired bootstrap CI [0.04569875, 0.17045], collections 10.
- `ablation/retransport_minus_direct_equivariant_task`: mean delta -0.26544999999999996, 95% paired bootstrap CI [-0.30580124999999997, -0.23204999999999995], collections 10.
- `zeroshot/structured_minus_capacity_matched_ordinary_unseen`: mean delta 0.47430000000000005, 95% paired bootstrap CI [0.4589, 0.4895025], collections 10.
- `zeroshot/structured_minus_augmented_ordinary_unseen`: mean delta 0.4657, 95% paired bootstrap CI [0.44910000000000005, 0.48249999999999993], collections 10.
- `zeroshot/structured_minus_learned_table_unseen`: mean delta -0.002299999999999991, 95% paired bootstrap CI [-0.008999999999999975, 0.003300000000000003], collections 10.
- `zeroshot/structured_minus_random_action_unseen`: mean delta 0.4574999999999999, 95% paired bootstrap CI [0.43859999999999993, 0.4765025], collections 10.
- `zeroshot/structured_minus_wrong_action_unseen`: mean delta 0.6329, 95% paired bootstrap CI [0.6129, 0.6525000000000001], collections 10.
- `cifar/structured_minus_generic_moe`: mean delta -0.0758, 95% paired bootstrap CI [-0.08559999999999998, -0.06600000000000002], collections 5.
- `cifar/structured_minus_low_rank_adapter`: mean delta -0.03219999999999999, 95% paired bootstrap CI [-0.0409, -0.023499999999999993], collections 5.
- `cifar/structured_minus_direct_equivariant_task`: mean delta -0.08689999999999996, 95% paired bootstrap CI [-0.1001, -0.07409999999999997], collections 5.
- `sample_efficiency/structured_minus_ordinary`: mean delta 0.049499999999999995, 95% paired bootstrap CI [-0.010500000000000015, 0.0998], collections 5.
- `sample_efficiency/structured_minus_ordinary`: mean delta 0.049800000000000004, 95% paired bootstrap CI [-0.004499999999999993, 0.09840000000000002], collections 5.
- `sample_efficiency/structured_minus_ordinary`: mean delta 0.0018000000000000017, 95% paired bootstrap CI [-0.07100000000000001, 0.06640000000000001], collections 5.
- `sample_efficiency/structured_minus_ordinary`: mean delta -0.168, 95% paired bootstrap CI [-0.21080000000000002, -0.1278], collections 5.
- `sample_efficiency/structured_minus_ordinary`: mean delta -0.1851, 95% paired bootstrap CI [-0.21559999999999996, -0.1528], collections 5.
- `sample_efficiency/structured_minus_ordinary`: mean delta -0.2198, 95% paired bootstrap CI [-0.25410000000000005, -0.1739], collections 5.
- `ablation/equivariant_chart_minus_best_ordinary`: mean delta 0.5004999980330467, 95% paired bootstrap CI [0.4837987473234534, 0.5170499921962619].

## Gate status and negative findings

- `ablation/equivariant_chart_benefit`: True.
- `ablation/retransport_benefit`: False.
- `ablation/multi_expert_benefit`: False.
- `ablation/twistedmerge_specific_benefit_over_tta_at_matched_cost`: False.
- `ablation/twistedmerge_specific_benefit_over_direct_equivariant_task_at_matched_cost`: False.
- `ablation/twistedmerge_specific_benefit`: False.
- `ablation/all_saved_logits_label_permutation_invariant`: True.
- `zeroshot/unseen_task_beats_all_ordinary_learned_baselines`: False.
- `zeroshot/heldout_chart_labels_never_exposed_during_fitting`: True.
- `zeroshot/equivariance_error_below_tolerance`: True.
- `zeroshot/multiplication_error_below_tolerance`: True.
- `zeroshot/random_and_wrong_action_controls_fail`: True.
- `zeroshot/strict_zeroshot_gate_passed`: False.
- `cifar/discovery_gate_passed`: False.
- `cifar/confirmation_executed`: False.
- `cifar/confirmation_required_when_discovery_passes`: True.
- `cifar/all_saved_logits_label_permutation_invariant`: True.
- `cost/structured_complete_path_faster_than_single`: False.
- `cost/structured_storage_lower_than_single`: False.
- `cost/structured_on_accuracy_latency_pareto_frontier`: False.
- `cost/structured_on_accuracy_storage_pareto_frontier`: False.
- `cost/all_complete_paths_used_preregistered_timing_repetitions`: True.
- `compression/D4/chart_token_student/0.25`: no overall compression claim confirmed.
- `compression/D4/chart_token_student/0.5`: no overall compression claim confirmed.
- `compression/D4/chart_token_student/0.75`: no overall compression claim confirmed.
- `compression/D4/finite_state_chart_module/0.25`: no overall compression claim confirmed.
- `compression/D4/finite_state_chart_module/0.5`: no overall compression claim confirmed.
- `compression/D4/finite_state_chart_module/0.75`: no overall compression claim confirmed.
- `compression/D4/low_rank_group_generators/0.25`: no overall compression claim confirmed.
- `compression/D4/low_rank_group_generators/0.5`: no overall compression claim confirmed.
- `compression/D4/low_rank_group_generators/0.75`: no overall compression claim confirmed.
- `compression/D4/ordinary_single_model_control/0.25`: no overall compression claim confirmed.
- `compression/D4/ordinary_single_model_control/0.5`: no overall compression claim confirmed.
- `compression/D4/ordinary_single_model_control/0.75`: no overall compression claim confirmed.
- `compression/D4/pruned_structured_student/0.25`: no overall compression claim confirmed.
- `compression/D4/pruned_structured_student/0.5`: no overall compression claim confirmed.
- `compression/D4/pruned_structured_student/0.75`: no overall compression claim confirmed.
- `compression/D4/quantized_structured_student/0.25`: no overall compression claim confirmed.
- `compression/D4/quantized_structured_student/0.5`: no overall compression claim confirmed.
- `compression/D4/quantized_structured_student/0.75`: no overall compression claim confirmed.
- `compression/D4/shared_canonical_backbone_group_head/0.25`: no overall compression claim confirmed.
- `compression/D4/shared_canonical_backbone_group_head/0.5`: no overall compression claim confirmed.
- `compression/D4/shared_canonical_backbone_group_head/0.75`: no overall compression claim confirmed.
- `compression/D4/tensor_factorized_equivariant_head/0.25`: no overall compression claim confirmed.
- `compression/D4/tensor_factorized_equivariant_head/0.5`: no overall compression claim confirmed.
- `compression/D4/tensor_factorized_equivariant_head/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/chart_token_student/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/chart_token_student/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/chart_token_student/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/finite_state_chart_module/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/finite_state_chart_module/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/finite_state_chart_module/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/low_rank_group_generators/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/low_rank_group_generators/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/low_rank_group_generators/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/ordinary_single_model_control/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/ordinary_single_model_control/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/ordinary_single_model_control/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/pruned_structured_student/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/pruned_structured_student/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/pruned_structured_student/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/quantized_structured_student/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/quantized_structured_student/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/quantized_structured_student/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/shared_canonical_backbone_group_head/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/shared_canonical_backbone_group_head/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/shared_canonical_backbone_group_head/0.75`: no overall compression claim confirmed.
- `compression/FashionMNIST/tensor_factorized_equivariant_head/0.25`: no overall compression claim confirmed.
- `compression/FashionMNIST/tensor_factorized_equivariant_head/0.5`: no overall compression claim confirmed.
- `compression/FashionMNIST/tensor_factorized_equivariant_head/0.75`: no overall compression claim confirmed.
- `compression/S3/chart_token_student/0.25`: no overall compression claim confirmed.
- `compression/S3/chart_token_student/0.5`: no overall compression claim confirmed.
- `compression/S3/chart_token_student/0.75`: no overall compression claim confirmed.
- `compression/S3/finite_state_chart_module/0.25`: no overall compression claim confirmed.
- `compression/S3/finite_state_chart_module/0.5`: no overall compression claim confirmed.
- `compression/S3/finite_state_chart_module/0.75`: no overall compression claim confirmed.
- `compression/S3/low_rank_group_generators/0.25`: no overall compression claim confirmed.
- `compression/S3/low_rank_group_generators/0.5`: no overall compression claim confirmed.
- `compression/S3/low_rank_group_generators/0.75`: no overall compression claim confirmed.
- `compression/S3/ordinary_single_model_control/0.25`: no overall compression claim confirmed.
- `compression/S3/ordinary_single_model_control/0.5`: no overall compression claim confirmed.
- `compression/S3/ordinary_single_model_control/0.75`: no overall compression claim confirmed.
- `compression/S3/pruned_structured_student/0.25`: no overall compression claim confirmed.
- `compression/S3/pruned_structured_student/0.5`: no overall compression claim confirmed.
- `compression/S3/pruned_structured_student/0.75`: no overall compression claim confirmed.
- `compression/S3/quantized_structured_student/0.25`: no overall compression claim confirmed.
- `compression/S3/quantized_structured_student/0.5`: no overall compression claim confirmed.
- `compression/S3/quantized_structured_student/0.75`: no overall compression claim confirmed.
- `compression/S3/shared_canonical_backbone_group_head/0.25`: no overall compression claim confirmed.
- `compression/S3/shared_canonical_backbone_group_head/0.5`: no overall compression claim confirmed.
- `compression/S3/shared_canonical_backbone_group_head/0.75`: no overall compression claim confirmed.
- `compression/S3/tensor_factorized_equivariant_head/0.25`: no overall compression claim confirmed.
- `compression/S3/tensor_factorized_equivariant_head/0.5`: no overall compression claim confirmed.
- `compression/S3/tensor_factorized_equivariant_head/0.75`: no overall compression claim confirmed.

Strict zero-shot status is the recorded `zeroshot/strict_zeroshot_gate_passed` value. CIFAR transfer status is the recorded discovery gate and conditional-confirmation status. End-to-end cost status is determined only from complete-path timing in `cost/`. Storage, latency, and Pareto compression statuses are separate fields in `compression/claims.csv`.

## Artifact index

Machine-readable paths and SHA-256 values are in `reports/chart_followup/experiment_manifest.csv`, `experiment_manifest.json`, and `artifact_checksums.csv`. Local unpublished checkpoints are listed in `checkpoint_manifest.csv`.
