# Same-Base Task-Vector Extended Replication

Generated from the committed benchmark series 31 same-base CSVs plus an optional benchmark series 37 extension run.

## Scope

This is a same-base task-vector replication only. It does not mix with independent-seed/rebasin diagnostics and does not certify Brauer/projective obstruction.

Added settings:

- `mnist_mod3_subsets`, width 64, seeds `7400:7420`.
- `fashion_semantic_subsets`, width 64, seeds `7400:7420`.

Width 256 was not run. Runtime was acceptable for the two added width-64 task families, and the optional width-256 branch was skipped under the stop condition.

## Exact Commands

Original benchmark series 31 source rows are read from `reports/csv/same_base_task_vector_benchmark.csv`.

Additional raw extension run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache MPLCONFIGDIR=/private/tmp/codex-mpl \
.venv/bin/python - <<'PY'
# Runtime-patched TASK_PRESETS:
# mnist_mod3_subsets = digits_mod0/digits_mod1/digits_mod2
# fashion_semantic_subsets = upper_garments/lower_formal/footwear_bag
# Then call experiments.same_base_task_vector_benchmark.main() with:
# --datasets mnist,fashion_mnist
# --task-presets mnist_mod3_subsets,fashion_semantic_subsets
# --widths 64 --seeds 7400:7420
# --base-epochs 3 --finetune-epochs 2
# --max-train-samples 6000 --max-test-samples 2000
# --max-base-train-samples 5000 --max-task-train-samples 1800
# --max-task-val-samples 600 --max-task-test-samples 600
# --batch-size 128 --bootstrap-samples 1000 --device auto
# --reports-dir /private/tmp/twistedmerge_same_base_extended/raw
# --no-update-claims-audit
PY
```

Final merge/summarization command was the Python merge step that read the original CSV plus `/private/tmp/twistedmerge_same_base_extended/raw/csv/same_base_task_vector_benchmark.csv`, then wrote the `*_extended` artifacts.

## Outputs

- `reports/same_base_task_vector_extended.md`
- `reports/csv/same_base_task_vector_extended.csv`
- `reports/csv/same_base_task_vector_extended_summary.csv`
- `reports/tables/same_base_task_vector_extended.tex`

## Headline

Best comparable single-model mean accuracy in the extended table is `task_arithmetic` on `mnist/mnist_digit_subsets/W128` at 0.8839.

Claim-decision counts across exact settings: `{'supported_exact_setting_delta_vs_greedy': 6, 'descriptive_best_mean_ci_overlaps_greedy': 2}`.

## Completed Exact Settings

| dataset | task_preset | width | n_tasks | n_unique_seeds | best_single_model_method_by_mean_accuracy | claim_decision |
| --- | --- | --- | --- | --- | --- | --- |
| fashion_mnist | fashion_class_subsets | 64 | 3 | 20 | dare | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | 3 | 20 | dare | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | 3 | 20 | dare | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | 3 | 20 | task_arithmetic | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | 3 | 20 | task_arithmetic | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_four_subsets | 64 | 4 | 20 | task_arithmetic | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | 4 | 20 | task_arithmetic | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_mod3_subsets | 64 | 3 | 20 | task_arithmetic | supported_exact_setting_delta_vs_greedy |

## Added Task-Family Results

| dataset | task_preset | width | method | n_unique_seeds | mean_average_test_accuracy | accuracy_ci_low | accuracy_ci_high | mean_worst_task_accuracy | mean_delta_vs_greedy_soup | delta_vs_greedy_ci_low | delta_vs_greedy_ci_high | claim_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fashion_mnist | fashion_semantic_subsets | 64 | dare | 20 | 0.7460 | 0.7422 | 0.7496 | 0.5464 | 0.0041 | 0.0021 | 0.0069 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | greedy_soup | 20 | 0.7419 | 0.7379 | 0.7457 | 0.5354 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | individual_finetuned_mean | 20 | 0.8170 | 0.8143 | 0.8193 | 0.6252 | 0.0752 | 0.0717 | 0.0793 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | slerp_sequential | 20 | 0.7419 | 0.7380 | 0.7457 | 0.5357 | 0.0000 | -0.0001 | 0.0002 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | task_arithmetic | 20 | 0.7453 | 0.7416 | 0.7491 | 0.5443 | 0.0035 | 0.0014 | 0.0065 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | ties_merging | 20 | 0.7398 | 0.7363 | 0.7433 | 0.5459 | -0.0021 | -0.0043 | 0.0007 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | weight_average | 20 | 0.7419 | 0.7379 | 0.7457 | 0.5354 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | dare | 20 | 0.8492 | 0.8458 | 0.8526 | 0.7857 | 0.0007 | -0.0009 | 0.0023 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | greedy_soup | 20 | 0.8485 | 0.8455 | 0.8519 | 0.7816 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | individual_finetuned_mean | 20 | 0.9501 | 0.9490 | 0.9511 | 0.9261 | 0.1015 | 0.0988 | 0.1044 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | slerp_sequential | 20 | 0.8485 | 0.8455 | 0.8518 | 0.7814 | -0.0000 | -0.0001 | 0.0001 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | task_arithmetic | 20 | 0.8505 | 0.8477 | 0.8535 | 0.7904 | 0.0020 | 0.0004 | 0.0036 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | ties_merging | 20 | 0.8490 | 0.8456 | 0.8527 | 0.7872 | 0.0005 | -0.0011 | 0.0021 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | weight_average | 20 | 0.8485 | 0.8455 | 0.8519 | 0.7816 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |

## Full Method Summary

| dataset | task_preset | width | method | n_unique_seeds | mean_average_test_accuracy | mean_worst_task_accuracy | mean_delta_vs_greedy_soup | delta_vs_greedy_ci_low | delta_vs_greedy_ci_high | claim_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fashion_mnist | fashion_class_subsets | 64 | dare | 20 | 0.6936 | 0.6419 | 0.0109 | 0.0048 | 0.0161 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 64 | greedy_soup | 20 | 0.6828 | 0.6403 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 64 | individual_finetuned_mean | 20 | 0.7779 | 0.6743 | 0.0951 | 0.0888 | 0.1015 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 64 | slerp_sequential | 20 | 0.6832 | 0.6396 | 0.0005 | -0.0004 | 0.0019 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 64 | task_arithmetic | 20 | 0.6932 | 0.6438 | 0.0105 | 0.0057 | 0.0148 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 64 | ties_merging | 20 | 0.6462 | 0.5573 | -0.0366 | -0.0482 | -0.0249 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 64 | weight_average | 20 | 0.6831 | 0.6393 | 0.0004 | -0.0007 | 0.0018 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | dare | 20 | 0.7279 | 0.6762 | 0.0316 | 0.0259 | 0.0372 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | greedy_soup | 20 | 0.6963 | 0.6212 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | individual_finetuned_mean | 20 | 0.8019 | 0.6977 | 0.1055 | 0.0998 | 0.1114 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | slerp_sequential | 20 | 0.6974 | 0.6157 | 0.0011 | -0.0010 | 0.0046 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | task_arithmetic | 20 | 0.7266 | 0.6712 | 0.0302 | 0.0236 | 0.0365 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | ties_merging | 20 | 0.6433 | 0.5137 | -0.0530 | -0.0684 | -0.0375 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_class_subsets | 128 | weight_average | 20 | 0.6973 | 0.6155 | 0.0010 | -0.0011 | 0.0044 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | dare | 20 | 0.7460 | 0.5464 | 0.0041 | 0.0021 | 0.0069 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | greedy_soup | 20 | 0.7419 | 0.5354 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | individual_finetuned_mean | 20 | 0.8170 | 0.6252 | 0.0752 | 0.0717 | 0.0793 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | slerp_sequential | 20 | 0.7419 | 0.5357 | 0.0000 | -0.0001 | 0.0002 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | task_arithmetic | 20 | 0.7453 | 0.5443 | 0.0035 | 0.0014 | 0.0065 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | ties_merging | 20 | 0.7398 | 0.5459 | -0.0021 | -0.0043 | 0.0007 | supported_exact_setting_delta_vs_greedy |
| fashion_mnist | fashion_semantic_subsets | 64 | weight_average | 20 | 0.7419 | 0.5354 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | dare | 20 | 0.8556 | 0.7722 | 0.0392 | 0.0364 | 0.0420 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | greedy_soup | 20 | 0.8164 | 0.6597 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | individual_finetuned_mean | 20 | 0.9267 | 0.8952 | 0.1102 | 0.1072 | 0.1135 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | slerp_sequential | 20 | 0.8166 | 0.6603 | 0.0002 | 0.0001 | 0.0004 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | task_arithmetic | 20 | 0.8603 | 0.7993 | 0.0439 | 0.0407 | 0.0474 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | ties_merging | 20 | 0.8245 | 0.6826 | 0.0081 | 0.0051 | 0.0111 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 64 | weight_average | 20 | 0.8164 | 0.6597 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | dare | 20 | 0.8770 | 0.8072 | 0.0425 | 0.0405 | 0.0445 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | greedy_soup | 20 | 0.8345 | 0.6945 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | individual_finetuned_mean | 20 | 0.9372 | 0.9096 | 0.1027 | 0.0999 | 0.1056 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | slerp_sequential | 20 | 0.8347 | 0.6948 | 0.0001 | 0.0000 | 0.0003 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | task_arithmetic | 20 | 0.8839 | 0.8308 | 0.0494 | 0.0469 | 0.0521 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | ties_merging | 20 | 0.8500 | 0.7362 | 0.0155 | 0.0114 | 0.0196 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_digit_subsets | 128 | weight_average | 20 | 0.8345 | 0.6945 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_four_subsets | 64 | dare | 20 | 0.8530 | 0.8095 | -0.0004 | -0.0014 | 0.0007 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 64 | greedy_soup | 20 | 0.8533 | 0.8112 | 0.0000 | 0.0000 | 0.0000 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 64 | individual_finetuned_mean | 20 | 0.9213 | 0.8927 | 0.0680 | 0.0657 | 0.0703 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 64 | slerp_sequential | 20 | 0.8531 | 0.8107 | -0.0002 | -0.0008 | 0.0001 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 64 | task_arithmetic | 20 | 0.8541 | 0.8132 | 0.0008 | -0.0000 | 0.0016 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 64 | ties_merging | 20 | 0.8451 | 0.7954 | -0.0082 | -0.0119 | -0.0051 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 64 | weight_average | 20 | 0.8531 | 0.8106 | -0.0003 | -0.0008 | 0.0000 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | dare | 20 | 0.8765 | 0.8388 | 0.0001 | -0.0012 | 0.0013 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | greedy_soup | 20 | 0.8764 | 0.8361 | 0.0000 | 0.0000 | 0.0000 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | individual_finetuned_mean | 20 | 0.9328 | 0.9085 | 0.0564 | 0.0544 | 0.0584 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | slerp_sequential | 20 | 0.8764 | 0.8359 | -0.0001 | -0.0002 | 0.0000 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | task_arithmetic | 20 | 0.8774 | 0.8388 | 0.0010 | -0.0004 | 0.0023 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | ties_merging | 20 | 0.8710 | 0.8290 | -0.0054 | -0.0079 | -0.0030 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_four_subsets | 128 | weight_average | 20 | 0.8764 | 0.8361 | 0.0000 | 0.0000 | 0.0000 | descriptive_best_mean_ci_overlaps_greedy |
| mnist | mnist_mod3_subsets | 64 | dare | 20 | 0.8492 | 0.7857 | 0.0007 | -0.0009 | 0.0023 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | greedy_soup | 20 | 0.8485 | 0.7816 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | individual_finetuned_mean | 20 | 0.9501 | 0.9261 | 0.1015 | 0.0988 | 0.1044 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | slerp_sequential | 20 | 0.8485 | 0.7814 | -0.0000 | -0.0001 | 0.0001 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | task_arithmetic | 20 | 0.8505 | 0.7904 | 0.0020 | 0.0004 | 0.0036 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | ties_merging | 20 | 0.8490 | 0.7872 | 0.0005 | -0.0011 | 0.0021 | supported_exact_setting_delta_vs_greedy |
| mnist | mnist_mod3_subsets | 64 | weight_average | 20 | 0.8485 | 0.7816 | 0.0000 | 0.0000 | 0.0000 | supported_exact_setting_delta_vs_greedy |

## Claim Boundaries

- Exact-setting claims only unless support repeats across task families and widths.
- Same-base task-vector methods are not independent-seed rebasin methods.
- This extension does not support broad superiority by itself; it adds two width-64 task-family checks to the original benchmark series 31 table.
- The per-task oracle summary is not a single merged model.
