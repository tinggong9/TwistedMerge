# Context Router Generalization Report

Claim decision: **unsupported** for a learned practical router.

## Exact command

```bash
<repository-root>/.venv/bin/python experiments/context_router_generalization.py --groups S3,D4 --seeds 0:19 --n-validation-per-context 200 --n-test-per-context 300
```

- Git commit at execution: `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf`
- Training contexts: `e, s, r, sr`
- Held-out contexts: `rs, ss, rr, srsr, srr, rsr, srs, rrs`
- Held-out word strings are disjoint from training word strings.
- Saved candidate branch logits: `reports/next_benchmarks/logits/context_router_logits.npz`
- Label-permutation regression: `True`

The learned router receives only noisy raw word-feature coordinates and model inputs. It does not receive a context ID or test labels. The face-table router is explicitly a validation diagnostic and falls back to its validation-majority branch for unseen contexts. The supplied-context result is reported separately as an oracle.

## Summary

| group_name | router | n_seeds | mean_context_classification_accuracy | mean_downstream_task_accuracy | unseen_context_accuracy_ci_low | unseen_context_accuracy_ci_high | mean_calibration_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D4 | learned_feature_router | 20 | 0 | 0.22102 | 0.17612 | 0.26069 | 0.40377 |
| D4 | majority_branch | 20 | 0.25 | 0.31323 | 0.30219 | 0.32465 | 0.375 |
| D4 | no_router | 20 | 0.25 | 0.31323 | 0.30252 | 0.32452 | 0.75 |
| D4 | random_router | 20 | 0.12444 | 0.24887 | 0.2455 | 0.25246 | 0.21833 |
| D4 | supplied_context_oracle | 20 | 1 | 1 | 1 | 1 | 0 |
| D4 | validation_face_table_router | 20 | 0.25 | 0.31323 | 0.30202 | 0.3244 | 0.375 |
| S3 | learned_feature_router | 20 | 0 | 0.3754 | 0.29918 | 0.46609 | 0.48021 |
| S3 | majority_branch | 20 | 0.25 | 0.56394 | 0.51741 | 0.60993 | 0.375 |
| S3 | no_router | 20 | 0.25 | 0.56394 | 0.51732 | 0.60992 | 0.75 |
| S3 | random_router | 20 | 0.16581 | 0.49831 | 0.44221 | 0.55577 | 0.27721 |
| S3 | supplied_context_oracle | 20 | 1 | 1 | 1 | 1 | 0 |
| S3 | validation_face_table_router | 20 | 0.25 | 0.56394 | 0.51975 | 0.61384 | 0.375 |

## Claims

| claim_id | status | safe_wording |
| --- | --- | --- |
| learned_router_generalizes_to_unseen_contexts | unsupported | Supplied-context prediction remains valid, but the learned feature router is not supported as a practical unseen-context router. |
| supplied_context_oracle | supported | An executed supplied-context branch predictor selects the planted group action; it is an oracle diagnostic, not a learned router. |

## Safe interpretation

If the learned router fails on held-out words, only the supplied-context oracle is retained. No result in this report licenses calling a validation face table a learned practical router.
