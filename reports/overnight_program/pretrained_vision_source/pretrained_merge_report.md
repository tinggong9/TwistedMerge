# Modern Shared-Base Pretrained Model-Merging Smoke Report

Decision: **not run at full required scale due to exact blockers; smoke completed**.

## Exact command

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python /Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-stage4/experiments/pretrained_merge_smoke.py --seed 0 --train-samples 512 --validation-samples 256 --test-samples 512 --head-epochs 30 --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data --out-dir /Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-stage4/reports/overnight_program/pretrained_vision_source
```

- Git commit at execution: `66f7eaf811a57d4e94e1dd699c1025dc2fa4ea03`
- Backbone: torchvision ResNet-18 with `IMAGENET1K_V1` ImageNet weights (`https://download.pytorch.org/models/resnet18-f37072fd.pth`)
- Dataset: CIFAR-10
- Four task heads: classes 0-4, classes 5-9, even labels, odd labels
- Shared base: identical frozen pretrained backbone and identical initialized linear head
- Smoke samples: train `512`, validation `256`, test `512`
- Seeds: one (`0`), below the required five

## Smoke results

| seed | method | average_accuracy | task_0_accuracy | task_1_accuracy | task_2_accuracy | task_3_accuracy | worst_task_accuracy | forgetting_interference | calibration_ece | parameter_count | actual_trainable_parameters | stored_parameters | parameter_multiplier | inference_multiplier | measured_inference_time_seconds | selection_budget | selected_by_validation | selector_source_method | branch_candidate_activated | obstruction_certificate_passed | implementation_status | saved_logits_path | saved_logits_sha256 | label_permutation_regression_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | weight_average | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.0546974 | 11181642 | 5130 | 11181642 | 1 | 1 | 0.000500792 | 0 | False |  | False | False | internal_faithful_smoke | reports/overnight_program/pretrained_vision_source/logits/pretrained_vision_seed0.npz | cfaa5aa1966e5246939029d589bde0f66188d5acb0976e2efe1b7d0b8a0e8668 | True |
| 0 | greedy_soup | 0.578125 | 0.608209 | 0.545082 | 0.610039 | 0.545455 | 0.545082 | 0.17703 | 0.0391626 | 11181642 | 5130 | 11181642 | 1 | 1.02188 | 0.00051175 | 256 | False |  | False | False | internal_faithful_smoke | reports/overnight_program/pretrained_vision_source/logits/pretrained_vision_seed0.npz | cfaa5aa1966e5246939029d589bde0f66188d5acb0976e2efe1b7d0b8a0e8668 | True |
| 0 | task_arithmetic | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.0546974 | 11181642 | 5130 | 11181642 | 1 | 0.995841 | 0.000498709 | 0 | False |  | False | False | internal_faithful_smoke | reports/overnight_program/pretrained_vision_source/logits/pretrained_vision_seed0.npz | cfaa5aa1966e5246939029d589bde0f66188d5acb0976e2efe1b7d0b8a0e8668 | True |
| 0 | ties | 0.541016 | 0.563433 | 0.516393 | 0.640927 | 0.438735 | 0.438735 | 0.214354 | 0.231467 | 11181642 | 5130 | 11181642 | 1 | 0.948913 | 0.000475208 | 0 | False |  | False | False | internal_faithful_smoke | reports/overnight_program/pretrained_vision_source/logits/pretrained_vision_seed0.npz | cfaa5aa1966e5246939029d589bde0f66188d5acb0976e2efe1b7d0b8a0e8668 | True |
| 0 | dare | 0.521484 | 0.507463 | 0.536885 | 0.644788 | 0.395257 | 0.395257 | 0.233128 | 0.0773216 | 11181642 | 5130 | 11181642 | 1 | 0.965053 | 0.000483291 | 0 | False |  | False | False | internal_faithful_smoke | reports/overnight_program/pretrained_vision_source/logits/pretrained_vision_seed0.npz | cfaa5aa1966e5246939029d589bde0f66188d5acb0976e2efe1b7d0b8a0e8668 | True |
| 0 | slerp | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.112756 | 11181642 | 5130 | 11181642 | 1 | 0.948663 | 0.000475083 | 0 | False |  | False | False | internal_faithful_smoke | reports/overnight_program/pretrained_vision_source/logits/pretrained_vision_seed0.npz | cfaa5aa1966e5246939029d589bde0f66188d5acb0976e2efe1b7d0b8a0e8668 | True |
| 0 | twistedmerge_exact_gauge_soup_selector | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.0546974 | 11181642 | 5130 | 11181642 | 1 | 0.965636 | 0.000483583 | 256 | True | weight_average | False | False | internal_faithful_smoke | reports/overnight_program/pretrained_vision_source/logits/pretrained_vision_seed0.npz | cfaa5aa1966e5246939029d589bde0f66188d5acb0976e2efe1b7d0b8a0e8668 | True |

## Exact blockers to a full ICLR/JMLR benchmark

1. The required five-seed, full fine-tuning protocol was not computationally justified for this package run; this smoke freezes the backbone and fine-tunes only task heads.
2. Official Task Arithmetic, TIES, DARE, and SLERP repositories/licenses/commits were not pinned and integrated. The smoke uses labeled internal faithful implementations, which are not publication-grade external-baseline reproductions.
3. The full protocol needs separate validation/test sets at useful scale and paired statistics across at least five seeds; one smoke seed cannot support an accuracy claim.
4. No exact centrality/closure certificate passed, so no central/Brauer obstruction or branch candidate is claimed.

The checkpoint files and raw CSVs are retained only as feasibility evidence. They are excluded from paper-number release eligibility.
