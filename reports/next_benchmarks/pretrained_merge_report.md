# Modern Shared-Base Pretrained Model-Merging Smoke Report

Decision: **bounded feasibility run completed; the full-scale protocol was not executed**.

## Exact command

```bash
<repository-root>/.venv/bin/python experiments/pretrained_merge_smoke.py --seed 0 --train-samples 512 --validation-samples 256 --test-samples 512 --head-epochs 30
```

- Git commit at execution: `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf`
- Backbone: torchvision ResNet-18 with `IMAGENET1K_V1` ImageNet weights (`https://download.pytorch.org/models/resnet18-f37072fd.pth`)
- Dataset: CIFAR-10
- Four task heads: classes 0-4, classes 5-9, even labels, odd labels
- Shared base: identical frozen pretrained backbone and identical initialized linear head
- Smoke samples: train `512`, validation `256`, test `512`
- Seeds: one (`0`), below the required five

## Smoke results

| seed | method | average_accuracy | task_0_accuracy | task_1_accuracy | task_2_accuracy | task_3_accuracy | worst_task_accuracy | forgetting_interference | calibration_ece | parameter_count | parameter_multiplier | inference_multiplier | selection_budget | selected_by_validation | selector_source_method | branch_candidate_activated | obstruction_certificate_passed | implementation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | weight_average | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.0546974 | 11181642 | 1 | 1 | 0 | False |  | False | False | internal_faithful_smoke |
| 0 | greedy_soup | 0.578125 | 0.608209 | 0.545082 | 0.610039 | 0.545455 | 0.545082 | 0.17703 | 0.0391626 | 11181642 | 1 | 1 | 256 | False |  | False | False | internal_faithful_smoke |
| 0 | task_arithmetic | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.0546974 | 11181642 | 1 | 1 | 0 | False |  | False | False | internal_faithful_smoke |
| 0 | ties | 0.541016 | 0.563433 | 0.516393 | 0.640927 | 0.438735 | 0.438735 | 0.214354 | 0.231467 | 11181642 | 1 | 1 | 0 | False |  | False | False | internal_faithful_smoke |
| 0 | dare | 0.521484 | 0.507463 | 0.536885 | 0.644788 | 0.395257 | 0.395257 | 0.233128 | 0.0773216 | 11181642 | 1 | 1 | 0 | False |  | False | False | internal_faithful_smoke |
| 0 | slerp | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.112756 | 11181642 | 1 | 1 | 0 | False |  | False | False | internal_faithful_smoke |
| 0 | twistedmerge_exact_gauge_soup_selector | 0.597656 | 0.634328 | 0.557377 | 0.675676 | 0.517787 | 0.517787 | 0.157934 | 0.0546974 | 11181642 | 1 | 1 | 256 | True | weight_average | False | False | internal_faithful_smoke |

## Scope and limitations

1. This historical run uses one seed, a frozen backbone, and fine-tuned task heads; it is a feasibility result rather than a full comparison.
2. Task Arithmetic, TIES, DARE, and SLERP are labeled internal implementations; official repositories were not integrated for this run.
3. A full comparison requires disjoint validation and test sets at larger scale, at least five seeds, and paired intervals.
4. No exact centrality/closure certificate passed, so no central/Brauer obstruction or branch candidate is claimed.

The checkpoints and raw CSVs are retained as reproducibility evidence for this bounded run.
