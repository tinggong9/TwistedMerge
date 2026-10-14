# BLOCKED — real Qwen feasibility audit

Audit: `20261014T175957Z`, 2026-10-14T17:59:58.109067+00:00. Source commit:
`358f1cf570d73b74464608a5c31e730a209931cf`. Entry checkout was clean `main`;
reports are prepared on `codex/kbs-resource-readiness-20261014`. This is a readiness audit,
not a real pilot STOP based on measured model outcomes.

The real experiment cannot start. No model was loaded, no training or inference
ran, and no additional synthetic experiment or regression suite was run.
`test_access: false`; no final-test labels were inspected, scored, or unsealed.
The existing evidence and manuscripts are preserved.

## Current measurements

| Item | Measured state |
| --- | --- |
| Host | Apple M1 / MacBookPro17,1; 8 logical CPU cores |
| Physical RAM | 8,589,934,592 bytes (8 GiB) |
| Available RAM snapshot | 1,188,315,136 bytes; changes with other processes |
| Accelerator | MPS available, shares host RAM; CUDA unavailable; no dedicated VRAM |
| MPS recommended memory budget | 5,726,633,984 bytes; framework recommendation, not free VRAM or a training peak |
| Free local disk | 99,550,265,344 bytes; physical capacity does not authorize downloads |
| Retained additional disk ceiling | 200,000,000 bytes, a conservative local plan |
| Newly supplied GPU/storage hosts | None; the user knows of none |

[Environment](environment.json), [native command receipts](host_commands.json),
[bounded local inventory](local_input_inventory.json), and
[repository link status](storage_mount_status.json) record the search limits.
Default caches contain no Qwen copy or four-task benchmark data; repository
`data` and `checkpoints` links have unavailable targets. No remote host was
contacted. This audit is not an exhaustive search of arbitrary personal or
remote storage. MPS availability does not establish feasible real training.

## Exact missing prerequisites

| Requirement | BLOCKED reason | Smallest input needed |
| --- | --- | --- |
| Pinned backbone/tokenizer | `model.local_path` and validated manifest are absent | Existing local Qwen/Qwen2.5-1.5B at revision `8faed761d45a263340a0528343f099c05c9a4323`, tokenizer/config/safetensors hashes and reviewed license |
| Real specialists | `adapters: []`; known rank-2 cohort is random/software-only | Four compatible BoolQ/PIQA/ARC-Challenge/GSM8K LoRAs sharing base/tokenizer/frozen head, rank/alpha/modules/seeds/group IDs and licenses, or an approved fixed real-data training plan |
| Task records/splits | No licensed native records or actual split manifest | Local pinned real records, source split lineage/licenses and seven immutable roles with exact/near-duplicate and contamination audit |
| Resources | No authorized target with measured loading/training feasibility; present memory snapshot below stored base weights alone | Existing approved accelerator/workspace with sufficient measured host/device memory and approved storage |
| Baselines/protocol | `baselines: {}`, no episodes or training recipe | Resolve base/specialists and all eleven methods, weights/ranks/dtypes/density/seeds, specialist-quality criteria and pilot-only selection before compute |

[Machine-readable readiness](readiness.json) records 17 errors from the existing
validator, plus resource/license/recipe gaps and a separate structural-branch limitation.
The structural branch separately remains inapplicable/uncertified without an independently justified neural coefficient/confidence model. An otherwise-ready future pilot may explicitly abstain; this is not an admission blocker.

Backend implementation is already available; the missing baseline definitions
are unresolved real-run configuration, not missing merge kernels.
[Model/cohort status](model_tokenizer_adapter_provenance.json) and
[per-task split status](real_task_split_status.json) contain explicit BLOCKED
entries rather than invented provenance or leakage successes. Preparing splits
requires the real inputs; no `prepare-splits`, `train-cohort`, pilot, freeze or
confirmatory command was executed.

## Bounded proposal, pending resources and approval

Use the existing `train-cohort` backend for one independently trained group of
four rank-8 specialists. Rank 16 is secondary and is not admitted now. Keep the
shared pinned base/head/tokenizer and q/k/v/o targets; batch-one AdamW and a
fixed recipe must be reviewed on actual `train`/`pilot` records. Choose adequate
real task subsets, explicit seeds/steps/learning rate/alpha/dropout/token lengths
and specialist-quality criteria before evaluation. The old four-step generated
fixture recipe is not a research training proposal.

The [pinned public file listing](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)
and bounded metadata receipt establish these storage costs:

| Component | Bytes / capacity |
| --- | --- |
| Stored base safetensors | 3,087,467,144 bytes |
| Base loading files including tokenizer/config | 3,098,955,511 bytes |
| Rank-8 factor parameters per specialist | 2,179,072 |
| Four FP32 adapter factors, raw lower bound | 34,865,152 bytes |
| Base loading files + four FP32 factor lower bound | 3,133,820,663 bytes |

These are file-size/arithmetic bounds, not model memory measurements. They omit
safetensors headers, licensed data, split copies, logs, temporary files and
candidate outputs. An existing approved read-only model mount avoids another
base copy; otherwise even the base exceeds the 200 MB ceiling by over 15 times.
Actual data/output allowances must be measured before resolving a disk cap.

Prefer an existing institutional accelerator. A 24 GB GPU is a planning
preference, not a verified minimum. Loading and a bounded real-data step on an
authorized target must establish device/host peaks and throughput before the
cohort budget is approved. No measured training time, minimum VRAM, GPU-hour
estimate or cloud price is available. Proposed pilot ceiling: at most 8 GPU
hours; the 48-hour campaign entry remains planning only and is outside this
readiness action. Neither number authorizes spending. Current approved neural
compute: zero. No paid provision or large model/data download is authorized.
See [cost proposal](training_cost_proposal.json) and
[resource/license scope](approved_resources_and_licenses.json).

## Smallest action to unblock

Supply paths and access approval for an **existing** workspace containing the
pinned base/tokenizer and licensed four-task records, plus a compatible cohort
or approval to prepare one under a reviewed bounded recipe. That workspace
must have measured feasible memory and storage. If none exists, leave the real
pilot BLOCKED; this run requests no purchase or resource provisioning. Once
inputs exist, first resolve the pilot config, audit the seven roles, preserve
final-test isolation, and measure feasibility. A real pilot GO and separately
reviewed frozen protocol are still required before any confirmatory access.

The [evidence map](arxiv_v2_evidence_map.md) preserves the narrower existing
algebra/software and historical negative results. It supplies no new neural
efficacy, safety, cost-saving, or natural-certificate claim.
