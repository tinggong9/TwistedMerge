# Post-ICLR Official Baseline Integration

This integration was generated from the isolated `codex/post-iclr-experiments` worktree. Official sources were cloned outside the tracked repository and pinned by commit. No third-party source is vendored.

## Source ledger

| Baseline | Repository | Commit | License | Clone status | Dirty after required patch |
| --- | --- | --- | --- | --- | --- |
| `official_git_rebasin` | https://github.com/samuela/git-re-basin.git | `ef40098257ab97243930eba737d6dcb8edd5863e` | MIT | source_pinned | False |
| `official_c2m3` | https://github.com/crisostomi/cycle-consistent-model-merging.git | `ea1eca76b19c5d57ed97b1ef396368189e864eee` | MIT | source_pinned | True |
| `official_model_soups` | https://github.com/mlfoundations/model-soups.git | `d5398f181ea51c5cd9d95ebacc6ea7132bb108ec` | MIT | source_pinned | False |
| `official_task_arithmetic` | https://github.com/mlfoundations/task_vectors.git | `826a64c67082fab0f40628233287948f0f8d7fa3` | NO_REPOSITORY_LICENSE_FILE | source_pinned | False |
| `official_ties` | https://github.com/prateeky2806/ties-merging.git | `44e7891fc84f3de7e4caa52664cd864ca3715e91` | BSD-3-Clause | source_pinned | False |
| `official_dare` | https://github.com/yule-BUAA/MergeLM.git | `6d49ad96fd69c92013654b837041b868aa806564` | NO_REPOSITORY_LICENSE_FILE | source_pinned | False |

## Installation and adapter boundary

The official trees are external to the tracked repository. Recreate the source ledger with `git clone <repository-url> <official-root>/<directory>` followed by `git -C <official-root>/<directory> checkout <commit-from-the-table>`. The isolated Git Re-Basin worker environment was created with `python3.12 -m venv <jax-env>` and `<jax-env>/bin/python -m pip install jax==0.4.38 scipy`. The main integration uses the repository's existing Python 3.12 environment. The exact smoke and confirmatory commands are recorded in the config and run CSV.

- Git Re-Basin: a Python 3.12 environment installed `jax==0.4.38` and SciPy; the worker converts PyTorch MLP tensors to the axes expected by the official `src/weight_matching.py`, executes that file, converts back, and evaluates in TwistedMerge. The optimizer source is unmodified; only an import-only `rngmix` shim avoids pulling the unrelated Flax/W&B application stack.
- C2M3: the adapter bypasses the Hydra/Lightning application initializer, supplies an exact one-hidden-layer `PermutationSpec`, and executes the official Frank-Wolfe synchronized matcher. The tracked patch `external_baselines/patches/c2m3_cpu_device.patch` replaces one hard-coded `.cuda()` with the current permutation tensor's device.
- TIES: the adapter flattens the saved common-base task deltas, executes the official BSD-licensed `merge_utils.merge_methods` trim/elect/disjoint-mean kernel, restores the state dictionary, and chooses density/scale using validation data only. At the keep-all boundary, the adapter maps density 1.0 to the immediately preceding floating-point value because the official `topk_values_mask` requests invalid `k=0` at exactly 1.0; this preserves the intended keep-all mask without changing the official source.
- Model Soups: the MIT source is pinned, but `main.py` is inseparable from the official CLIP/ImageNet loader/evaluator. Replacing those components would no longer be an official execution, so no official metric is emitted.
- Task Arithmetic and DARE: the pinned author repositories contain no LICENSE or COPYING file. They are recorded as legal-use blockers and no publishable official metric is emitted.

## Exact checkpoint families

Independent-initialization runs use the existing MNIST one-hidden-layer MLP groups at `/Users/tinggong/Documents/GitHub/TwistedMerge/reports/checkpoints/external_baselines`. Common-base TIES runs use the available MNIST `mlp2` base/task groups at `/Users/tinggong/Documents/GitHub/TwistedMerge/reports/checkpoints/same_base_task_vector`. Independent internal comparisons are read from the exact-setting report CSV; common-base TIES, greedy-soup, and weight-average controls are recomputed on the exact saved 7200--7202 checkpoints because the aggregate same-base CSV now represents a later seed range. Completed training is not rerun.

## Outcome

Evaluated rows: `43`. Blocked integration/status rows: `3`. Successful rows are labeled `adapter_assisted_official_core`, not unmodified official end-to-end runs. Failed or legally blocked methods have no metric row.

The integration supports comparison only on the exact checkpoint families and regimes in `reports/csv/post_iclr_official_baseline_runs.csv`. It does not support a broad official-baseline or SOTA claim.
