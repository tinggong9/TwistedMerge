# External Baselines

This directory documents the license-clean external-baseline layer for the
MNIST MLP merge comparisons. No third-party code is vendored here. The runnable
benchmark in `experiments/external_baseline_comparison.py` uses in-repository,
faithful implementations where official code was not integrated directly.

## Integration Status

| Baseline | Paper name | Official repository | License | Integration mode | Deviations from official code | Output type | Uses validation data | Capacity matched to weight average / C2M3 | Extra inference cost | Fair for this paper's current claims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Git Re-Basin | Git Re-Basin: Merging Models modulo Permutation Symmetries | <https://github.com/samuela/git-re-basin> | MIT | Faithful in-repo reimplementation; no vendored code | Uses a single hidden-layer ReLU MLP and activation-correlation Hungarian matching to align every model to model 0, then averages. It does not run the official JAX experiment stack, weight-matching variants, ResNet paths, or loss-barrier analysis. | Single merged model | No | Yes | No | Fair only as a Git-ReBasin-style pairwise permutation baseline on this MNIST MLP benchmark. Not fair as a claim against the official Git Re-Basin implementation. |
| C2M3 | Cycle Consistent Model Merging | <https://github.com/crisostomi/cycle-consistent-model-merging> | MIT | Faithful internal C2M3-style reimplementation; no vendored code | Uses the repo's activation-based pairwise permutation estimates and synchronizes them to a globally cycle-consistent reference before averaging. It does not run the official Hydra/uv/wandb training and matching pipeline or official config zoo. | Single merged model | No | Yes | No | Fair as the internal C2M3-style reference used by the paper's current MNIST MLP claims. Not fair as evidence that TwistedMerge++ beats official C2M3. |
| Model Soups | Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time | <https://github.com/mlfoundations/model-soups> | MIT | Faithful in-repo greedy soup implementation; no vendored code | Uses the standard greedy soup rule on the same MNIST MLP checkpoints with validation accuracy/loss. The official repo's downloaded CLIP/ViT soup checkpoints, dataset suite, and plotting pipeline are not run. Uniform soup is represented by ordinary weight averaging on the same candidate set. | Soup, still a single averaged model | Yes for greedy soup; no for uniform soup / weight average | Yes | No | Fair as a faithful greedy/uniform soup baseline on this MNIST MLP benchmark. Not fair as evidence that TwistedMerge++ beats official Model Soups broadly. |
| Weight average | Ordinary uniform parameter averaging control | Not an external repository | Project code | Native internal control | Uniformly averages the same trained MNIST MLP checkpoints without alignment. In this benchmark it is also the uniform-soup analogue over the original candidate set. | Single merged model | No | Yes | No | Fair as the unaligned single-model control. |
| Improved validated ladder selector | TwistedMerge internal selector | Not an external repository | Project code | Native internal method | Selects among already-computed single-model or soup candidates by validation accuracy/loss only. It does not read test metrics for method selection. | Validation-selected single model or single-model soup | Yes | Yes | No | Fair for the limited claim that the selector outperforms the internal C2M3-style baseline on this MNIST MLP benchmark if supported by the generated paired intervals. |
| Monomial scaling | TwistedMerge exact ReLU positive-scale gauge | Not an external repository | Project code | Native internal method | Applies positive hidden-unit scalings after permutation synchronization; positive ReLU scaling is exact before averaging. | Single merged model | No for raw monomial scaling; yes for soup variants | Yes | No | Fair as an internal exact-symmetry ablation, not as an external baseline. |

## External Baseline Integration Status

Official source repositories and licenses were inspected for Git Re-Basin,
C2M3, and Model Soups. The current benchmark does not vendor or import their
code. Direct wrapping was not used because the official repositories target
their own model families, configuration systems, checkpoint formats, and
experiment launchers. For the small MNIST one-hidden-layer ReLU MLP comparison,
the cleaner and more reproducible layer is therefore:

- faithful Git-ReBasin-style pairwise permutation alignment to model 0;
- faithful C2M3-style cycle-consistent permutation synchronization;
- faithful Model Soups greedy soup over the same candidate checkpoints;
- ordinary weight averaging as the uniform-soup analogue.

This status should be cited whenever reporting the generated numbers. The
benchmark does not support claims that TwistedMerge++ beats official Git
Re-Basin, official C2M3, or official Model Soups.

## Optional Baselines

TIES-Merging, RegMean, and other model-merging baselines were not integrated in
this pass. They remain future work unless their official license and checkpoint
interface can be verified and connected without vendoring incompatible code or
changing the MNIST MLP comparison protocol.
