# Shared D4 Adapter Corpus Report

Decision: **complete smoke corpus construction**.

## Exact command

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/holonomy_shared_corpus.py --mode smoke --data-dir /Users/tinggong/Documents/GitHub/TwistedMerge/data --device auto
```

- Evidence label: `natural_measured`
- Frozen encoder: torchvision ResNet-18 `IMAGENET1K_V1`
- Dataset: CIFAR-10 from the existing local cache
- Charts: eight D4 transforms using the repository's audited action convention
- Adapter: rank-4 residual map in a 64-dimensional train-only PCA feature space plus a ten-class head
- Independent model-training seeds: [0]
- Successful chart adapters: 8 / 8
- Mean validation accuracy: 0.177734
- Worst chart/seed validation accuracy: 0.078125
- Corpus quality gate: passed
- Test logits: saved and hashed before any test labels are accessed
- Test labels used during corpus construction: no
- Validation label-permutation guard: passed
- Failures: 0

## Boundary

This corpus is the only trained model family authorized for Applications A-D. Later phases must load the exact feature cache, checkpoints, splits, and logits identified in the manifests; they must not retrain chart adapters. Validation accuracy is a corpus-quality diagnostic, not an application result. No test accuracy or holonomy claim is made here.

The encoder and all its parameters remain frozen. PCA fitting uses only `adapter_train` features. Transition fitting is reserved for `overlap_fit`, transition validation for `overlap_validation`, method selection for `validation`, and final application scoring for the untouched `test` identities.
