# Post-ICLR v2 Experimental Evidence

This namespace contains post-ICLR v2 audits and new gated experiments. It does not contain manuscript edits.

## Status

- Current evidence audit: complete.
- Selector attribution: complete; 10 independent groups and 40 exact settings; all preregistered positive gates failed.
- Selector-attribution verdict: enriched-pool selection; no TwistedMerge-specific algorithmic gain established.
- BatchNorm-aware gauge: complete; compatible ResNet-18 BasicBlock permutations and frozen-evaluation affine compensations are supported within the preregistered float32 tolerances.
- BatchNorm boundary: running-statistics-only scaling is nonexact, and arbitrary channelwise positive scaling is not a train-mode exact gauge.
- ResNet-18 CIFAR-10: real-data pipeline and resumable smoke complete; BatchNorm exactness gate opened; the validation-only base-quality pilot remains pending about 18.13 measured local compute hours.
- Later planted, prediction, selector, and biomedical phases: gated.

## Files

- `current_evidence_audit.md`
- `current_claim_matrix.csv`
- `current_artifact_manifest.csv`
- `journal_evidence_matrix.csv`
- `selector_attribution/`
- `batchnorm_gauge/`
- `resnet18_cifar10/`
- `proposed_claim_update.md`
- `paper_editor_evidence_brief.md`

The selector-attribution phase used new seeds `9300`--`9309`, model counts 3 and 4, and widths 32 and 64. A5 trailed the exactly candidate-count- and selector-evaluation-matched ordinary control B0 by `0.001865` accuracy (paired group-bootstrap 95% CI `[-0.002578, -0.001215]`). See `selector_attribution/report.md`.

The BatchNorm phase tested five independent random ResNet-18 parameter states at epsilons `1e-5`, `1e-3`, and `1e-1`. Compatible permutations had maximum logit error `2.563e-6` in eval mode and `9.548e-5` in train mode, with zero prediction disagreement. See `batchnorm_gauge/report.md` and `batchnorm_gauge/derivation.md`.
