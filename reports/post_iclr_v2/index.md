# Post-ICLR v2 Experimental Evidence

This namespace contains post-ICLR v2 audits and new gated experiments. It does not contain manuscript edits.

## Status

- Current evidence audit: complete.
- Selector attribution: complete; 10 independent groups and 40 exact settings; all preregistered positive gates failed.
- Selector-attribution verdict: enriched-pool selection; no TwistedMerge-specific algorithmic gain established.
- BatchNorm-aware gauge: pending; scientifically independent next phase, not started in this branch.
- ResNet-18 CIFAR-10: pending the BatchNorm derivation and base-quality preregistration.
- Later planted, prediction, selector, and biomedical phases: gated.

## Files

- `current_evidence_audit.md`
- `current_claim_matrix.csv`
- `current_artifact_manifest.csv`
- `journal_evidence_matrix.csv`
- `selector_attribution/`
- `proposed_claim_update.md`
- `paper_editor_evidence_brief.md`

The selector-attribution phase used new seeds `9300`--`9309`, model counts 3 and 4, and widths 32 and 64. A5 trailed the exactly candidate-count- and selector-evaluation-matched ordinary control B0 by `0.001865` accuracy (paired group-bootstrap 95% CI `[-0.002578, -0.001215]`). See `selector_attribution/report.md`.
