# BatchNorm gauge verification

## Automated checks

- `python -m pytest -q tests/test_batchnorm_channel_gauge.py tests/test_post_iclr_selector_attribution.py tests/test_post_iclr_v2_current_evidence_audit.py`: **14 passed**.
- Smoke stage: 16 exactness rows, zero execution failures.
- Pilot stage: 64 exactness rows, zero execution failures; the five-seed, three-epsilon confirmatory protocol was frozen afterward.
- Confirmatory stage: 240 exactness rows across 15 seed--epsilon settings, zero execution failures.
- Every purported-exact strategy passed its preregistered logit tolerance with zero prediction disagreement.
- `git diff --check`: clean.
- Both generated plots were inspected at full resolution; labels, scales, thresholds, and legends are readable.
- Every entry in `artifact_manifest.csv` was SHA-256 checked after final generation.

The full repository suite was not repeated for this phase. An earlier attempt on this branch reached an unrelated existing `test_block_gauge_branch_closure.py` subprocess that stalled in `block_gauge_phase_diagram.py`; the three status files touched by that attempt were restored byte-for-byte. The focused audit, selector, and BatchNorm suites pass together.

## Numerical boundary

- Permutation maximum logit error: `2.563e-6` in eval mode and `9.548e-5` in train mode; zero prediction disagreement.
- Canonicalized per-location activation maximum: `1.645e-5` in eval mode and `9.007e-4` in train mode. The deeper train activation value records float32 reduction-order accumulation; the preregistered decision is logit-level.
- Running-statistics-only scaling reaches `0.2030` maximum eval logit error at epsilon `0.1`, so it is explicitly negative rather than rounded into an exact claim.
- All configurations preserve architecture, parameter count, branch count, and 1x inference. Transform time, identity-evaluation time, calibration batches, stored bytes, and process peak RSS are in `resource_accounting.csv`.

## Scope

The implementation and claim cover torchvision-style two-Conv ResNet BasicBlocks, BatchNorm affine parameters and buffers, projected shortcuts, identity-shortcut basis constraints, and the final classifier input. Bottleneck blocks and grouped or depthwise convolutions remain outside scope.
