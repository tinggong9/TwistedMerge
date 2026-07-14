# Stage 0 Provenance and Evidence Audit

## Isolation and baseline

- Worktree: `/Users/tinggong/Documents/Codex/2026-07-14/ru-3/work/TwistedMerge-overnight`
- Branch: `twistedmerge-practical-research-2026`
- Base/HEAD: `d71d1a3651a4c4c23a3e6e80c834b729d6a8aa2e`
- Original checkout preserved with 1,408 staged additions; no file from that index was modified.
- Baseline tests: `372 passed, 5 subtests passed in 20.30s`

## Provenance repair

The former release builder assigned one global evidence commit to every entry. That is incorrect because several artifacts were executed at different commits. The release contract now requires each artifact's own JSON execution record and rejects missing or malformed commits instead of silently falling back to a global value.

| artifact family | actual execution commit | source record |
| --- | --- | --- |
| two_loop_holonomy | `0a41f76d3c8a77acc3a47514c2639b81fbc5b280` | `reports/next_benchmarks/two_loop_holonomy_config.json` |
| controlled_central_and_period_index | `0a41f76d3c8a77acc3a47514c2639b81fbc5b280` | `reports/next_benchmarks/central_reproduction_manifest.json` |
| context_router | `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf` | `reports/next_benchmarks/context_router_config.json` |
| matched_selector | `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf` | `reports/next_benchmarks/matched_selector_config.json` |
| heldout_diagnostic | `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf` | `reports/next_benchmarks/diagnostic_prediction_config.json` |
| pretrained_smoke | `8c369a6f1a7f08b7443626ae1dece7d25fc06ddf` | `reports/next_benchmarks/pretrained_merge_config.json` |

## Invalid empirical evidence

`src/controlled_nonabelian_holonomy.py` still contains `target_accuracy_for_method` and `logits_with_target_accuracy`; its accuracy artifacts remain `INVALID_AS_EMPIRICAL_ACCURACY_EVIDENCE`. The files are retained for auditability. Standalone group definitions and structural residual calculations may be cited only as structural evidence.

## Safe boundary

- New accuracy reports must use saved executed logits and a post-hoc label-permutation regression.
- `ensemble_upper_bound` is treated as legacy unsafe terminology; new work uses `ensemble_reference`.
- No manuscript file was edited.
