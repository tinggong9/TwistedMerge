# Stage 12: capacity, latency, and robustness audit

The audit normalizes 76 benchmark-method summaries and reports the preregistered practical score for all 3×3 combinations of regret penalty {0, 0.1, 0.25} and inference penalty {0, 0.01, 0.05}. It does not select a lambda after seeing results. Wrong-action, wrong-generator, wrong-order/context, random-branch, and quaternion controls are retained in `robustness_controls.csv`.

Observed false-positive activation rate across four negative/certificate gates is 0.0000. False-negative rate is not identifiable without verified positive natural examples. Missing FLOPs, peak memory outside Stage 1, batch-size sensitivity, alignment-noise sweeps, and branch-count scaling are left missing rather than estimated; the available measurements do not support a full systems conclusion. Cross-benchmark native scores are not pooled into one headline accuracy.
