# End-to-end controlled systems audit

Execution commit: `1237f70e2276c249ee5cbc1eedb977fc1043db26`. Seven actual PyTorch implementations were trained and evaluated across 40 independent S3/D4 settings. Cold start and 100 synchronized warm repetitions were measured at batch sizes 1, 8, 32, and 128. FLOPs are marked unavailable because no model-aware counter is installed. The matched accuracy-and-cost gate passed.
