# Baseline and systems-cost proxy audit

Execution commit: `00a00e704f8daf4f7ae158af6d03c755fbb1d1c6`. Executed accuracy rows and parameter/storage metadata were collected at batch sizes 1, 8, 32, and 128. The batch-size latency, memory, and FLOP rows are explicitly labeled shape-matched NumPy proxies, not end-to-end timings or official external baseline executions. Consequently, 0 of 1 families passed a matched systems-cost gate. Conditional negative stages were not promoted into this audit.
