# Readiness diagnosis

| Dependency | Current state | Smallest remaining requirement |
| --- | --- | --- |
| A. Base/cohort/licensing | blocked for real Qwen; generated software cohort works | compatible local pinned Qwen weights/tokenizer, four task adapters or an approved fixed training recipe, reviewed real dataset/adapter licenses |
| B. Backend/baselines | software integration addressed | target-hardware dtype and performance validation; official end-to-end launchers are not claimed |
| C. Splits/statistics | seven actual software JSONL roles pass recomputed leakage audit; real roles unavailable | local pinned benchmark records and final frozen statistical protocol; no noninferiority claim without justification |
| D. Resources | real pilot infeasible under current local availability and cap | suitable already-authorized GPU/memory and storage; user knows of no such host |

The missing backend is now implemented and exercised, rather than retained as
a placeholder. The software cohort is not a substitute Qwen research cohort.
No final benchmark labels were loaded/evaluated. No research GO or frozen
confirmatory manifest is emitted. Resource checks are cooperative between
operations; native kernels are not preempted, and budget receipts account for
each observed invocation rather than unobserved interrupted-process work.

No model-size/dataset substitution or protocol revision was made. Pilot
throughput, confirmatory time, GPU memory peaks and cloud prices remain unknown.
