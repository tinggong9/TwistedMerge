# Remaining blockers

1. Implement and test the real language training/evaluation backend and
   matching-artifact resume, including model, selection and inference costs.
2. Obtain compatible pinned trained language adapters and local base weights.
   Official Llama-3-8B is gated and its terms were not accepted. Public adapter
   cards lack license/seed/head details; one cohort cannot supply five groups.
3. On a suitable available accelerator, measure a small pilot before assigning
   training settings, dtype tolerances or a confirmatory compute estimate. This
   Mac has 8 GB shared RAM; the recorded free-memory snapshot and unavailable
   cohort do not support the proposed full campaign. No cloud price is quoted
   because no provider or pilot throughput was selected or measured. The
   initial campaign cap remains 48 accelerator-hours, not an estimate.
4. Create and hash actual disjoint split/example manifests; implement near-
   duplicate checks, fixed prompts/scoring and justified confidence assumptions.
5. Integrate pinned KnOTS and Core Space with parity on the same checkpoints;
   freeze all primary baselines, gates, finite candidate families and endpoints.
6. Restore the disconnected historical data drive only if checkpoint-level
   legacy reruns are required. Historical CSV reanalysis already succeeds.

Current evidence supports coefficient-disciplined diagnostics and algebraic
verification, not a new practical-method superiority claim for KBS. Do not
start another broad holonomy search to replace these missing prerequisites.
