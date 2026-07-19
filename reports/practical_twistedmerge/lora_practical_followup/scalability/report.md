# Low-rank-native LoRA scalability benchmark

Decision: **positive factor-space scalability result**.

- Successful process-isolated method/shape cases: 336 / 336.
- Timed trials per successful case: 3, after one warmup.
- Dimensions: [768, 1024, 2048, 4096]; ranks: [4, 8, 16, 32]; adapter counts: [4, 8, 16]; precision: float32.
- TwistedMerge factor methods used zero dense effective-update allocations: `True`.
- Gauge-invariance probe gate: `True` at tolerance `2.0e-04`.
- Minimum analytical temporary-memory ratio, global synchronization versus deterministic dense SVD: `0.00773709`.
- First recorded half-memory crossover: dimension `768`, rank `4`, adapters `4`.
- Global synchronization used lower measured incremental peak RSS in every rank/count case beginning at dimension `4096`.
- Measured half-memory cases: `24` / `48`; minimum measured incremental-RSS ratio: `0.0312705`.
- At dimension `4096`, global synchronization was faster in `10` / `12` rank/count cases; its median runtime ratio versus deterministic dense SVD was `0.220642`. This is not a uniform runtime-superiority claim.
- Dense deterministic temporary memory range: `2.391` to `66.500` MiB.
- Global synchronization temporary memory range: `0.095` to `17.000` MiB.
- Failures/timeouts: 0.

Random matrices here test systems and numerical behavior only. They do not establish application accuracy. Runtime comparisons are reported without a superiority claim unless output-quality and resource rows support the exact case. Dense deterministic truncated SVD and randomized dense SVD remain valid gauge-invariant baselines; the supported distinction is low-rank-native execution and measured/analytical memory, not uniqueness.
