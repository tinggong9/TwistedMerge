# Factor-space scalability complexity audit

The systems fixture uses the RMS scale of the 40 trained holonomy adapters but is not application-performance evidence. All comparisons use float32, identical dimensions, adapter counts, input rank, and rank-32-bounded method logic.

## Implemented accounting

- Dense deterministic and randomized methods explicitly allocate an `m x n` effective-update mean, recorded by the runtime sentinel, then use a rank-sized range-SVD workspace. Their leading memory is `O(m n)` and their dense product cost includes the effective-update materialization.
- Canonical factor space uses thin QR factorizations and `r x r` core SVDs for each adapter. It uses `O(k r (m+n) + k r^2)` storage and never requests an `m x n` buffer.
- Pairwise reference alignment whitens each B factor with an `r x r` Gram matrix, estimates orthogonal rank-space maps, and averages aligned factors. It uses `O(k r (m+n) + k^2 r^2)` storage.
- Global synchronization adds a complete rank-space transition graph and a rank-space least-squares synchronization. Its recorded implementation realizes `O(k r (m+n) + k^2 r^2)` storage; no dense update is constructed.
- Cycle-aware alignment uses the same low-rank transition graph and gauge-invariant orthogonal cycle diagnostics. In this benchmark it is forbidden to use the Phase-A dense SVD safety fallback; it must either complete in factor space or abstain.

The `temporary_allocation_bytes_analytical` field is method accounting, while `peak_rss_bytes` and `incremental_peak_rss_bytes` are process-isolated measurements. Python/runtime baseline RSS is reported separately. Stored result size is the two rank factors and is the same rank-bounded form for every successful method.
