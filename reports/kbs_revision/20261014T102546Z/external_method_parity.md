# External comparator fidelity

12 primary small-fixture checks passed. Exact reviewed
upstream tensor excerpts are executed with a hash-locked manifest, independently
of the adaptations. Full upstream repositories, model loaders and pickle
weights are not executed. Source/license/hash details are recorded in
external_method_parity.json and configs/kbs_revision/external_parity_sources.json.

KnOTS applies pinned dense effective-update SVD with upstream cutoff/promotion,
then globally flattened multilayer TIES in aligned coordinates. Core Space
uses separate factor bases and nonlinear core-coordinate merging. It is not
the compact weighted-sum SVD operator. Heterogeneous weights require the named
corrected_per_task adaptation; official parity covers common/equal weights.
Fixed output-rank compression is an added adaptation and is labelled explicitly.

Conditioning stress records, including failures, remain in the parity JSON.
KnOTS inherits upstream float32 precision and cannot meet a universal strict
1e-10 claim even in ordinary conditions; primary parity tolerances account for
source precision. No official end-to-end neural benchmark result is claimed.
