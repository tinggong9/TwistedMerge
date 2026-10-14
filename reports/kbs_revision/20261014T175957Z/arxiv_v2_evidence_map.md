# Evidence map for a possible arXiv v2

This is a claim audit, not a manuscript edit or submission. Existing artifacts
are read in place at source commit 358f1cf570d73b74464608a5c31e730a209931cf.
Their original code/evidence provenance is retained; their tests were not
rerun. The present readiness-only run adds no algebraic or neural experiment.

| Proposed statement | Classification | Evidence and precise boundary |
| --- | --- | --- |
| Stored-edge torus examples give exact nonzero μ2/F2 certificates | verified existing | [Prior algebra checks](../20261014T095143Z/algebraic_checks.json), [scope](../20261014T095143Z/claims.md); controlled realizable stored-edge examples, not natural neural classes; general μn is not implemented |
| A simply-connected complex admits no such obstruction under the stated constant-edge model | verified existing | Same checks/scope; constant coefficients, context-independent inverse-consistent G edges and central μ2 triangle residuals are essential; arbitrary oracle face classes need not be edge realizable |
| Vector Hodge admission and the fully matched vector policy use equivalent information | verified existing | [Gate/algebra controls](../20261014T095143Z/gate_ablation_controls.json); same real-vector complex, inner product and gates; approximate diagnostics are not exact finite coefficients |
| Compact weighted-sum QR/core-SVD agrees with the dense reference in tested cases | verified existing | [Checked parity](../20261014T095143Z/baseline_parity.json); tested float64/complex128 arithmetic, not a universal conditioning or memory theorem |
| Efficient implementation avoids ambient-shaped Torch outputs and records CPU timings | software-only | [CPU shapes/profile](../20261014T095143Z/cpu_shape_profile.json), [scope](../20261014T095143Z/claims.md); opaque LAPACK workspace excluded; saturated cores lose the favorable memory regime; failed ill-conditioned settings remain |
| Local Qwen/PEFT backend, native task schemas, safe reload and atomic resume execute | software-only | [179-test receipt](../20261014T102546Z/test_results.txt), [integration](../20261014T102546Z/software_integration.json), [summary](../20261014T102546Z/execution_summary.md); random two-block Qwen-shaped model, rank-2 generated specialists, four steps/one generated record each; no pretrained Qwen benchmark |
| Eleven candidates and P0–P4 policies execute; all policies choose compact fallback in the fixture | software-only | [Integration](../20261014T102546Z/software_integration.json), [policy scope](../20261014T102546Z/primary_policy_summary.md); invented records do not establish useful or useless population decisions; P3/P4 equivalence supports no new advantage |
| KnOTS/Core adaptations match reviewed tensor sources on primary cases | software-only | [Parity](../20261014T102546Z/external_method_parity.json), [fidelity](../20261014T102546Z/external_method_parity.md); 12 primary checks, retained conditioning/float32 failures, common/equal upstream weighting; fixed output compression/heterogeneous-weight corrections are adaptations; no official end-to-end study |
| Current policy cost tables show shared-panel cost attribution | software-only | [Policy scope](../20261014T102546Z/primary_policy_summary.md); actual generation/fitting stages, but no standalone sequential savings or measured inference multiplier |
| Cycle diagnostics failed to improve held-out image-MLP prediction in the retained study | historical negative | [Historical ledger](../20261014T095143Z/claim_ledger.json): cycle R² −0.2055342077 vs pairwise 0.3871983595 on 120 dependent checkpoint collections; not 120 training groups; dirty executed-source/checkpoint limitations retained |
| Matched MNIST selector A5 underperformed B0 slightly | historical negative | Same ledger: A5−B0 −0.001865, group-bootstrap 95% CI [−0.0025775625, −0.001215], 10 training groups/40 settings; validation allowances/pool counts matched, generation kernels/cost not fully matched |
| Feature-layer LoRA dense SVD exceeded global alignment; the cycle-aware rule always fell back | historical negative | Same ledger: dense-SVD−global +0.051725 in five groups combining three pilot and two extension groups; 300/300 fallback; single CIFAR feature layer, dependent scrambles, not a transformer confirmatory campaign |
| No natural certificate was found in the retained corpus | historical negative | Same ledger and [scope](../20261014T095143Z/claims.md); corpus-specific absence, not universal nonexistence |
| Pinned Qwen and four real tasks were evaluated in this run | unsupported | [New readiness](readiness.json), [prior block](../20261014T102546Z/pilot/readiness.json); no real pilot, compatible cohort or licensed real splits |
| Structural certification improves P4−P0/P4−P1 loss, calibration, harmful-merge risk or abstention benefit | unsupported | [Primary policy scope](../20261014T102546Z/primary_policy_summary.md), [prior decision](../20261014T102546Z/pilot_decision.md); no identified neural coefficient/confidence model or actual held-out neural utility/risk/coverage evidence |
| Standalone speedup, sequential early-exit savings or inference-cost advantage has been measured | unsupported | [Policy scope](../20261014T102546Z/primary_policy_summary.md); shared-panel attribution cannot establish this |
| General μn neural central cohomology, practical noninferiority, powered/significant confirmatory advantage | unsupported | [Prior claims](../20261014T102546Z/claims.md), [power scope](../20261014T102546Z/confidence_and_power.md); no justified finite/general coefficient model, task-specific margin, pilot-informed power or independent-group confirmatory run |

There are **no qualifying `verified newly algebraic`, `real-pilot`, or
`confirmed neural` entries added by this run**. Existing mathematical results
were already verified in the earlier scoped algebra run; they must not be
relabeled as newly established here. The only new finding is a measured,
source-pinned resource/readiness BLOCKED status. A narrower theory/diagnostic
framing may use the scoped algebra and retained negatives, subject to actual
novelty/review; it cannot describe TwistedMerge as an improved neural merger on
the strength of fixtures. No manuscript, theorem, bibliography or arXiv file
was changed, uploaded or pushed.
