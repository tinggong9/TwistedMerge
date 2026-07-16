# Spatial-output program factual report

## Stage status

- `B0_dataset_audit`: completed; Primary selection: Kvasir-SEG; local paired subset counts {'train': 160, 'validation': 40, 'test': 40}.
- `B1_segmentation_discovery`: completed; 5 seeds and 20 methods executed
- `B2_zeroshot_segmentation`: completed; 5 diagnostic seeds; gate=False
- `B3_chart_uncertainty`: completed; 5 seeds and 7 perturbations executed
- `B4_complete_cost`: completed; 36 complete-path timing rows
- `C1_multidomain_experts`: completed; 5 seeds; synthetic-domain gate=False
- `C2_missing_expert_robustness`: completed; 245 scenario-method rows
- `D1_transition_geometry`: completed; stable residual certificate=False
- `D2_residual_correction`: gate_closed; D1 stable-residual certificate was false
- `E1_second_biomedical_dataset`: gate_closed; B1 retransport and TwistedMerge-specific gates were false
- `E2_biomedical_landmarks`: blocked; selected biomedical dataset has no independent landmark, keypoint, or center annotations
- `F1_medical_3d`: gate_closed; positive 2D retransport gate was not established
- `F2_microscopy_multiview`: blocked; no public multiview microscopy archive with view metadata and segmentation annotations was resolved and audited
- `S1_exact_mask_retransport`: completed; exact output action and asymmetric negative controls executed
- `S2_exact_spatial_output_actions`: completed; 123/123 checks passed
- `S3_trivial_vs_nontrivial_output`: completed; trivial labels and four spatial representations compared
- `Z0_finalization`: completed; claim ladder and manifests written; test exit=0

## Protocol coverage and data

- Dataset-ready check: True; bounded split counts: {'train': 160, 'validation': 40, 'test': 40}.
- Dataset manifest rows: 2; dataset SHA-256 aggregate: b96dfb8b16afc51a1ceb5285b1170535064b6007a3e8c555bbd5d93eabe3e11d.
- Canonical Kvasir-SEG terms recorded by B0: research and educational use, citation required, and permission required for commercial use.
- Kvasir-SEG has no patient, center, site, scanner, institution, tissue, or organ-domain metadata in the resolved archive; synthetic color/stain shifts are labeled synthetic domains.
- Execution commits recorded in `commands.csv`: `241dd25523c46818b71804d1ed8bac9b296f1757`, `350085b249e4cdccaf603bcf4f9001b12d8e3c01`, `69dc6182d9fc1675a198380225f71c97b1deabdf`, `949518d90d23eb3a471a88117eb75c552fa849d1`.
- Finalizer execution commit: `02b8818e4da9ba700336b24b22fda8ea101d11ef`.
- Candidate segmentation predictions were persisted before mask metrics and label-permutation hash audits were recorded by B1, B2, B3, and C1.

## Numerical results

- Mean B1 Dice, `inferred_chart_canonicalize_pool_retransport`: 0.067316.
- Mean B1 Dice, `inferred_canonical_no_output_retransport`: 0.067248.
- Mean B1 Dice, `direct_d4_equivariant_unet`: 0.068133.
- Mean B1 Dice, `d4_test_time_augmentation`: 0.133581.
- Mean B1 Dice, `generic_soft_moe`: 0.146581.
- Mean B1 Dice, `one_canonical_inferred_inverse_and_retransport`: 0.133581.
- Mean B1 Dice, `supplied_chart_canonicalize_pool_retransport`: 0.067122.

## Paired confidence intervals

- `retransport_vs_no_output` Dice delta 0.000067, 95% CI [-0.000430, 0.000631], seeds=5.
- `retransport_vs_generic_soft` Dice delta -0.079266, 95% CI [-0.173117, 0.000000], seeds=5.
- `retransport_vs_generic_hard` Dice delta -0.079038, 95% CI [-0.125729, -0.030344], seeds=5.
- `four_vs_one_after_inferred_chart` Dice delta -0.066408, 95% CI [-0.200372, 0.001148], seeds=5.
- `full_vs_direct_equivariant` Dice delta -0.000818, 95% CI [-0.200372, 0.198759], seeds=5.
- `full_vs_tta` Dice delta -0.066266, 95% CI [-0.200372, 0.001575], seeds=5.
- `random_control_gap` Dice delta -0.066266, 95% CI [-0.200372, 0.001575], seeds=5.
- `wrong_output_control_gap` Dice delta 0.000003, 95% CI [-0.000000, 0.000010], seeds=5.
- `wrong_order_control_gap` Dice delta -0.066266, 95% CI [-0.200372, 0.001575], seeds=5.

## Domain and center results

- Real center/site results were not computed because the resolved archive contains no center/site metadata; C1 domains are synthetic color/stain domains.
- C1 `structured_vs_generic_moe` Dice delta 0.000000, 95% CI [0.000000, 0.000000], seeds=5.
- C1 `structured_vs_domain_router` Dice delta -0.000701, 95% CI [-0.091656, 0.080430], seeds=5.
- C1 `structured_vs_direct_equivariant` Dice delta 0.188535, 95% CI [0.062199, 0.314871], seeds=5.
- C1 `structured_vs_tta` Dice delta 0.126904, 95% CI [0.000000, 0.253809], seeds=5.
- C1 `structured_vs_one_canonical` Dice delta 0.126904, 95% CI [0.000000, 0.253809], seeds=5.

## Exact actions and component attribution

- Claim levels: S1=True, S2=True, S3=False, S4=False, S5=False, S6=False.
- Exact mask, landmark, heatmap, point-set, and vector-field actions are recorded under `sanity/`.
- B1 comparisons separately attribute output retransport, chart inference, multi-expert pooling, direct D4 equivariance, D4 test-time augmentation, generic routing, and supplied-chart inference.
- C1 uses exact D4 chart actions and separately labeled synthetic non-group domains.

## Complete cost and residual results

- Complete-path cost rows: 36.
- B4 includes chart inference, canonicalization, expert evaluation, pooling, sigmoid thresholding, output retransport, warm-ups, timed repetitions, process memory, accelerator memory where available, stored bytes, and checkpoint-derived training time.
- B4 batch-1 `inferred_full_retransport` median complete latency 95.561479 ms, stored bytes 123796, training time 9.581145 s.
- B4 batch-1 `direct_d4_equivariant_unet` median complete latency 57.789688 ms, stored bytes 30148, training time 0.341934 s.
- B4 batch-1 `d4_test_time_augmentation` median complete latency 53.103979 ms, stored bytes 30148, training time 0.815665 s.
- B4 batch-1 `generic_moe` median complete latency 16.120541 ms, stored bytes 123796, training time 8.242951 s.
- B4 batch-1 component medians for the measured setup: chart 2.985646 ms, four-expert evaluation 12.772229 ms, input transform 0.105001 ms, output retransport 0.072791 ms, threshold 0.030000 ms.
- Inferred full retransport on any measured frontier: False.
- D1 bottleneck cycle residual 1.198202, closure 1.312241, centrality 1.404711, distance to coboundaries 1.212052, stable rank=None, exceeds every matched null=None.
- D1 `activation_bootstrap` matched-null maximum cycle residual: 1.455846 across 200 draws.
- D1 `edge_shuffle` matched-null maximum cycle residual: 1.421460 across 200 draws.
- D1 `graph_topology_shuffle` matched-null maximum cycle residual: 1.547488 across 200 draws.
- D1 `matched_fit_random_gauge` matched-null maximum cycle residual: 1.259753 across 200 draws.
- D1 `matched_norm_coboundary` matched-null maximum cycle residual: 0.000000 across 200 draws.
- D1 stable residual certificate: False; D2 correction remained inactive when the D1 gate was closed.

## Negative and gated findings

- `S3 inferred_spatial_retransport` did not pass; evidence: biomedical/discovery/claims.csv retransport_gate.
- `S4 twistedmerge_specific_spatial_benefit` did not pass; evidence: B1 paired accuracy gates and B4 matched-cost gate.
- `S5 multi_domain_benefit` did not pass; evidence: multidomain/claims.csv; domains labeled synthetic.
- `S6 realistic_residual_correction` did not pass; evidence: transitions/residuals.csv and correction_claims.csv.
- No real multi-center conclusion was made because center/site metadata is absent.
- No second-dataset, real-landmark, 3D, or multiview-microscopy result was substituted when its required audited data were unavailable.
- Test command exit code: 0.

## Artifact paths

- Machine-readable claims: `claim_ladder.json` and `claim_ladder.md`.
- Experiment inventory: `experiment_manifest.csv` and `experiment_manifest.json`.
- Integrity inventory: `artifact_checksums.csv` and `checkpoint_manifest.csv`.
- Tests: `test_results.txt`; commands: `commands.csv`; failures: `failures.csv`.
