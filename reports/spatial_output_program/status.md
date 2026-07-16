# Spatial-output program status

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
