# Fashion-MNIST component ablation

Execution commit: `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`. The stage executed 10 fresh seeds with disjoint local-model, chart-training, validation, calibration, threshold-selection, and final-test roles.

The seven same-chart-probability methods used byte-identical calibrated D4 chart probabilities. Candidate task logits were saved before final-test labels were evaluated, and the label-permutation audit left every saved candidate hash unchanged.

Attribution gates are recorded without reinterpretation in `claims.csv`; negative intervals remain negative findings.
