# Strict zero-shot D4 chart generalization

Execution commit: `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`. Image chart labels for [2, 3, 5, 6, 7] were absent from gradient training, early stopping, calibration, augmentation, threshold selection, and architecture selection.

Generator, inverse, multiplication-law, and seen-chart equivariance supervision variants were fixed before final-test evaluation. Symbolic group relations did not expose held-out image chart labels.

The preregistered gate and each constituent condition are recorded in `claims.csv`; no failed condition is replaced by a weaker post-hoc criterion.
