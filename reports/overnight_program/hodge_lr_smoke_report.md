# Stage 2 smoke report

All 4 focused test files passed. Orthogonal transition held-out relative error was 4.008e-16; weighted Hodge reconstruction error was 2.776e-17, and maximum weighted orthogonality error was 1.665e-17. The extracted residual rank was q=1 with 1.000000 explained energy. The inference-feature router achieved 0.9900 held-out branch accuracy. Distillation KL fell from 0.297360 to 0.000000.

The deliberately inconclusive gain sample did not pass the conservative gate, so the dispatcher returned `ordinary_validated_family` and created no branches. This is the intended false-positive control. These are component smoke results, not a natural-data accuracy claim.
