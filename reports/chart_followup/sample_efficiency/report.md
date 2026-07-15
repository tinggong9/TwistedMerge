# Chart-label sample efficiency

Trained and evaluated 180 method-seed-budget combinations at chart-label budgets [32, 64, 128, 256, 512, 1000].

Each seed uses disjoint local-task, chart-training, early-stopping, calibration, and fixed held-out test roles. Candidate logits were persisted before test labels were used for evaluation.

Threshold crossings include 50% and 60% task accuracy and 90% of the seed-averaged supplied-chart oracle gain target (0.688900).
