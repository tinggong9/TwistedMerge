# Complete Fashion-MNIST end-to-end cost audit

Execution commit: `11c8fb838cfa616d934dfc724777b3cd3fa6c05e`. Ten ABLATION checkpoints were timed at batch sizes [1, 8, 32, 128] with 10 warm-ups and 100 synchronized repetitions per complete method path.

Complete paths include transformations, chart inference, expert evaluation, routing, feature pooling or retransport where applicable, abstention, and final prediction. Component timings are separately recorded in `components.csv`.

Speed, storage, and Pareto claims are reported independently in `claims.csv`; no component-only timing is used as a complete-path claim.
