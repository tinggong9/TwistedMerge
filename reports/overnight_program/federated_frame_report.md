# Stage 10: federated sensor-frame smoke

Four actual linear MNIST clients were trained in 0/90/180/270-degree pixel coordinate frames. Raw FedAvg and frame-synchronized merges were executed on held-out MNIST, as were greedy validation merging, branch pooling, the conservative Hodge/LR dispatcher, and parameter controls. Exact frame transitions have maximum cycle residual 0.000e+00; they are removable, so Hodge/LR correctly creates no lift. All saved-logit leakage checks pass.

Exact full-run blocker: this smoke has exact calibration, a complete four-client graph, one overlap size, one loop family, and no unseen-client training. No noisy/missing-overlap/connectivity grid or central/noncentral frame family is completed. Run `python experiments/federated_sensor_frame_merge.py --mode full` only after those data roles and client splits are implemented.
