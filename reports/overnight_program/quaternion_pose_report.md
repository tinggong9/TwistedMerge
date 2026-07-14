# Stage 6: quaternion/projective pose smoke

This is the permitted generated-3D/quaternion fallback, not real-dataset evidence. Quaternion lifts exhibit negative cycle signs in every seed while the underlying SO(3) rotations remain sign invariant. The two-sheeted lift paired accuracy delta over the best strict synchronization baseline has 95% CI [+0.000000, +0.000000], so the preregistered superiority gate is **not passed**.

Exact blocker: no ModelNet, ShapeNet, SYMSOL, licensed pose dataset, or object-mesh corpus is installed in the repository/environment. A full run requires attaching one and implementing its train/validation/test split; command: `python experiments/quaternion_projective_pose_merge.py --mode full` after data integration. Saved prediction tensors are label/target independent and all target-permutation hashes pass.
