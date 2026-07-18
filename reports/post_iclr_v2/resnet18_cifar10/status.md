# ResNet-18 CIFAR-10 phase status

- Training and validation pipeline: complete and smoke-tested on real CIFAR-10.
- Dataset archive: SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
- Architecture: 11,173,962-parameter torchvision ResNet-18 with CIFAR stem.
- Leakage check: zero test evaluations; train/validation index overlap is zero.
- Resume behavior: best and last checkpoints written after every epoch under the ignored `checkpoints/` namespace.
- Base-quality pilot: pending measured local compute of about 18.13 hours.
- Frozen recipe: absent until the pilot passes.
- Confirmatory merge study: gated and not started.

This status is an implementation/compute result only, not a positive or negative model-merging result.
