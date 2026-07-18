# ResNet-18 CIFAR-10 pipeline verification

- `python -m pytest -q tests/test_cifar_resnet_benchmark.py tests/test_batchnorm_channel_gauge.py tests/test_post_iclr_selector_attribution.py tests/test_post_iclr_v2_current_evidence_audit.py`: 21 passed; the focused ResNet suite has 7 tests.
- The real CIFAR archive passes torchvision's MD5 check (`c58f30108f718f92721af3b95e74349a`).
- Standard smoke: one MPS epoch on 2,048 training examples and 512 validation examples completed with zero failures and zero test evaluations.
- Full-epoch compute probe: 45,000 training and 5,000 validation examples completed with zero failures and zero test evaluations.
- Model stem, output shape, parameter count, split determinism, split disjointness, calibration metrics, gate logic, and recipe serialization are unit-tested.
- CPU, larger-batch MPS, float16, and channels-last feasibility probes are reported rather than silently discarded.

The base-quality gate has not been evaluated; the smoke is not cited as a training-quality result.
