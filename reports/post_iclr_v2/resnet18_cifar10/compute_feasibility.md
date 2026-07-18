# ResNet-18 CIFAR-10 compute feasibility

## Outcome

The real-data pipeline is operational, checkpoint-resumable, and test-isolated. The requested base-quality pilot has not been interpreted or frozen because it has not run: an actual 45,000-example MPS training epoch took `135.65` seconds and the 5,000-example validation pass took `9.37` seconds.

At that measured rate:

- one 150-epoch model requires approximately `6.04` hours;
- the preregistered three-model pilot requires approximately `18.13` serial hours;
- the five four-model confirmatory groups require approximately `120.85` serial hours before merge evaluation.

These estimates use the measured training plus validation time and exclude checkpoint serialization, merge computation, and final metric generation. Every epoch is durably checkpointed, so an intentionally launched run can resume after interruption.

## Throughput probes

- MPS, batch 128, 45,000/5,000 examples: `135.65` seconds training plus `9.37` seconds validation.
- Increasing the MPS batch from 128 to 512 did not improve throughput on the fixed 2,048-example probe.
- CPU at batch 512 was about four times slower than the corresponding MPS subset probe.
- MPS float16 autocast was slightly slower than float32 in a fixed five-step kernel probe (`0.3800` versus `0.3536` seconds per step).
- Channels-last forward succeeded, but backward failed reproducibly under PyTorch `2.12.1` MPS with a noncontiguous-view runtime error. The committed recipe therefore uses stable contiguous tensors.

## Scientific boundary

The one-epoch smoke accuracy is not evidence about model quality or merging. The >=`0.92` mean validation-accuracy gate remains pending, the recipe is not frozen, the CIFAR-10 test partition has not been loaded, and no ResNet-18 merge claim is allowed yet.

The required pilot can be launched with:

```bash
/Users/tinggong/Documents/GitHub/TwistedMerge/.venv/bin/python experiments/post_iclr_resnet18_cifar10.py --stage pilot
```

The script uses seeds `25100,25101,25102`, writes a durable checkpoint after every epoch, and freezes the later group seeds only if all preregistered validation gates pass.
