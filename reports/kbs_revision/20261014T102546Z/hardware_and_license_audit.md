# Hardware and license audit

Initial total RAM: 8589934592 bytes; available RAM snapshot:
1269399552 bytes. Accelerator inventory is recorded in
initial_environment.json; CUDA is unavailable and MPS shares system memory.
This execution used CPU only. The user reported no known GPU/checkpoint host
and requested conservation of resources. No paid compute, weights download,
license acceptance, remote upload, or push was performed.

The pinned [Qwen file listing](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323) displays model.safetensors as 3.09 GB;
its [model card](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/8faed761d45a263340a0528343f099c05c9a4323/README.md) records Apache-2.0,
1.54B parameters and 28 layers. A download exceeds both the previous 2 GB
ceiling and this run's stricter 200 MB allowance, before data or checkpoints.
The local free-memory snapshot also falls below weight storage alone. These
are admission blockers, not measured full-model OOM or training benchmarks.

KnOTS tensor source is MIT; Core Space is Apache-2.0, with exact license text
and source pins in configs/kbs_revision/external_parity_sources.json. The
reference harness runs reviewed, hash-locked tensor excerpts offline, without
external model loaders. Generated fixture weights are marked CC0/software-only.
Real dataset and adapter licensing remains unresolved because no cohort was
downloaded or supplied. No gated or proprietary tensors are in the review bundle.
