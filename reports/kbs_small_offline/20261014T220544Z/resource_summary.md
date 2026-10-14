# Measured resource use

CPU only, two threads. Total process wall time including imports and execution
through report completion: 7.392719 seconds. Pilot through decision:
4.660 seconds. No network request, model download, dataset fetch, remote
inference, external experimental AI dependency or paid compute.

Absolute process peak RSS: 389,529,600 bytes
(371.484 MiB). Explicit startup **high-water**
baseline after imports/before loading digits: 378,716,160 bytes.
High-water growth: 10,813,440 bytes. This is growth
over the startup high-water, not a measurement of growth over instantaneous
startup resident memory. The absolute process peak itself is below 1 GiB,
so the additional-memory cap holds independently of that distinction.

A 5 ms monitor plus checks each training batch/stage tracked memory/time;
resource_checkpoints.jsonl records stage boundaries. Per-policy RSS is sampled,
while resource.getrusage supplies the process lifetime high-water. No native
operation is preempted. Conservative host admission required 128 MiB estimated
additional working memory plus 256 MiB safety reserve; available memory at
startup was 1,251,557,376 bytes.

Before final receipts/export, experiment artifacts occupied 648,131
bytes, including 430,302 bytes of six locally retained model NPZs. Final
artifact byte count is recorded in verification_receipt.json. Weights remain
under ignored weights/ and are hashed in model/group manifests, not committed.
New code, tests and reports are comfortably below the 150 MB limit.

Cold policy calls load fresh model/factor state from NPZ; the process, imports
and OS page cache may be warm. Warm calls reuse that policy's loaded base and
factors, then reconstruct candidates. These are actual standalone sequential
policy runs, not isolated fresh-process/OS-cold startup benchmarks. P4 and P0
build the same fallback and have similar measured times; tiny signed timing
differences do not establish an added overhead or reliable speed advantage.
