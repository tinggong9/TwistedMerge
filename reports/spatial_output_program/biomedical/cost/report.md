# Complete biomedical segmentation cost audit

- Methods: 9; batch sizes: (1, 4, 8, 16); complete-path timing rows: 36.
- Warm-ups: 10; repetitions: 100 for batches 1/4 and 30 for batches 8/16.
- Every complete path included sigmoid thresholding; inferred paths included chart inference, canonicalization, expert evaluation, pooling, and output retransport.
- Component chart, expert, input-transform, output-retransport, and threshold latencies were timed separately at matched repetition counts.
- Inferred full retransport appears on any measured frontier: False.
