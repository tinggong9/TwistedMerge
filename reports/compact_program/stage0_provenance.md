# Compact benchmark provenance

This run starts from commit `d30b02821526c13428d96143d4af1456918b2452` on branch `twistedmerge-compact-benchmark-2026`. The existing test suite completed successfully: `389 passed, 5 subtests passed in 21.11s`.

The benchmark uses MNIST and Fashion-MNIST from their public sources. CIFAR-10 is optional until its canonical download or licensed mirror succeeds; a missing cache is never replaced with fabricated data. Downloaded data and model checkpoints remain in ignored local directories, while checksums and source records are public.

Candidate predictions are saved before test-label evaluation. Every accuracy stage performs a byte-identity regression after label permutation. Test labels are not used for model fitting, candidate construction, routing, or selection.

This repository contains scientific evidence and reproducibility artifacts only.
