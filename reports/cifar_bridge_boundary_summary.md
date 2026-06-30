# CIFAR and Bridge Boundary Summary

This appendix summary consolidates the CIFAR and bridge-dataset evidence into a bounded, claim-clean story. It is based on:

- `reports/cifar_or_colored_mnist_feasibility.md`
- `reports/cifar_rescue_or_no_go_report.md`
- `reports/cifar_final_channel_gauge_confirmatory_report.md`
- `reports/bridge_dataset_channel_gauge_expansion.md`
- `reports/claims_audit.md`

## Boundary Claim

Allowed: CIFAR is included as a bounded appendix boundary, not as a broad model-merging win.

Not supported: CIFAR confirms the main method.

## Chronology

The first CIFAR gate failed. The initial CIFAR-10 probe reached test accuracy `0.2480`, below the `0.45` plumbing threshold and far below the `0.60` meaningful-claim threshold. No CIFAR merge-performance claim was allowed from that probe.

The bounded rescue changed only the gate status. With a larger no-BatchNorm two-convolution ReLU CNN, normalized inputs, longer training, and basic random crop/flip augmentation, the rescue cleared the base-accuracy gate: the merge setting reported max individual accuracy `0.658333`. This permitted merge diagnostics, but the report had only one merge setting, so method wins remained descriptive.

The final CIFAR confirmatory run kept CIFAR in the appendix rather than the main claim path. It used five seeds for the rescued no-BatchNorm CIFAR architecture and reported mean individual max accuracy `0.650620` with minimum setting max `0.646400`, so the base gate passed. However, exact channel-gauge methods were only descriptive: optimized channel scaling was `+0.000540` versus C2M3 with CI `[-0.000880,0.002740]`, and shrinkage/global scaling were similarly tiny with intervals crossing zero.

Greedy soup remains the CIFAR boundary. In the final run, greedy soup reached mean test accuracy `0.648000`, while C2M3-style channel synchronization reached `0.464420`; exact single-model channel gauges stayed near C2M3 and far below greedy soup. The union candidate soup was only `+0.000440` over greedy soup with CI touching zero, and the greedy-safe selector tied greedy soup exactly. The ensemble upper bound was higher at `0.672760`, but it is extra-capacity and not a capacity-matched method.

Bridge datasets support the C2M3-versus-greedy boundary pattern, but not CIFAR or broad vision generality. Rotated-MNIST and colored-MNIST clear their bridge accuracy gates and show greedy soup above C2M3-style synchronization. The expanded bridge run reports overall greedy soup delta versus C2M3 `+0.084724`, CI `[0.063629,0.103986]`; the main rotated-25 `N=3` 10-seed setting reports `+0.089670`, CI `[0.067857,0.112412]`. These are MNIST-derived bridge datasets, so they cannot be promoted to CIFAR or general vision claims.

## Compact Table

| Dataset / stage | Architecture | Base accuracy | Strongest C2M3 delta | Strongest greedy-soup delta | Claim status |
| --- | --- | --- | --- | --- | --- |
| CIFAR-10 failed probe | early small CNN probe | `0.248000` | N/A; no merge rows | N/A; no merge rows | Below plumbing; probe only |
| CIFAR-10 bounded rescue | no-BatchNorm two-conv ReLU CNN | `0.658333` max individual | `+0.171333` for greedy soup / greedy-safe selector vs C2M3, one setting | `0.000000` for greedy-safe selector vs greedy soup; ensemble `+0.016667` is extra-capacity | Gate passed; descriptive diagnostics only |
| CIFAR-10 final confirmatory | no-BatchNorm two-conv ReLU CNN, five seeds | `0.650620` mean max, min `0.646400` | `+0.000540` optimized channel scale vs C2M3, CI crosses zero | `+0.000440` union candidate soup vs greedy soup, CI touches zero; greedy-safe selector ties | Bounded appendix boundary, not broad win |
| Rotated-MNIST bridge feasibility | no-BatchNorm small ReLU CNN | `0.924583` mean max | `+0.087917` greedy soup / selector vs C2M3, CI `[0.062500,0.113333]` | `0.000000` selector vs greedy soup | Bridge-only evidence |
| Rotated/colored-MNIST bridge expansion | no-BatchNorm small ReLU CNN | `0.928694` mean max overall | `+0.084724` greedy soup vs C2M3 overall; optimized exact scale `+0.014729` vs C2M3 | `0.000000` selector vs greedy soup | Supports bridge boundary pattern only |

## Why Exact Channel Gauges Are Descriptive On CIFAR

The architecture has no BatchNorm, so channel permutations and positive channel scales are exact ReLU reparameterizations. Exactness does not imply an accuracy gain after averaging independently trained checkpoints. In the final CIFAR run, exact scale variants changed accuracy by only about `0.00004` to `0.00054` versus C2M3, with bootstrap intervals crossing zero, while all exact single-model scale rows remained about `0.183` below greedy soup.

## Why Greedy Soup Is The Boundary

Greedy soup is the strongest capacity-matched single-model baseline in the CIFAR appendix rows. The final selector used validation metrics only and selected/tied greedy soup; the union candidate soup produced only a tiny descriptive gain with a confidence interval touching zero. Therefore CIFAR belongs in the appendix as a boundary result rather than as support for a main method win.

## Bridge-To-CIFAR Boundary

Bridge datasets are useful because they show the same C2M3-versus-greedy boundary pattern under simple MNIST-derived shifts. They are not CIFAR. They do not establish robustness to natural-image statistics, broader architectures, BatchNorm, external official baselines, or general vision model merging.
