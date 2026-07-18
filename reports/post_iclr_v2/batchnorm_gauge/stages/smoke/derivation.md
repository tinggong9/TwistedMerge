# BatchNorm-aware channel-gauge derivation

## Convention

For a representation with old channel vector $h$, a permutation $p$ defines the new coordinates $h'_j=h_{p(j)}$. A convolution from input basis $p_{\mathrm{in}}$ to output basis $p_{\mathrm{out}}$ therefore transforms as

\[
W' = W[p_{\mathrm{out}},p_{\mathrm{in}}], \qquad b'=b[p_{\mathrm{out}}].
\]

The following BatchNorm weight, bias, running mean, and running variance are indexed by $p_{\mathrm{out}}$. The tracked batch count is a scalar and is unchanged. A classifier consumes the final basis by permuting its input columns. For a projected shortcut, its Conv input/output and BatchNorm output receive the same incident bases. An identity shortcut has no parameters that can change basis, so exact residual addition requires $p_{\mathrm{out}}=p_{\mathrm{in}}$. The implementation rejects inconsistent identity shortcuts.

## Positive scaling before BatchNorm

In frozen evaluation mode, one channel is

\[
y=\gamma\frac{z-\mu}{\sqrt{v+\epsilon}}+\beta.
\]

After $z'=sz$, $s>0$:

- Keeping the original statistics and affine parameters is not exact.
- With only transformed statistics $\mu'=s\mu$, $v'=s^2v$, the denominator is $\sqrt{s^2v+\epsilon}$, not $s\sqrt{v+\epsilon}$. It is approximate whenever $\epsilon>0$ and $s\ne1$.
- Keeping the original frozen statistics is eval-exact with
  $\gamma'=\gamma/s$ and
  $\beta'=\beta+\gamma\mu(1/s-1)/\sqrt{v+\epsilon}$.
- Transforming statistics is eval-exact when epsilon is compensated through
  $\gamma'=\gamma\sqrt{s^2v+\epsilon}/(s\sqrt{v+\epsilon})$, with $\beta'=\beta$.

These static affine corrections depend on frozen running statistics. They are not a train-mode exact channel gauge for arbitrary batches. In train mode the batch variance changes, and PyTorch BatchNorm has one scalar epsilon rather than a per-channel epsilon. A uniform scale can instead be made exact by scaling the scalar epsilon by $s^2$, but arbitrary channelwise scales cannot use that escape hatch in standard BatchNorm.

## Recalibration

Recalibration and complete statistic recomputation restore statistics that describe the scaled activations, but running-stat transformation alone still leaves the epsilon discrepancy. They are therefore measured and labeled approximate unless an epsilon-aware affine correction is applied after the final statistics are frozen.

## Exact no-BatchNorm control

For `Conv -> ReLU -> Conv`, multiplying the first Conv output channel by $s>0$ and dividing the second Conv input channel by $s$ is exact because $\operatorname{ReLU}(sz)=s\operatorname{ReLU}(z)$. This control separates the BatchNorm epsilon issue from the ordinary positive ReLU gauge.

## Scope

The permutation implementation covers torchvision-style ResNet BasicBlocks with two convolutions, BatchNorm after each convolution, optional Conv+BatchNorm projection shortcuts, and a classifier named `fc`. Arbitrary grouped/depthwise convolutions and Bottleneck blocks are outside this implementation and must not inherit its exactness claim.
