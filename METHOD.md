# Method and analysis

Detail behind the summary in the main [README](../README.md).

---

## What we measured

From 3,200 training pairs, before choosing an architecture:

```
y = K(x) · Gamma(L, 1/L) + N(0, sigma_g^2)     K = separable bicubic, stride 2
```

**The speckle is Gamma, not Gaussian.** The skewness of the multiplicative
component on bright pixels is +0.21 to +0.38; Gaussian speckle requires exactly
zero. An independent maximum-likelihood fit returns a mean of 1.000 ± 0.004, as
a unit-mean Gamma must.

**The downsampling kernel is bicubic with a = −0.75.** A weighted least-squares
fit over 300 pairs gives a 4×4 core that is 99.97 % rank-1 — separable, which
fitting noise never is. Its 1D factor is `[−0.098, 0.593, 0.584, −0.080]`,
sitting 0.037 from bicubic a = −0.75, 0.077 from a = −0.5 and 0.239 from a box
average. PIL and PyTorch use a = −0.5; OpenCV's `INTER_CUBIC` uses −0.75.

**The Gaussian noise is applied last.** High-pass filtering the input leaves
mostly noise, whose autocorrelation matches white noise (−0.40 predicted, −0.31
to −0.46 measured). Noise added before downsampling would have been low-pass
filtered by the decimation and come out correlated.

**Parameter ranges.** Sigma_g from 0 to 0.198, median 0.021, decaying from zero.
Gamma shape L from 11.3 to 54.9, median 33.8. Median fit R² of 0.974.

Full detail in [`results/measured_forward_model.json`](results/measured_forward_model.json).

---

## Method

Four unrolled stages. Each takes a data-fidelity gradient step through the
measured downsampling operator, then applies a learned proximal denoiser. The
denoiser is a NAFNet U-Net (LayerNorm, SimpleGate, simplified channel attention)
and is **shared across all four stages**, so depth costs no extra parameters. A
small separate network estimates the two noise parameters from the degraded
image and feeds them to the solver as conditioning maps.

Every convolution runs at 128×128 with a single pixel-shuffle to 256×256 —
working at full resolution costs four times the compute for no gain.

Loss is Charbonnier on pixels plus an orthonormalised frequency-domain L1 term.
No adversarial or perceptual loss: both trade fidelity for texture, lower PSNR,
and invent detail that was never measured.

---

## Ablation study

Every row retrained from scratch on an identical 25-epoch schedule and seed.

| Configuration | PSNR (dB) | Change |
|---|---|---|
| Full model | 26.899 | reference |
| Without the data-fidelity term | 26.897 | **−0.002** |
| Without noise conditioning | 26.906 | **+0.007** |
| Without synthetic re-degradation | 27.122 | **+0.223** |
| One stage instead of four | 26.538 | −0.361 |
| Without the frequency-domain loss | 26.850 | −0.049 |

**Three of our four design contributions do not work,** and we report that
rather than hiding it. Only solver depth and the frequency loss earn their place.

The learned step sizes explain why. Initialised at 0.100, they became
`[+0.075, −0.047, −0.029, +0.076]` — two went negative, meaning those stages
ascend the consistency objective rather than descending it. The cause is
measurable: bicubic upsampling followed by the measured downsampling is nearly
the identity, so the initial estimate already satisfies measurement consistency.

| Quantity | Value | Relative |
|---|---|---|
| Consistency residual at initialisation | 0.0019 | 0.41 % of signal scale |
| Noise the model must actually remove | 0.0920 | 48× larger |

For a forward model whose downsampling is bicubic, data consistency is
essentially free and the task is about 98 % pure denoising, so the unrolled
scaffold degenerates into a deep residual denoiser.

Two further ideas also failed on measurement. Widening the synthetic noise
distribution beyond the measured range cost 0.223 dB. Rescaling predictions to
the known [0, 1] output range cost 4.4 dB, because the estimator minimising
squared error is the conditional mean and is correctly shrunk toward the middle.

---
