# Results

Blind Joint Despeckling and 2× Super-Resolution — KLA PS-01

Every number below was measured. Nothing is estimated or carried over from
published work. Where a result is a bound we constructed rather than a
published method, it is labelled as such.

---

## 1. Headline

| | |
|---|---|
| **PSNR** | **28.39 dB** (100 held-out images) · **28.50 dB** (all 384) |
| **SSIM** | 0.7921 |
| **LPIPS** | 0.2718 |
| Improvement over bicubic | **+5.57 dB** |
| Improvement over best classical method | **+1.48 dB** |
| Parameters | 4.02 M |
| Training | 6.6 h on one Tesla T4, free tier, zero cost |
| Inference | 0.073 s/image (T4, single pass) · 0.58 s/image (8× ensemble) |

---

## 2. Comparison against baselines

100 held-out validation images, identical for every row. Each classical method
denoises at 128×128 then upsamples 2× bicubically; our model does both jointly.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic (no denoising) | 22.82 | 0.5747 | 0.4292 |
| Frost filter (1982) | 22.92 | 0.5436 | 0.5940 |
| Lee filter (1980) | 25.65 | 0.6960 | 0.3660 |
| Homomorphic log + BM3D | 26.91 | 0.7505 | 0.3336 |
| **Our model** | **28.39** | 0.7921 | **0.2718** |
| Oracle shrinkage *(bound we derived, not a published method)* | 28.18 | **0.8019** | 0.2892 |

Best PSNR and best LPIPS of everything tested. Second on SSIM, behind an
estimator that is allowed to see the ground truth.

**On the oracle row.** It applies the mathematically optimal shrinkage to every
DCT coefficient *while knowing the true clean image*, so no real shrinkage-based
denoiser can beat it. It bounds *local transform-domain shrinkage followed by
bicubic upsampling*. Our model exceeds it by 0.21 dB because it learns the
upsampling prior instead of using bicubic, and exploits non-local structure that
local shrinkage cannot see. It is a reference point for how much noise is
removable at all, not a competitor.

**On the Frost row.** It is worse than bicubic on SSIM and much worse on LPIPS.
It blurs without removing enough speckle. Reported as measured.

Reproduce: `python src/comparison_table.py`

---

## 3. Training

Unrolled solver, width 32, 4 stages, batch 32, learning rate 4.5e-4,
120-epoch one-cycle schedule, stopped at epoch 91.

| Epoch | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 91 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val PSNR (dB) | 22.99 | 25.21 | 26.22 | 27.19 | 27.81 | 28.13 | 28.33 | 28.42 | 28.47 | 28.50 |

Converged: +0.03 dB across the final eleven epochs. Zero failed batches and zero
out-of-memory events across every run.

Validation is held out by **scene**, not at random. The 3,200 training images
resolve into only 2,589 distinct scenes, the largest cluster holding 122
overlapping crops, so a random split would place crops of one scene on both
sides of the boundary and inflate the score.

---

## 4. Ablation study

Every row retrained from scratch on an identical 25-epoch schedule and the same
seed, so the comparison is like for like. The proximal network is shared across
stages, so all six rows have the same parameter count and the depth row isolates
solver depth from model capacity.

| Configuration | PSNR (dB) | Change | Parameters |
|---|---|---|---|
| Full model | 26.899 | reference | 4.02 M |
| Without the data-fidelity term | 26.897 | **−0.002** | 4.02 M |
| Without noise conditioning | 26.906 | **+0.007** | 4.02 M |
| Without synthetic re-degradation | 27.122 | **+0.223** | 4.02 M |
| One stage instead of four | 26.538 | −0.361 | 4.02 M |
| Without the frequency-domain loss | 26.850 | −0.049 | 4.02 M |

**Three of our four design contributions do not work.** Only solver depth
(+0.361 dB) and the frequency loss (+0.049 dB) earn their place. We report this
rather than omitting it.

Reproduce: `python src/ablations.py`

---

## 5. Measured degradation model

Recovered from 3,200 training pairs before any architecture was chosen.

```
y = K(x) · Gamma(L, 1/L) + N(0, σg²)        K = separable bicubic, stride 2
```

Fit quality of `Var(residual) = σg² + σs²·d²`: median R² **0.974**, above 0.90
on **84.8 %** of pairs.

### Parameters

| Parameter | Min | p1 | Median | p99 | Max |
|---|---|---|---|---|---|
| σg (additive) | 0.000 | 0.000 | 0.0208 | 0.130 | 0.198 |
| L (speckle shape) | 11.3 | 15.8 | 33.8 | 51.5 | 54.9 |
| σs = 1/√L | — | 0.104 | 0.167 | 0.262 | — |

σg is not uniform: it decays from zero, and 25.8 % of images have σg < 0.005.
σs and L are the same parameter, so there are only **two unknowns per image**,
and they are effectively independent (correlation −0.045).

### The speckle is Gamma, not Gaussian

| Image | Measured skewness | Gaussian requires |
|---|---|---|
| 000000 | +0.21 | 0.00 |
| 000002 | +0.23 | 0.00 |
| 000009 | +0.30 | 0.00 |
| 000012 | +0.34 | 0.00 |
| 000013 | +0.38 | 0.00 |

Confirmed independently: a maximum-likelihood Gamma fit returns a mean of
**1.000 ± 0.004** in every image, as a unit-mean Gamma must, and its implied
1/√L matches the moment-based σs to within 0.008.

### The downsampling kernel

4×4 core, sum 1.0000, **rank-1 energy 99.9728 %**, condition number 1.12×10⁴.

```
[[  0.0150  -0.0535  -0.0587   0.0128 ]
 [ -0.0486   0.3345   0.3575  -0.0486 ]
 [ -0.0539   0.3281   0.3516  -0.0549 ]
 [  0.0073  -0.0417  -0.0511   0.0042 ]]
```

Recovered 1D factor: `[−0.0978, 0.5931, 0.5842, −0.0795]`

| Candidate | Distance |
|---|---|
| **Bicubic a = −0.75** | **0.0368** |
| Bicubic a = −0.50 | 0.0768 |
| Box average 2×2 | 0.2385 |

PIL and PyTorch use a = −0.5; OpenCV's `INTER_CUBIC` uses a = −0.75. The
recovered coefficient therefore identifies which library generated the dataset.

### The Gaussian noise is applied last

High-pass filtering the input with a Laplacian leaves mostly noise. Its
autocorrelation matches white noise, which it could not if the noise had been
added before downsampling and low-pass filtered by the decimation.

| Lag | White-noise prediction | Measured range |
|---|---|---|
| (0,1) and (1,0) | −0.40 | −0.31 to −0.46 |
| (1,1) | +0.10 | +0.04 to +0.18 |

### Two further dataset properties

- **Ground truth is per-image min-max normalised:** minimum exactly 0 and
  maximum exactly 1 in **3,200 of 3,200** images.
- **The test set is noisier than the training set:** median standard deviation
  0.198 → 0.215, fraction of pixels above 1.0 0.0084 → 0.0126, maximum
  1.392 → 1.445.

Reproduce: `python src/characterise.py --root <dataset>`

---

## 6. Error budget

Measured before building anything, to establish how much improvement exists.

| | PSNR |
|---|---|
| Bicubic upsampling of the noisy input | 22.87 dB |
| **Our model** | **28.50 dB** |
| Bicubic upsampling of the *clean* low-resolution image | 33.04 dB |

The third row is perfect denoising with no learned prior at all, so noise
removal alone is worth **10.2 dB** and everything an image prior can add sits
above 33 dB. The problem is **noise-limited, not capacity-limited**, which is
why the network was not scaled up.

Per-image ceilings track σg: clean images reach 42 dB, while high-noise images
cap in the low twenties and dominate the average error.

33.04 dB is not an absolute ceiling — a learned prior beats bicubic by 3–4 dB on
clean 2× data — but reaching it would require near-perfect denoising, and that
information is genuinely destroyed.

---

## 7. Negative results

All measured, all reported.

**The physics data-fidelity term contributes −0.002 dB.** The learned step sizes
show why: initialised at 0.100, they became `[+0.075, −0.047, −0.029, +0.076]`.
Two went negative, meaning those stages ascend the consistency objective rather
than descending it — the network stopped using the operator as a solver.

| Quantity | Value | Relative |
|---|---|---|
| Consistency residual at initialisation | 0.0019 | 0.41 % of signal scale |
| Noise the model must actually remove | 0.0920 | 48× larger |

Bicubic upsampling followed by the measured downsampling is nearly the identity,
so the initial estimate already satisfies measurement consistency. For this
forward model, data consistency is essentially free and the task is about **98 %
pure denoising**. The unrolled scaffold degenerates into a deep residual
denoiser. This also explains why conditioning does not help: the proximal
network can read the noise level off the image itself.

**Widening the synthetic noise distribution costs 0.223 dB.** We sampled σg up
to 0.22 against a true maximum of 0.198, from a distribution shaped unlike the
real one, whose median is only 0.021. Half of every batch trained on noise
levels that do not occur. The parameter estimator confirms it: its L1 error
falls from 0.050 to 0.034 when trained on the real pairs alone.

**Exploiting the known [0,1] output range costs 4.4 dB.** Since every ground
truth image spans exactly [0,1], rescaling each prediction to that range looked
free. Measured, PSNR fell from 23.04 to 18.65. The estimator that minimises
squared error is the conditional mean, which necessarily has lower variance than
the truth, so our predictions are correctly shrunk toward the middle. Stretching
them back moves away from the optimum, and one outlying pixel sets the scale for
the whole image.

---

## 8. Noise estimator

A small CNN predicts the two noise parameters from the degraded image alone.

| | Training set | Test set | Change |
|---|---|---|---|
| Median predicted σg | 0.021 | 0.0302 | **+44 %** |
| Median predicted 1/√L | 0.167 | 0.1703 | +2 % |

This independently reproduces what raw image statistics show — that the test set
carries more additive noise and identical speckle — using a completely different
method. Two unrelated measurements agreeing indicates the estimator carries real
information rather than predicting the training mean.

---

## 9. Compute and efficiency

| Item | Value |
|---|---|
| Parameters | 4.02 M |
| Multiply-accumulates | 2.33 GMAC per stage, 9.30 GMAC across four stages |
| Training hardware | One NVIDIA Tesla T4 (16 GB), Kaggle free tier |
| Main training run | 91 epochs, 6.6 h, 260 s per epoch |
| Ablation study | 6 configurations × 25 epochs, 9.2 h |
| Peak GPU memory | 3.5 GB with gradient checkpointing, 13.2 GB without |
| Checkpoint size | 16.5 MB |
| Failed or skipped batches | 0 |
| Monetary cost | Zero |

Gradient checkpointing on the proximal network cut peak memory by 3.7× for 17 %
more time per epoch.

### Inference

| Configuration | Per image | 400 images |
|---|---|---|
| Tesla T4, single pass | 0.073 s | ~30 s |
| Tesla T4, 8× self-ensemble | 0.58 s | ~4 min |
| CPU, single pass | 0.9 s | ~6 min |
| CPU, 8× self-ensemble | 7.4 s | ~50 min |

---

## 10. Test set output

400 restored images, verified before submission.

| Check | Result |
|---|---|
| File count | 400, numbered 000000–000399, contiguous |
| Shape and type | 256×256 float32, all files |
| Non-finite values | 0 |
| Values within [0,1] | 400 / 400 |
| Structure | flat directory, no nesting |

Reproduce: `python evaluate.py --input_dir <test> --output_dir <out>`

---

## 11. What is implemented, and what is not

**Implemented:** four-stage unrolled solver with the fitted bicubic operator, a
variance-weighted least-squares data term, the learned noise-parameter estimator
with FFDNet-style conditioning, a NAFNet proximal network run entirely at low
resolution with one pixel-shuffle, Charbonnier plus orthonormal frequency loss,
EMA weights, 8× dihedral self-ensemble, scene-grouped validation split,
on-the-fly re-degradation, and gradient checkpointing.

**Designed but not implemented,** stated so that nothing here is overclaimed:

1. The exact Gamma × Gaussian negative log-likelihood. What is implemented is a
   Gaussian second-moment approximation to it.
2. A self-calibrating loop that re-estimates the noise parameters from each
   stage's reconstruction rather than once at the start.
3. The homomorphic log domain with ψ(L) bias correction.
4. Per-stage projection onto the known feasible set.

The ablations point away from added mechanism and toward the denoiser itself,
which is where the remaining decibels are. The proximal network is
architecture-agnostic and a stronger backbone can replace it without changing
anything else.
