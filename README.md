# PixelRevive

**KLA PS-01 — AI-Based Restoration of Degraded Images** · i4C Hackathon

Removes multiplicative speckle and additive Gaussian noise and performs 2×
super-resolution in a single forward pass. 4.02 M parameters. No internet
access, API keys, model downloads or manual configuration required.

**Team:** T. Vamsi Krishna Sai · M. Harsha Vardhan Reddy · Tharun Jawaharlal S · Darshini R

---

## Setup

```bash
git clone https://github.com/Dynemies/PixelRevive.git
cd PixelRevive
pip install -r requirements.txt
```

Python 3.9 or newer. Inference needs only `torch`, `numpy` and `Pillow`, all
pinned in `requirements.txt`.

## Run

```bash
python run.py --input_dir /path/to/degraded --output_dir /path/to/restored
```

That is the whole procedure. The script loads `best.pt` from the repository
folder itself, so there is nothing to configure, download or edit. It selects an
NVIDIA GPU automatically when one is present and falls back to CPU otherwise.

`evaluation.py` is an identical copy kept under its earlier name; either file
works.

### Accepted calling conventions

```bash
python run.py --input_dir IN --output_dir OUT
python run.py --input IN --output OUT
python run.py -i IN -o OUT
python run.py IN OUT
```

### Input and output

- Reads every `.npy` file in the input directory. Other file types and
  subdirectories are ignored.
- Creates the output directory, including any missing parent directories.
- Writes exactly one `.npy` output per `.npy` input, keeping the input filename
  unchanged.
- Outputs are float32 grayscale of shape `(H, W)`, clipped to `[0, 1]`, with no
  NaN or Inf values.
- The model upsamples 2×: a 128×128 input produces a 256×256 output.
- Inputs are read as float without rescaling, so values outside `[0, 1]` are
  handled correctly. Inputs shaped `(H, W, 1)` and float64 inputs are accepted.
- `.png`, `.tif` and `.jpg` inputs are also supported and return the same
  format, but `.npy` in always gives `.npy` out.

### Optional flags

| Flag        | Default   | Meaning                                                                                                                            |
| ----------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `--tta`     | `8`       | Dihedral variants averaged, 1–8. **Pass `--tta 1` if inference time is being measured** — roughly 8× faster for 0.06 dB less PSNR. |
| `--device`  | `auto`    | Force `cpu` or `cuda`                                                                                                              |
| `--batch`   | `8`       | Images per forward pass                                                                                                            |
| `--weights` | `best.pt` | Alternative checkpoint                                                                                                             |


### Speed

| Configuration        | Per Image | 400 Images |
| -------------------- | --------: | ---------: |
| NVIDIA T4, `--tta 1` |   0.073 s |      ~30 s |
| NVIDIA T4, `--tta 8` |    0.58 s |     ~4 min |
| CPU, `--tta 1`       |     0.9 s |     ~6 min |

---

## Results

100 held-out validation images, identical for every row. Classical methods
denoise at 128×128 then upsample bicubically; our model does both jointly.
| Method                 |    PSNR ↑ |     SSIM ↑ |    LPIPS ↓ |
| ---------------------- | --------: | ---------: | ---------: |
| Bicubic (no denoising) |     22.82 |     0.5747 |     0.4292 |
| Lee filter (1980)      |     25.65 |     0.6960 |     0.3660 |
| Homomorphic log + BM3D |     26.91 |     0.7505 |     0.3336 |
| **Our model**          | **28.39** | **0.7921** | **0.2718** |


**+5.57 dB over bicubic, +1.48 dB over the strongest classical baseline.** Over
all 384 validation images the model reaches 28.50 dB.

Validation is held out by **scene**, not at random: the 3,200 training images
contain only 2,589 distinct scenes, so a random split would leak overlapping
crops across the boundary and inflate the score.

Full numbers, the ablation study and the measured degradation model are in
[RESULTS.md](RESULTS.md); the method write-up is in [METHOD.md](METHOD.md).

---

## Repository contents

### Running the model

| File               | Purpose                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `run.py`           | **Main entry point.** Input directory in, restored `.npy` files out. Loads the model itself. |
| `evaluation.py`    | Identical copy of `run.py` under its earlier name                                            |
| `best.pt`          | Trained weights, 16.5 MB — EMA model, noise estimator and the measured downsampling kernel   |
| `requirements.txt` | Complete `pip freeze` from the training environment, all versions pinned                     |


### Restored test outputs
| Folder                            | Contents                                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `npy 1` … `npy 5`                 | **All 400 restored test images as float32 `.npy`**, numbered `000000` to `000399`, shape `(256, 256)`, values within `[0, 1]` |
| `restore png 1` … `restore png 5` | The same 400 images as 16-bit PNG, for quick visual inspection                                                                |



Both sets are split across five folders only because of upload size limits.
The `.npy` set is the model's exact output; the PNG set is provided for
convenience and has a round-trip error of 7.7 × 10⁻⁶, an effective ceiling above
100 dB.

### Reproducing the work

| File | Purpose |
|---|---|
| `train.py` | Reproduces training from scratch, about 6.6 h on one NVIDIA T4 |
| `characterise.py` | Recovers the degradation model from the training pairs and builds the scene-grouped split |

### Results and documentation

| File | Purpose |
|---|---|
| `RESULTS.md` | Every measured result: comparison table, ablations, error budget, negative results |
| `METHOD.md` | How the degradation was measured, the architecture, and why three of our four design ideas failed |
| `comparison_table.json` | PSNR, SSIM and LPIPS for every method |
| `ablation_results.json` | All six ablation configurations |
| `measured_forward_model.json` | Recovered degradation parameters and the evidence for each |
| `measured_kernel.npy` | The recovered 4×4 downsampling kernel |
| `input_examples.png` | Example degraded inputs from the training set |

---

## Approach

We recovered the degradation model from 3,200 training pairs before choosing an
architecture:

```
y = K(x) · Gamma(L, 1/L) + N(0, sigma_g^2)     K = separable bicubic, stride 2
```

The speckle is Gamma with shape 11–55, not Gaussian: its skewness is +0.21 to
+0.38 where Gaussian speckle requires exactly zero, and a maximum-likelihood fit
returns unit mean to within 0.4 %. The downsampling kernel is separable bicubic
with a = −0.75, recovered at 99.97 % rank-1 separability. The Gaussian noise is
applied last, proved by residual whiteness.

The model is a four-stage unrolled solver whose learned denoiser is a NAFNet
U-Net shared across all stages, conditioned on noise parameters predicted from
the input by a small CNN. Loss is Charbonnier plus a frequency-domain term. No
adversarial loss, which would lower PSNR and invent detail that was never
measured.

We also ablated our own design choices and report that three of four contributed
nothing measurable. See [METHOD.md](METHOD.md) for the numbers and the
explanation.

## Reproducing training

```bash
python characterise.py --root /path/to/dataset   # recover the forward model
python train.py                                  # about 6.6 h on one NVIDIA T4
```

Trained on one NVIDIA Tesla T4 (16 GB) on Kaggle's free tier, in 9.6 hours, at
no cost.

---

## References

1. Monga, Li and Eldar, *Algorithm Unrolling*, IEEE Signal Processing Magazine, 2021.
2. Chen, Chu, Zhang and Sun, *Simple Baselines for Image Restoration* (NAFNet), ECCV, 2022.
3. Zhang, Zuo and Zhang, *FFDNet*, IEEE TIP, 2018.
4. Chierchia, Cozzolino, Poggi and Verdoliva, *SAR Image Despeckling Through CNNs*, IGARSS, 2017.
5. Parrilli, Poderico, Angelino and Verdoliva, *SAR-BM3D*, IEEE TGRS, 2012.
6. Lee, *Digital Image Enhancement and Noise Filtering by Use of Local Statistics*, IEEE TPAMI, 1980.
7. Frost, Stiles, Shanmugan and Holtzman, *A Model for Radar Images*, IEEE TPAMI, 1982.
8. Zhang, Isola, Efros, Shechtman and Wang, *LPIPS*, CVPR, 2018.
