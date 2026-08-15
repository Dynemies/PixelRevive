# Blind Joint Despeckling and 2× Super-Resolution

**KLA PS-01 — AI-Based Restoration of Degraded Images** · i4C Hackathon

Removes multiplicative speckle and additive Gaussian noise and performs 2×
super-resolution in a single forward pass. 4.02 M parameters.

**Team:** T. Vamsi Krishna Sai · M. Harsha Vardhan Reddy · Tharun Jawaharlal S · Darshini R

---

## Setup

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

Python 3.9 or newer. Inference needs only `torch`, `numpy` and `Pillow`.

## Run inference

```bash
python evaluation.py --input_dir /path/to/test_images --output_dir /path/to/results
```

That is all. The script loads `best.pt` from the repository folder itself —
nothing to configure, download or edit. It works from any working directory,
uses a GPU when one is present and the CPU otherwise, and needs no network
access.

These are all equivalent:

```bash
python evaluation.py --input_dir IN --output_dir OUT
python evaluation.py -i IN -o OUT
python evaluation.py IN OUT
```

You can also open `evaluation.py` and set `INPUT_DIR` and `OUTPUT_DIR` at the top,
then run `python evaluation.py` with no arguments.

**Input:** `.npy`, `.png`, `.tif`, `.jpg`. **Output:** one restored image per
input, same filename, same format. A 128×128 input gives a 256×256 output.

### Optional flags

| Flag        | Default   | Meaning                                                                                                          |
| ----------- | --------- | ---------------------------------------------------------------------------------------------------------------- |
| `--tta`     | `8`       | Dihedral variants averaged. **Use `--tta 1` if inference time is measured** — about 8× faster for ~0.15 dB less. |
| `--device`  | `auto`    | Force `cpu` or `cuda`                                                                                            |
| `--batch`   | `8`       | Images per forward pass                                                                                          |
| `--weights` | `best.pt` | Alternative checkpoint                                                                                           |

### Speed

| Configuration       | Per Image | 400 Images |
| ------------------- | --------: | ---------: |
| Tesla T4, `--tta 1` |   0.073 s |      ~30 s |
| Tesla T4, `--tta 8` |    0.58 s |     ~4 min |
| CPU, `--tta 1`      |     0.9 s |     ~6 min |


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

Validation is held out by **scene**: the 3,200 training images contain only
2,589 distinct scenes, so a random split would leak overlapping crops.

Full numbers, the ablation study and the measured degradation model are in
[RESULTS.md](RESULTS.md).

---

## Files

| Path                  | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `evaluate.py`         | Standalone inference. Input directory in, restored images out. |
| `checkpoints/best.pt` | Trained weights, 16.5 MB. Loaded automatically.                |
| `src/train.py`        | Reproduces training from scratch (~6.6 h on one T4)            |
| `outputs/`            | Our restored outputs on the provided test set                  |
| `requirements.txt`    | Dependencies                                                   |
| `docs/METHOD.md`      | How the degradation was measured, architecture, ablations      |
| `results/`            | Measured forward model, ablation and comparison results        |
| `src/`                | Characterisation, ablation, comparison and demo scripts        |


The restored test outputs are split across five folders because GitHub's web
uploader caps each upload; together they hold all 400 restored images.

---

## Approach

We recovered the degradation model from 3,200 training pairs before choosing an
architecture:

```
y = K(x) · Gamma(L, 1/L) + N(0, sigma_g^2)     K = separable bicubic, stride 2
```

The speckle is Gamma with shape 11–55, not Gaussian — its skewness is +0.21 to
+0.38 where Gaussian speckle requires zero, and a maximum-likelihood fit returns
unit mean to within 0.4 %. The downsampling kernel is separable bicubic with
a = −0.75, recovered at 99.97 % rank-1 separability. The Gaussian noise is
applied last, proved by residual whiteness.

The model is a four-stage unrolled solver whose learned denoiser is a NAFNet
U-Net shared across stages, conditioned on noise parameters predicted from the
input by a small CNN. Loss is Charbonnier plus a frequency-domain term. No
adversarial loss, which would lower PSNR and invent detail that was never
measured.

We also ablated our own design choices and report that three of four contributed
nothing measurable — see [RESULTS.md](RESULTS.md) for the numbers and the
explanation.

## Reproduce training

```bash
python train.py
```

Trained on one NVIDIA Tesla T4 (16 GB), Kaggle free tier, 16.6 hours, at no cost.

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
