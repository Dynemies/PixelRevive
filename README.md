# Blind Joint Despeckling and 2× Super-Resolution

**KLA PS-01 — AI-Based Restoration of Degraded Images** · i4C Hackathon

Removes multiplicative speckle and additive Gaussian noise and performs 2×
super-resolution in a single forward pass. 4.02 M parameters.

**Team:** T. Vamsi Krishna Sai · M. Harsha Vardhan Reddy · Tharun Jawaharlal S · Darshini R

---

## Setup

```bash
git clone <this-repository>
cd <repository-folder>
pip install -r requirements.txt
```

Python 3.9 or newer. Inference needs only `torch`, `numpy` and `Pillow`.

## Run inference

```bash
python evaluate.py --input_dir /path/to/test_images --output_dir /path/to/results
```

That is all. The script loads `checkpoints/best.pt` itself — nothing to
configure, download or edit. It works from any working directory, uses a GPU if
one is present and the CPU otherwise, and needs no network access.

These are all equivalent:

```bash
python evaluate.py --input_dir IN --output_dir OUT
python evaluate.py -i IN -o OUT
python evaluate.py IN OUT
```

**Input:** `.npy`, `.png`, `.tif`, `.jpg`. **Output:** one restored image per
input, same filename, same format. A 128×128 input gives a 256×256 output.

### Optional flags

|| Flag        | Default               | Meaning                                                                                                          |
| ----------- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--tta`     | `8`                   | Dihedral variants averaged. **Use `--tta 1` when measuring inference time** — about 8× faster for ~0.15 dB less. |
| `--device`  | `auto`                | Force `cpu` or `cuda`                                                                                            |
| `--batch`   | `8`                   | Images per forward pass                                                                                          |
| `--weights` | `checkpoints/best.pt` | Alternative checkpoint                                                                                           |

### Speed

| Configuration       | Per Image | 400 Images |
| ------------------- | --------: | ---------: |
| Tesla T4, `--tta 1` |   0.073 s |      ~30 s |
| Tesla T4, `--tta 8` |    0.58 s |     ~4 min |
| CPU, `--tta 1`      |     0.9 s |     ~6 min |

---

## Results

100 held-out validation images, identical for every row.

| Method                 |    PSNR ↑ |     SSIM ↑ |    LPIPS ↓ |
| ---------------------- | --------: | ---------: | ---------: |
| Bicubic (no denoising) |     22.82 |     0.5747 |     0.4292 |
| Lee filter (1980)      |     25.65 |     0.6960 |     0.3660 |
| Homomorphic log + BM3D |     26.91 |     0.7505 |     0.3336 |
| **Our model**          | **28.39** | **0.7921** | **0.2718** |

+5.57 dB over bicubic, +1.48 dB over the strongest classical baseline. Over all
384 validation images the model reaches 28.50 dB.

Validation is held out by **scene**: the 3,200 training images contain only
2,589 distinct scenes, so a random split would leak overlapping crops.

---

## Repository contents

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


## Reproduce training

```bash
python src/characterise.py --root /path/to/dataset   # recover the forward model
python src/train.py                                  # ~6.6 h on one Tesla T4
```

Trained on one NVIDIA Tesla T4 (16 GB), Kaggle free tier, 6.6 hours, at no cost.

## Approach in one paragraph

We recovered the degradation model from 3,200 training pairs before choosing an
architecture: the speckle is Gamma with shape 11–55, the downsampling is
separable bicubic with a = −0.75, and the Gaussian noise is applied last. The
model is a four-stage unrolled solver whose learned denoiser is a NAFNet U-Net
shared across stages, conditioned on noise parameters predicted from the input.
Loss is Charbonnier plus a frequency-domain term; no adversarial loss, which
would lower PSNR and invent detail that was never measured. Full detail, and the
ablations showing which of our design choices actually helped, are in
[`docs/METHOD.md`](docs/METHOD.md).
