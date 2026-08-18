

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



INPUT_DIR  = r""     # e.g. r"C:\Users\user\Pictures\mamo\VLSI RESTORE\vlsi project\Test_NoisyLR\NoisyLR"
OUTPUT_DIR = r""     # e.g. r"C:\Users\user\Pictures\mamo\VLSI RESTORE\restored"
WEIGHTS    = r""     # leave empty: best.pt is found automatically

# =====================================================================

IMG_EXT = (".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


# ----------------------------------------------------------------- model
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.w + self.b


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        d = c * expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, d, 1)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(d // 2, d // 2, 1))
        self.conv2 = nn.Conv2d(d // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, d, 1)
        self.conv4 = nn.Conv2d(d // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.sg(self.dw(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        y = self.conv4(self.sg(self.conv3(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(nn.Module):
    def __init__(self, in_ch, out_ch, width=32, enc=(2, 2, 4), mid=6,
                 dec=(2, 2, 2)):
        super().__init__()
        assert len(enc) == len(dec)
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encs, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        self.decs, self.ups = nn.ModuleList(), nn.ModuleList()
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.tail = nn.Conv2d(c, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.intro(x)
        skips = []
        for e, d in zip(self.encs, self.downs):
            x = e(x)
            skips.append(x)
            x = d(x)
        x = self.mid(x)
        for u, dc, s in zip(self.ups, self.decs, reversed(skips)):
            x = u(x) + s
            x = dc(x)
        return self.tail(x)


class Unrolled(nn.Module):
    """Four unrolled stages sharing one proximal network. The measured
    downsampling kernel is stored as a buffer, so it travels with the
    checkpoint and no separate kernel file is needed."""

    def __init__(self, width=32, stages=4, kernel=None):
        super().__init__()
        self.stages = stages
        self.prox = NAFNet(4 + 2, 4, width=width)
        self.step = nn.Parameter(torch.full((stages,), 0.1))
        if kernel is None:
            k1 = torch.tensor([-0.09375, 0.59375, 0.59375, -0.09375])
            kernel = torch.outer(k1, k1)
        self.register_buffer("K", kernel.float().view(1, 1, *kernel.shape))

    def A(self, x):
        p = self.K.shape[-1] // 2
        return F.conv2d(F.pad(x, (p - 1, p, p - 1, p), mode="reflect"),
                        self.K, stride=2)

    def At(self, r):
        p = self.K.shape[-1] // 2
        out = F.conv_transpose2d(r, self.K, stride=2)
        return out[..., p - 1:p - 1 + r.shape[-2] * 2,
                   p - 1:p - 1 + r.shape[-1] * 2]

    def forward(self, y, sg, sl):
        x = F.interpolate(y.float(), scale_factor=2, mode="bicubic",
                          align_corners=False)
        cond = torch.cat([sg.expand_as(y), sl.expand_as(y)], 1)
        for k in range(self.stages):
            xf, yf = x.float(), y.float()
            var = (sg.float() ** 2
                   + (sl.float() * yf.clamp_min(0.05)) ** 2).clamp_min(1e-3)
            w = 1.0 / var
            w = (w / w.mean(dim=(1, 2, 3), keepdim=True)).clamp(0.0, 20.0)
            x = xf - self.step[k].float() * self.At(w * (self.A(xf) - yf))
            u = F.pixel_unshuffle(x, 2)
            x = x + F.pixel_shuffle(self.prox(torch.cat([u, cond], 1)).float(), 2)
        return x.clamp(-0.5, 1.5)


class Estimator(nn.Module):
    """Predicts the two noise parameters from the input image alone."""

    def __init__(self, w=24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(w * 2, w * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(w * 4, 2)

    def forward(self, y):
        o = self.head(self.body(y).flatten(1))
        sg = F.softplus(o[:, :1]) * 0.1
        ss = F.softplus(o[:, 1:]) * 0.1 + 0.05
        return sg.view(-1, 1, 1, 1), ss.view(-1, 1, 1, 1)


# ------------------------------------------------------------------- io
def find_weights(explicit):
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"--weights not found: {explicit}")
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "checkpoints", "best.pt"),
              os.path.join(here, "best.pt"),
              os.path.join(here, "..", "checkpoints", "best.pt")):
        if os.path.isfile(c):
            return os.path.normpath(c)
    sys.exit("No checkpoint found. Expected checkpoints/best.pt beside this "
             "script, or pass --weights PATH.")


def peek_shape(path):
    """Height and width without decoding the whole image."""
    if os.path.splitext(path)[1].lower() == ".npy":
        with open(path, "rb") as fh:
            shape = np.lib.format.read_magic(fh) and \
                np.lib.format.read_array_header_1_0(fh)[0]
        dims = [d for d in shape if d > 4] or list(shape)
        return tuple(dims[-2:])
    from PIL import Image
    with Image.open(path) as im:
        return (im.size[1], im.size[0])


def load_input(path):
    """Returns (float32 HxW array roughly in [0,1], kind, scale)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.squeeze(np.load(path)).astype(np.float32)
        if a.ndim == 3:                       # HxWxC or CxHxW -> first plane
            a = a[..., 0] if a.shape[-1] <= 4 else a[0]
        return a, "npy", 1.0
    from PIL import Image
    im = Image.open(path)
    if im.mode not in ("L", "I;16", "I", "F"):
        im = im.convert("L")
    a = np.array(im).astype(np.float32)
    scale = 65535.0 if a.max() > 255.0 else (255.0 if a.max() > 1.5 else 1.0)
    return a / scale, "img", scale


def save_output(arr, path, kind, scale):
    if kind == "npy":
        np.save(path, arr.astype(np.float32))
        return
    from PIL import Image
    v = np.clip(arr, 0.0, 1.0) * scale
    if scale > 255.0:
        Image.fromarray(v.astype(np.uint16)).save(path)
    elif scale > 1.0:
        Image.fromarray(v.astype(np.uint8)).save(path)
    else:
        Image.fromarray((v * 255).astype(np.uint8)).save(path)


# ------------------------------------------------------------ inference
def d4(t, k):
    if k >= 4:
        t = torch.flip(t, [-1])
    return torch.rot90(t, k % 4, dims=(-2, -1))


def d4_inv(t, k):
    t = torch.rot90(t, -(k % 4), dims=(-2, -1))
    if k >= 4:
        t = torch.flip(t, [-1])
    return t


def restore_batch(model, est, batch, device, tta):
    """batch: float32 tensor (N,1,H,W). Returns (N,1,2H,2W) clamped to [0,1].

    The network downsamples three times internally, so H and W are padded up
    to a multiple of 8 and the result is cropped back. This lets the script
    accept sizes other than 128x128 without edits."""
    _, _, h, w = batch.shape
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        batch = F.pad(batch, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        sg, sl = est(batch)
        acc = torch.zeros(batch.shape[0], 1, batch.shape[2] * 2,
                          batch.shape[3] * 2, device=device)
        for k in range(tta):
            acc += d4_inv(model(d4(batch, k), sg, sl).float(), k)
        out = acc / tta
    if ph or pw:
        out = out[..., :h * 2, :w * 2]
    return out.clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(
        description="Restore degraded images (speckle + Gaussian noise + 2x "
                    "downsampling).")
    # Accepts flags in any common spelling, or two bare positional paths, so
    # the script works however the benchmarking harness chooses to call it:
    #   python run.py --input_dir IN --output_dir OUT
    #   python run.py --input IN --output OUT
    #   python run.py -i IN -o OUT
    #   python run.py IN OUT
    ap.add_argument("positional", nargs="*", default=[],
                    help="optionally: <input_dir> <output_dir>")
    ap.add_argument("--input_dir", "--input", "--input-dir", "-i",
                    dest="input_dir", default=None,
                    help="directory containing degraded input images")
    ap.add_argument("--output_dir", "--output", "--output-dir", "-o",
                    dest="output_dir", default=None,
                    help="directory to write restored images into")
    ap.add_argument("--weights", "--model", "--checkpoint", dest="weights",
                    default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--tta", type=int, default=8, choices=range(1, 9),
                    metavar="N",
                    help="dihedral variants averaged, 1-8 (default 8); "
                         "use 1 for about 8x faster inference")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    # Resolution order: command-line flags, then bare positional paths, then
    # the INPUT_DIR / OUTPUT_DIR constants at the top of this file.
    if args.input_dir is None and len(args.positional) >= 1:
        args.input_dir = args.positional[0]
    if args.output_dir is None and len(args.positional) >= 2:
        args.output_dir = args.positional[1]
    if not args.input_dir:
        args.input_dir = INPUT_DIR
    if not args.output_dir:
        args.output_dir = OUTPUT_DIR
    if not args.input_dir or not args.output_dir:
        ap.error("no input/output directory given.\n"
                 "  Either set INPUT_DIR and OUTPUT_DIR at the top of "
                 "run.py and run:\n"
                 "      python run.py\n"
                 "  or pass them on the command line:\n"
                 "      python run.py --input_dir IN --output_dir OUT")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    if not os.path.isdir(args.input_dir):
        sys.exit(f"--input_dir is not a directory: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(p for p in glob.glob(os.path.join(args.input_dir, "*"))
                   if os.path.splitext(p)[1].lower() in IMG_EXT
                   and not os.path.basename(p).startswith("._"))
    if not files:
        sys.exit(f"No images found in {args.input_dir}. Supported: "
                 f"{', '.join(IMG_EXT)}")

    wpath = find_weights(args.weights)
    ckpt = torch.load(wpath, map_location=device)
    sd = ckpt.get("ema", ckpt.get("model"))
    if sd is None or "est" not in ckpt:
        sys.exit(f"{wpath} is not a checkpoint from this project "
                 f"(expected 'ema' or 'model', plus 'est').")

    model = Unrolled().to(device).eval()
    est = Estimator().to(device).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit(f"Checkpoint does not match the model definition.\n"
                 f"  missing: {list(missing)[:5]}\n"
                 f"  unexpected: {list(unexpected)[:5]}")
    est.load_state_dict(ckpt["est"])

    n_par = sum(p.numel() for p in model.parameters()) + \
        sum(p.numel() for p in est.parameters())
    print(f"weights   : {wpath}")
    print(f"device    : {device}")
    print(f"parameters: {n_par / 1e6:.2f} M")
    print(f"images    : {len(files)}   tta: {args.tta}x   batch: {args.batch}")

    # Group by shape so every batch is rectangular. Only the shape is read
    # here; pixels are loaded chunk by chunk below, so peak memory stays flat
    # no matter how large the test set is.
    groups = {}
    for p in files:
        try:
            groups.setdefault(peek_shape(p), []).append(p)
        except Exception as e:
            print(f"  skipping {os.path.basename(p)}: {e}")
    if not groups:
        sys.exit("No readable images in the input directory.")

    t0 = time.time()
    done = 0
    last_print = 0
    for shape, paths in groups.items():
        for i in range(0, len(paths), args.batch):
            chunk = paths[i:i + args.batch]
            loaded = [load_input(p) for p in chunk]
            arr = np.stack([a for a, _, _ in loaded])[:, None]
            out = restore_batch(model, est,
                                torch.from_numpy(arr).to(device),
                                device, args.tta).cpu().numpy()
            for j, p in enumerate(chunk):
                _, kind, scale = loaded[j]
                stem = os.path.splitext(os.path.basename(p))[0]
                ext = ".npy" if kind == "npy" else os.path.splitext(p)[1]
                save_output(out[j, 0], os.path.join(args.output_dir, stem + ext),
                            kind, scale)
            done += len(chunk)
            if done - last_print >= 50 or done == len(files):
                last_print = done
                el = time.time() - t0
                print(f"  {done}/{len(files)}  {el:.1f}s  "
                      f"({el / max(done, 1) * 1000:.1f} ms/image)", flush=True)

    el = time.time() - t0
    print(f"\nrestored {done} images in {el:.1f}s "
          f"({el / max(done, 1) * 1000:.1f} ms per image on {device})")
    print(f"written to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()