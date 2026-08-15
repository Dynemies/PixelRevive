

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, time, glob, math, random, zipfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True
dev = "cuda" if torch.cuda.is_available() else "cpu"
LR_SIZE = 128
print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "")
if dev == "cuda":
    _f, _t = torch.cuda.mem_get_info()
    print(f"GPU memory: {_f/2**30:.2f} GB free of {_t/2**30:.2f} GB")
    if _f / 2**30 < 8.0:
        raise SystemExit("Under 8 GB free - restart the session first.")


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
    """LayerNorm -> 1x1 -> dwconv -> SimpleGate -> SCA -> 1x1, plus an FFN half."""
    def __init__(self, c, expand=2):
        super().__init__()
        d = c * expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, d, 1)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
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
    def __init__(self, in_ch, out_ch, width=32, enc=(2, 2, 4), mid=6, dec=(2, 2, 2)):
        super().__init__()
        assert len(enc) == len(dec), (
            f"encoder has {len(enc)} levels but decoder has {len(dec)} - the "
            f"output resolution will not match the input")
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
            x = e(x); skips.append(x); x = d(x)
        x = self.mid(x)
        for u, dc, s in zip(self.ups, self.decs, reversed(skips)):
            x = u(x) + s
            x = dc(x)
        return self.tail(x)


# ------------------------------------------------------------ the models
class Baseline(nn.Module):
    """Direct: [y, sigma_g map, 1/sqrt(L) map] -> HR estimate.
    Runs entirely at LR resolution, one PixelShuffle at the end."""
    def __init__(self, width=32):
        super().__init__()
        self.net = NAFNet(3, 4, width=width, enc=(2, 2, 4), mid=6,
                          dec=(2, 2, 2))
        self.up = nn.PixelShuffle(2)
        # Zero the output conv: at init the model returns EXACTLY the bicubic
        # upsample, so training starts from a known ~23 dB and learns a
        # correction. Random init instead injects noise the model must undo.
        nn.init.zeros_(self.net.tail.weight); nn.init.zeros_(self.net.tail.bias)

    def forward(self, y, sg, sl):
        cond = torch.cat([sg.expand_as(y), sl.expand_as(y)], 1)
        base = F.interpolate(y, scale_factor=2, mode="bicubic", align_corners=False)
        return base + self.up(self.net(torch.cat([y, cond], 1)))


class Unrolled(nn.Module):
    """4 stages sharing one prox. The HR estimate is pixel-unshuffled to LR
    resolution (lossless) so every conv stays cheap. Data term uses the
    measured bicubic operator."""
    def __init__(self, width=32, stages=4, kernel=None, use_ckpt=True):
        super().__init__()
        self.stages = stages
        self.use_ckpt = use_ckpt
        self.prox = NAFNet(4 + 2, 4, width=width, enc=(2, 2, 4), mid=6,
                           dec=(2, 2, 2))   # 4 unshuffled + 2 cond
        # Zero the prox output conv so the solver begins as the pure
        # data-fidelity iteration and learns a correction on top of it.
        nn.init.zeros_(self.prox.tail.weight); nn.init.zeros_(self.prox.tail.bias)
        self.step = nn.Parameter(torch.full((stages,), 0.1))
        if kernel is None:
            k1 = torch.tensor([-0.0625, 0.5625, 0.5625, -0.0625])
            kernel = torch.outer(k1, k1)
        self.register_buffer("K", kernel.float().view(1, 1, *kernel.shape))

    def A(self, x):
        p = self.K.shape[-1] // 2
        return F.conv2d(F.pad(x, (p - 1, p, p - 1, p), mode="reflect"),
                        self.K, stride=2)

    def At(self, r):
        p = self.K.shape[-1] // 2
        out = F.conv_transpose2d(r, self.K, stride=2)
        return out[..., p - 1:p - 1 + r.shape[-2] * 2, p - 1:p - 1 + r.shape[-1] * 2]

    def forward(self, y, sg, sl):
        x = F.interpolate(y.float(), scale_factor=2, mode="bicubic",
                          align_corners=False)
        cond = torch.cat([sg.expand_as(y), sl.expand_as(y)], 1)
        for k in range(self.stages):
            # ---- data-fidelity gradient, ALWAYS in fp32.
            # The noise-model weight 1/(sg^2 + (sl*y)^2) reaches ~1e5 when
            # sg is 0, which overflows fp16 (max 65504) -> inf -> NaN.
            # So: compute outside autocast, floor the variance, and
            # normalise per image so the step size means the same thing
            # regardless of noise level.
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                xf, yf = x.float(), y.float()
                var = (sg.float() ** 2
                       + (sl.float() * yf.clamp_min(0.05)) ** 2).clamp_min(1e-3)
                w = 1.0 / var
                w = (w / w.mean(dim=(1, 2, 3), keepdim=True)).clamp(0.0, 20.0)
                x = xf - self.step[k].float() * self.At(w * (self.A(xf) - yf))
            # ---- learned prox at LR resolution via lossless reshape.
            # Checkpointed: with 4 stages the stored activations are what
            # fills a T4. Recomputing them in backward costs ~30% time and
            # cuts activation memory roughly 4x.
            if self.use_ckpt and self.training:
                dx = checkpoint(self._prox_step, x, cond, use_reentrant=False)
            else:
                dx = self._prox_step(x, cond)
            # NO hard clamp here. clamp() has zero gradient outside its
            # range, so a saturated estimate can never recover - that is a
            # self-locking failure, not a safety net. Bound the final output
            # only, and softly.
            x = x + dx
        return x.clamp(-0.5, 1.5)

    def _prox_step(self, x, cond):
        u = F.pixel_unshuffle(x, 2)
        return F.pixel_shuffle(self.prox(torch.cat([u, cond], 1)).float(), 2)


# ----------------------------------------------------------------- config
CFG = dict(
    root      = "/kaggle/input",
    char_dir  = "/kaggle/working/char",     # params.csv, split.json, kernel.npy
    out_dir   = "/kaggle/working/run1",
    model     = "unrolled",                 # "baseline" or "unrolled"
    width     = 32,
    stages    = 4,
    epochs    = 120,
    batch     = 32,          # 3.5 GB at 16 with checkpointing; drop if OOM
    lr        = 4.5e-4,      # sqrt-scaled for batch 32
    real_frac = 0.5,                        # fraction of samples using the real pair
    sg_max    = 0.22,       # real max 0.198, mild widening                       # widened, test is noisier than train
    L_range   = (14.0, 60.0),   # L=14 -> sigma_s 0.267 = real p99
    jitter    = 0.25,                       # relative noise added to conditioning
    aux_w     = 0.05,                       # weight on the estimator loss
    fft_w     = 0.05,                       # weight on the FFT-domain L1 term
    ema       = 0.999,
    seed      = 1234,
    fresh     = False,      # resume after a session kill
    channels_last = True,
    time_budget_h = 11.0,                   # 120 epochs is ~9.7h; Kaggle caps at 12
)
os.makedirs(CFG["out_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"]); random.seed(CFG["seed"])

# ------------------------------------------------------------ file index
MAGIC = b"\x93NUMPY"
def _real(p):
    n = os.path.basename(p)
    if n.startswith(("._", "__", ".")): return False
    try:
        with open(p, "rb") as f: return f.read(6) == MAGIC
    except OSError: return False

def _dirs(root, want):
    out = []
    for dp, dns, _ in os.walk(root):
        for d in dns:
            if want.lower() in d.lower(): out.append(os.path.join(dp, d))
    return sorted(out)

def _is_test(path):
    return any("test" in q for q in path.lower().replace("\\", "/").split("/")[-2:])

GT, LRp = {}, {}
for d in _dirs(CFG["root"], "GT"):
    pre = "test::" if _is_test(d) else ""
    GT.update({pre + os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(d, "*.npy")) if _real(p)})
for d in _dirs(CFG["root"], "NoisyLR"):
    pre = "test::" if _is_test(d) else ""
    LRp.update({pre + os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(d, "*.npy")) if _real(p)})
print(f"indexed {len(GT)} GT, {len(LRp)} NoisyLR")

# ------------------------------------------------- characterisation
# /kaggle/working is wiped on session restart, so never DEPEND on it.
# If the artifacts are missing we recompute them here (~1 min for 3200 pairs)
# and keep them in memory. Saving is best-effort only.
def _down(x, f=2):
    h, w = x.shape[0] // f, x.shape[1] // f
    return x[:h*f, :w*f].reshape(h, f, w, f).mean(axis=(1, 3))

def characterise(paired):
    """Returns (params dict, 4x4 kernel, scene groups)."""
    prm = {}
    AtA = np.zeros((36, 36)); Atb = np.zeros(36); nk = 0
    offs = list(range(-2, 4)); sigs = []
    for i, s_ in enumerate(tqdm(paired, desc="characterising")):
        y = np.squeeze(np.load(LRp[s_])).astype(np.float64)
        x = np.squeeze(np.load(GT[s_])).astype(np.float64)
        if y.ndim != 2 or x.ndim != 2:
            continue
        d = _down(x, max(1, round(x.shape[0] / y.shape[0])))
        h, w = min(d.shape[0], y.shape[0]), min(d.shape[1], y.shape[1])
        d, yy = d[:h, :w], y[:h, :w]
        r = yy - d
        e = np.percentile(d, np.linspace(0, 100, 26)); mu, va = [], []
        for j in range(25):
            m = (d >= e[j]) & (d < e[j+1])
            if m.sum() > 80:
                mu.append(d[m].mean()); va.append(r[m].var())
        if len(mu) >= 4:
            mu = np.array(mu); va = np.array(va)
            A = np.vstack([np.ones_like(mu), mu**2]).T
            c = np.linalg.lstsq(A, va, rcond=None)[0]
            sg_ = math.sqrt(max(c[0], 0.0)); ss_ = math.sqrt(max(c[1], 0.0))
            if 0.02 < ss_ < 0.5:
                prm[s_] = (sg_, ss_)
        if i < 300 and x.shape[0] == 2 * y.shape[0]:
            lo, hi = 2, min(y.shape) - 2
            if hi > lo:
                ii, jj = np.mgrid[lo:hi, lo:hi]
                cols = [x[np.clip(2*ii+u, 0, x.shape[0]-1),
                          np.clip(2*jj+v, 0, x.shape[1]-1)].ravel()
                        for u in offs for v in offs]
                Ak = np.stack(cols, 1); bk = y[ii, jj].ravel()
                dk = _down(x, 2)[lo:hi, lo:hi].ravel()
                wk = 1.0 / (0.03**2 + (0.16*dk)**2)
                AtA += Ak.T @ (Ak * wk[:, None]); Atb += Ak.T @ (wk * bk); nk += 1
        t = _down(x, max(1, x.shape[0] // 12))[:12, :12]
        t = t - t.mean(); nrm = np.linalg.norm(t)
        sigs.append((t / nrm).ravel() if nrm > 0 else t.ravel())

    Kfit = None
    if nk:
        sc = np.trace(AtA) / 36
        Kfull = np.linalg.solve(AtA + 1e-6*sc*np.eye(36), Atb).reshape(6, 6)
        core = Kfull[1:5, 1:5]
        U, S, _ = np.linalg.svd(core)
        rank1 = S[0]**2 / (S**2).sum()
        print(f"  kernel fit: rank-1 energy {rank1:.4%}, cond {np.linalg.cond(AtA):.2e}")
        if rank1 > 0.99:
            Kfit = core / core.sum()
            print("  => separable resampling kernel, trusted")
        else:
            print("  => not separable, falling back to bicubic")

    M = np.stack(sigs); N = len(M); par = list(range(N))
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for i in tqdm(range(N), desc="grouping scenes"):
        for o in np.nonzero(M[i+1:] @ M[i] > 0.92)[0]:
            ra, rb = find(i), find(i+1+o)
            if ra != rb: par[max(ra, rb)] = min(ra, rb)
    g = {}
    for i, s_ in enumerate(paired[:N]): g.setdefault(find(i), []).append(s_)
    return prm, Kfit, list(g.values())


split_path = os.path.join(CFG["char_dir"], "split.json")
paired_all = sorted(set(GT) & set(LRp))
test_ids = sorted(set(LRp) - set(GT))
print(f"paired {len(paired_all)}  test {len(test_ids)}")

PARAMS, Kfit = {}, None
if os.path.exists(split_path):
    sp = json.load(open(split_path))
    train_ids, val_ids = sp["train"], sp["val"]
    print("loaded existing split.json")
    pc = os.path.join(CFG["char_dir"], "params.csv")
    if os.path.exists(pc):
        import csv
        for r in csv.DictReader(open(pc)):
            try:
                a, b = float(r["sigma_g"]), float(r["sigma_s"])
                if np.isfinite(a) and 0.02 < b < 0.5: PARAMS[r["stem"]] = (a, b)
            except (ValueError, KeyError): pass
    kpp = os.path.join(CFG["char_dir"], "kernel.npy")
    if os.path.exists(kpp):
        kk = np.load(kpp).astype(np.float64)
        if kk.shape[-1] == 6: kk = kk[1:5, 1:5]
        Kfit = kk / kk.sum()
else:
    print("no split.json - recomputing characterisation from scratch")
    PARAMS, Kfit, groups = characterise(paired_all)
    rng = np.random.default_rng(CFG["seed"])
    target = max(1, int(0.12 * len(paired_all))); val, taken = [], 0
    for gi in rng.permutation(len(groups)):
        gg = groups[gi]
        if taken >= target: break
        if taken + len(gg) > len(paired_all) - 1: continue
        val += gg; taken += len(gg)
    vs = set(val)
    if not vs or len(vs) >= len(paired_all):
        print(f"  !! grouping degenerate ({len(groups)} groups for "
              f"{len(paired_all)} images) - using a random split instead. "
              f"Validation will be slightly optimistic.")
        perm = rng.permutation(len(paired_all))
        vs = {paired_all[i] for i in perm[:target]}
    train_ids = [s_ for s_ in paired_all if s_ not in vs]
    val_ids = sorted(vs)
    assert train_ids and val_ids, "split still empty - stop and tell me"
    print(f"  {len(paired_all)} images -> {len(groups)} scene groups")
    try:
        os.makedirs(CFG["char_dir"], exist_ok=True)
        json.dump(dict(train=train_ids, val=val_ids, test=test_ids),
                  open(split_path, "w"))
        if Kfit is not None:
            np.save(os.path.join(CFG["char_dir"], "kernel.npy"), Kfit)
        print("  saved artifacts (best effort)")
    except OSError as e:
        print(f"  could not save artifacts ({e}) - continuing in memory")

print(f"train {len(train_ids)}  val {len(val_ids)}  test {len(test_ids)}")
if len(PARAMS) < 0.5 * len(paired_all):
    # split.json was present but params.csv was not, so the real-pair branch
    # would silently never fire and training would be 100% synthetic.
    print(f"only {len(PARAMS)} fitted parameters - refitting them now")
    PARAMS2, _, _ = characterise(paired_all)
    PARAMS.update(PARAMS2)
    try:
        with open(os.path.join(CFG["char_dir"], "params.csv"), "w") as fh:
            fh.write("stem,sigma_g,sigma_s\n")
            for k_, (a_, b_) in PARAMS.items():
                fh.write(f"{k_},{a_},{b_}\n")
    except OSError:
        pass
print(f"usable fitted parameters for {len(PARAMS)} images "
      f"({len(PARAMS)/max(len(paired_all),1):.0%} of pairs)")
assert len(PARAMS) > 0.5 * len(paired_all), \
    "still missing most fitted parameters - stop and tell me"

if Kfit is not None:
    K = Kfit
    print(f"using fitted kernel, 4x4 core, sum {K.sum():.4f}")
else:
    k1 = np.array([-0.0625, 0.5625, 0.5625, -0.0625])
    K = np.outer(k1, k1); print("using bicubic kernel")
K_t = torch.tensor(K, dtype=torch.float32).view(1, 1, *K.shape)


def degrade(x_np, sg, L, rng):
    """Apply the measured forward model to one 256x256 GT image."""
    x = torch.from_numpy(x_np).float().view(1, 1, *x_np.shape)
    p = K_t.shape[-1] // 2
    d = F.conv2d(F.pad(x, (p - 1, p, p - 1, p), mode="reflect"), K_t, stride=2)
    d = d.numpy()[0, 0]
    u = rng.gamma(L, 1.0 / L, size=d.shape)
    return (d * u + rng.normal(0.0, sg, size=d.shape)).astype(np.float32)


def sample_sg(rng):
    """Zero-heavy like the real distribution, with a uniform tail for coverage."""
    if rng.random() < 0.2:
        return rng.uniform(0.0, CFG["sg_max"])
    return min(abs(rng.normal(0.0, 0.045)), CFG["sg_max"])


class Pairs(Dataset):
    def __init__(self, ids, train=True):
        self.ids, self.train = ids, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        stem = self.ids[i]
        rng = np.random.default_rng(None if self.train else (hash(stem) % 2**31))
        x = np.squeeze(np.load(GT[stem])).astype(np.float32)

        use_real = (not self.train) or (rng.random() < CFG["real_frac"])
        if use_real and stem in PARAMS:
            y = np.squeeze(np.load(LRp[stem])).astype(np.float32)
            sg, ss = PARAMS[stem]
        else:
            sg = sample_sg(rng)
            L = rng.uniform(*CFG["L_range"])
            ss = 1.0 / math.sqrt(L)
            y = degrade(x, sg, L, rng)

        if self.train:                                    # D4 augmentation
            k = rng.integers(4)
            if k: x, y = np.rot90(x, k).copy(), np.rot90(y, k).copy()
            if rng.random() < 0.5: x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
            if rng.random() < 0.5: x, y = x[::-1].copy(), y[::-1].copy()

        return (torch.from_numpy(y)[None], torch.from_numpy(x)[None],
                torch.tensor([sg], dtype=torch.float32),
                torch.tensor([ss], dtype=torch.float32))


# ------------------------------------------------------------- estimator
class Estimator(nn.Module):
    """Predicts (sigma_g, 1/sqrt(L)) from the noisy image. Small on purpose."""
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
        h = self.body(y).flatten(1)
        o = self.head(h)
        sg = F.softplus(o[:, :1]) * 0.1          # keeps it in a sane range
        ss = F.softplus(o[:, 1:]) * 0.1 + 0.05
        return sg.view(-1, 1, 1, 1), ss.view(-1, 1, 1, 1)


# ---------------------------------------------------------------- losses
def charbonnier(a, b, eps=1e-3):
    return torch.sqrt((a - b) ** 2 + eps ** 2).mean()

def fft_l1(a, b):
    """Ortho-normalised, so this term is on the same scale as the pixel loss.
    An unnormalised rfft2 scales by N: the DC bin alone reaches ~32,000 at
    256x256, which makes the term ~285x the pixel L1 and lets it dominate
    training completely."""
    with torch.amp.autocast(device_type=a.device.type, enabled=False):
        fa = torch.fft.rfft2(a.float(), norm="ortho")
        fb = torch.fft.rfft2(b.float(), norm="ortho")
        return (fa - fb).abs().mean()


def psnr(pred, gt):
    pred = pred.clamp(0, 1).float(); gt = gt.float()
    mse = ((pred - gt) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def minmax_rescale(x):
    """GT is per-image min-max normalised in 3200/3200 images - free gain."""
    f = x.flatten(1)
    lo = f.min(1)[0].view(-1, 1, 1, 1); hi = f.max(1)[0].view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo).clamp_min(1e-6)


# ------------------------------------------------------------------ setup
model = (Unrolled(CFG["width"], CFG["stages"], kernel=torch.tensor(K).float())
         if CFG["model"] == "unrolled" else Baseline(CFG["width"])).to(dev)
est = Estimator().to(dev)
if CFG["channels_last"]:
    model = model.to(memory_format=torch.channels_last)

ema_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
params = list(model.parameters()) + list(est.parameters())
opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=1e-4, betas=(0.9, 0.9))
scaler = torch.amp.GradScaler()

tr_dl = DataLoader(Pairs(train_ids, True), batch_size=CFG["batch"], shuffle=True,
                   num_workers=2, pin_memory=True, drop_last=True,
                   persistent_workers=True)
va_dl = DataLoader(Pairs(val_ids, False), batch_size=CFG["batch"], shuffle=False,
                   num_workers=2, pin_memory=True)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=CFG["lr"], total_steps=CFG["epochs"] * len(tr_dl), pct_start=0.05)

ck = os.path.join(CFG["out_dir"], "ckpt.pt")
start_ep, best = 0, -1e9

def _all_finite(sd):
    return all(torch.isfinite(v).all().item() for v in sd.values()
               if torch.is_tensor(v) and v.dtype.is_floating_point)

if CFG["fresh"] and os.path.exists(ck):
    os.remove(ck); print("CFG['fresh'] is set - deleted the old checkpoint")
if os.path.exists(ck):
    st = torch.load(ck, map_location=dev)
    # A checkpoint saved from a diverged run contains NaN weights. Reloading
    # it guarantees every subsequent batch is non-finite and the run can
    # never recover, so refuse it outright.
    bad = [k for k in ("model", "est", "ema") if not _all_finite(st[k])]
    if bad:
        print(f"REFUSING to resume: {', '.join(bad)} contain non-finite "
              f"values (the earlier run diverged). Starting fresh.")
        os.remove(ck)
    else:
        model.load_state_dict(st["model"]); est.load_state_dict(st["est"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); ema_model = st["ema"]
        start_ep, best = st["epoch"] + 1, st["best"]
        print(f"resumed from epoch {start_ep}, best val PSNR "
              f"{'none yet' if best < -1e8 else f'{best:.3f}'}")

print(f"\nmodel {CFG['model']} w{CFG['width']}  "
      f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params  "
      f"+ estimator {sum(p.numel() for p in est.parameters())/1e6:.2f}M")

def quick_val(m, e, n_batches=6, use_est=False):
    m.eval(); e.eval()
    tot = n = 0
    with torch.no_grad():
        for i, (y, gt, sg, ss) in enumerate(va_dl):
            if i >= n_batches: break
            y, gt = y.to(dev), gt.to(dev)
            sg = sg.to(dev).view(-1, 1, 1, 1); ss = ss.to(dev).view(-1, 1, 1, 1)
            with torch.amp.autocast(device_type=dev):
                if use_est:
                    sg, ss = e(y)
                out = m(y, sg, ss)
            tot += psnr(out.float(), gt) * y.size(0); n += y.size(0)
    m.train(); e.train()
    return tot / max(n, 1)

if start_ep == 0:
    init_p = quick_val(model, est)
    bic = 0.0; nb = 0
    with torch.no_grad():
        for i, (y, gt, _, _) in enumerate(va_dl):
            if i >= 6: break
            up = F.interpolate(y.to(dev).float(), scale_factor=2,
                               mode="bicubic", align_corners=False)
            bic += psnr(up, gt.to(dev)) * y.size(0); nb += y.size(0)
    bic /= max(nb, 1)
    print(f"\nsanity before any training:")
    print(f"  plain bicubic upsample of the noisy input : {bic:.3f} dB")
    print(f"  untrained model (zero-init tail)          : {init_p:.3f} dB")
    if init_p < bic - 1.0:
        raise SystemExit(
            f"The untrained model is {bic - init_p:.1f} dB WORSE than bicubic. "
            f"With a zero-init tail it should match or beat it, so the "
            f"data-fidelity step is wrong. Stop and report these two numbers.")
    print(f"  -> wiring looks right, training should climb from here\n")

t_start = time.time()
for ep in range(start_ep, CFG["epochs"]):
    model.train(); est.train()
    run = est_err = 0.0; n_bad = 0; n_ok = 0; n_oom = 0
    run_pix = run_fft = 0.0
    bar = tqdm(tr_dl, desc=f"epoch {ep+1}/{CFG['epochs']}", leave=False)
    for y, gt, sg, ss in bar:
        y, gt = y.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
        sg = sg.to(dev).view(-1, 1, 1, 1); ss = ss.to(dev).view(-1, 1, 1, 1)
        if CFG["channels_last"]:
            y = y.contiguous(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        try:
            with torch.amp.autocast(device_type=dev):
                psg, pss = est(y)
                aux = F.l1_loss(psg, sg) + F.l1_loss(pss, ss)
                # jitter the TRUE parameters so the model tolerates estimator error
                j = CFG["jitter"]
                sgj = (sg * (1 + j * torch.randn_like(sg))).clamp(0, CFG["sg_max"])
                ssj = (ss * (1 + j * torch.randn_like(ss))).clamp(0.05, 0.35)
                out = model(y, sgj, ssj)
                l_pix = charbonnier(out, gt)
                l_fft = fft_l1(out, gt)
                loss = l_pix + CFG["fft_w"] * l_fft + CFG["aux_w"] * aux
        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            opt.zero_grad(set_to_none=True)
            del y, gt
            torch.cuda.empty_cache(); sched.step()
            if n_oom <= 3:
                print(f"\n  OOM on a batch (#{n_oom}) - skipped and cache "
                      f"cleared. If this repeats, lower CFG['batch'].")
            if n_oom > 20:
                raise SystemExit("persistent OOM - lower CFG['batch'] to 8")
            continue
        if not torch.isfinite(loss):
            n_bad += 1
            opt.zero_grad(set_to_none=True); sched.step()
            if n_bad <= 3:
                print(f"\n  non-finite loss on a batch (#{n_bad}) - skipped. "
                      f"y[{y.min():.3f},{y.max():.3f}] "
                      f"sg[{sg.min():.4f},{sg.max():.4f}] "
                      f"ss[{ss.min():.4f},{ss.max():.4f}]")
            if n_bad > 50:
                raise SystemExit("too many non-finite batches - stop and debug")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            n_bad += 1
        # scaler.step() already skips the optimiser internally when unscale_
        # found infs, and update() MUST be called after unscale_ or the next
        # unscale_ raises "already been called on this optimizer". Skipping
        # update() on a bad batch is what killed the epoch-92 run.
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    if torch.isfinite(v).all():
                        ema_model[k].mul_(CFG["ema"]).add_(v.detach(),
                                                          alpha=1 - CFG["ema"])
                else:
                    ema_model[k].copy_(v)
        run += loss.item(); est_err += aux.item(); n_ok += 1
        run_pix += l_pix.item(); run_fft += l_fft.item()
        bar.set_postfix(loss=f"{run/max(n_ok,1):.4f}",
                        pix=f"{run_pix/max(n_ok,1):.4f}",
                        fft=f"{CFG['fft_w']*run_fft/max(n_ok,1):.4f}",
                        est=f"{est_err/max(n_ok,1):.4f}", bad=n_bad, oom=n_oom)

    # ---- validation, using the ESTIMATOR (as at test time), with EMA weights
    if dev == "cuda":
        torch.cuda.empty_cache()
    backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_model)
    model.eval(); est.eval()
    tot = tot_r = n = 0
    with torch.no_grad():
        for y, gt, sg, ss in va_dl:
            y, gt = y.to(dev), gt.to(dev)
            if CFG["channels_last"]:
                y = y.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=dev):
                psg, pss = est(y)
                out = model(y, psg, pss)
            tot += psnr(out, gt) * y.size(0)
            tot_r += psnr(minmax_rescale(out.float()), gt) * y.size(0)
            n += y.size(0)
    vp, vpr = tot / n, tot_r / n
    model.load_state_dict(backup)
    del backup
    if dev == "cuda":
        torch.cuda.empty_cache()

    el = (time.time() - t_start) / 3600
    print(f"epoch {ep+1:3d}  train {run/max(n_ok,1):.4f} "
          f"(pix {run_pix/max(n_ok,1):.4f}, "
          f"fft {CFG['fft_w']*run_fft/max(n_ok,1):.4f})  "
          f"[{n_bad} bad, {n_oom} oom]  "
          f"val PSNR {vp:.3f}  (rescale variant {vpr:.3f}, not used)  "
          f"({el:.2f}h elapsed)")

    torch.save(dict(model=model.state_dict(), est=est.state_dict(),
                    opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict(), ema=ema_model, epoch=ep,
                    best=max(best, vp), cfg=CFG), ck)
    # Select on the PLAIN PSNR. The min-max rescale was measured to HURT by
    # ~4 dB: rescaling by a prediction's own min/max is driven by outlier
    # pixels, so one stray value shifts the whole image. The GT really is
    # min-max normalised, but that constrains clean GT, not a noisy estimate.
    if vp > best:
        best = vp
        torch.save(dict(ema=ema_model, est=est.state_dict(), cfg=CFG,
                        val_psnr=vp), os.path.join(CFG["out_dir"], "best.pt"))
        print(f"    new best {best:.3f} dB -> best.pt")
    if el > CFG["time_budget_h"]:
        print(f"    time budget reached, stopping cleanly. Re-run to resume.")
        break

print(f"\nbest val PSNR (EMA + rescale): {best:.3f} dB")
print(f"checkpoints in {CFG['out_dir']}")



print("\n" + "=" * 60)
print("test inference")

best_path = os.path.join(CFG["out_dir"], "best.pt")
if not os.path.exists(best_path):
    print("no best.pt - skipping inference (train first)")
else:
    st = torch.load(best_path, map_location=dev)
    model.load_state_dict(st["ema"]); est.load_state_dict(st["est"])
    model.eval(); est.eval()
    print(f"loaded best.pt (val PSNR {st.get('val_psnr', float('nan')):.3f} dB)")

    def d4(t, k):
        """The 8 dihedral transforms. k in 0..7."""
        if k >= 4:
            t = torch.flip(t, [-1])
        return torch.rot90(t, k % 4, dims=(-2, -1))

    def d4_inv(t, k):
        t = torch.rot90(t, -(k % 4), dims=(-2, -1))
        if k >= 4:
            t = torch.flip(t, [-1])
        return t

    pred_dir = os.path.join(CFG["out_dir"], "test_pred")
    os.makedirs(pred_dir, exist_ok=True)
    written = []
    with torch.no_grad():
        for stem in tqdm(test_ids, desc="predicting"):
            y = np.squeeze(np.load(LRp[stem])).astype(np.float32)
            yt = torch.from_numpy(y)[None, None].to(dev)
            # the estimator sees the untransformed image once
            with torch.amp.autocast(device_type=dev):
                psg, pss = est(yt)
            acc = torch.zeros(1, 1, y.shape[0] * 2, y.shape[1] * 2,
                              device=dev, dtype=torch.float32)
            for k in range(8):
                with torch.amp.autocast(device_type=dev):
                    o = model(d4(yt, k), psg, pss)
                acc += d4_inv(o.float(), k)
            out = acc / 8.0
            # plain clamp. minmax_rescale measured 4 dB WORSE on validation.
            out = out.clamp(0.0, 1.0)
            arr = out[0, 0].cpu().numpy().astype(np.float32)
            name = stem.replace("test::", "") + ".npy"
            np.save(os.path.join(pred_dir, name), arr)
            written.append(os.path.join(pred_dir, name))

    zpath = os.path.join(CFG["out_dir"], "predictions.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in written:
            z.write(f, os.path.basename(f))
    a = np.load(written[0])
    print(f"\nwrote {len(written)} predictions to {pred_dir}")
    print(f"  shape {a.shape}  dtype {a.dtype}  range [{a.min():.4f}, {a.max():.4f}]")
    print(f"  zipped -> {zpath}  ({os.path.getsize(zpath)/1e6:.1f} MB)")
    print("\nCHECK the organisers' required submission format before uploading:")
    print("  filenames, dtype, and whether they want .npy or something else.")
