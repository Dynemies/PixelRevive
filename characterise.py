import os
import re
import json
import glob
import argparse
import numpy as np

try:
    from tqdm import tqdm
except ImportError:                                    # Kaggle always has it
    def tqdm(it, **kw):
        total = kw.get("total", None)
        desc = kw.get("desc", "")
        n = 0
        for item in it:
            n += 1
            if n % 25 == 0:
                print(f"  {desc}: {n}{'/' + str(total) if total else ''}",
                      flush=True)
            yield item

NPY_MAGIC = b"\x93NUMPY"
KERNEL_SUPPORT = range(-2, 4)          # 6x6 support around each 2x2 block
PRIOR_SG = 0.03                        # rough weights for the kernel fit
PRIOR_SS = 0.16

def is_real_npy(path):
    """Reject macOS '._' sidecars and anything that isn't a real .npy."""
    name = os.path.basename(path)
    if name.startswith("._") or name.startswith("__"):
        return False
    try:
        with open(path, "rb") as fh:
            return fh.read(6) == NPY_MAGIC
    except OSError:
        return False


def find_dirs(root, wanted):
    """Every directory under root whose basename matches `wanted`."""
    hits = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if wanted.lower() in d.lower():
                hits.append(os.path.join(dirpath, d))
    return sorted(hits)


def index_folder(folder, prefix=""):
    """stem -> path for every genuine .npy in folder."""
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        if is_real_npy(p):
            out[prefix + os.path.splitext(os.path.basename(p))[0]] = p
    return out


def is_test_dir(path):
    """Test folders get namespaced stems: their files are usually numbered
    from 000000 too, and would otherwise clobber the train entries."""
    return "test" in os.path.basename(path).lower() or \
           "test" in os.path.dirname(path).lower().split(os.sep)[-1]


def collect(root):
    """Locate the paired train set and any unpaired (test) NoisyLR folder."""
    gt_dirs = find_dirs(root, "GT")
    lr_dirs = find_dirs(root, "NoisyLR")
    gt, lr = {}, {}
    for d in gt_dirs:
        gt.update(index_folder(d, "test::" if is_test_dir(d) else ""))
    n_test_files = 0
    for d in lr_dirs:
        pre = "test::" if is_test_dir(d) else ""
        got = index_folder(d, pre)
        if pre:
            n_test_files += len(got)
        lr.update(got)

    paired = sorted(set(gt) & set(lr))
    unpaired = sorted(set(lr) - set(gt))
    print(f"GT folders found      : {len(gt_dirs)}  ({len(gt)} arrays)")
    for d in gt_dirs:
        print(f"    {d}")
    print(f"NoisyLR folders found : {len(lr_dirs)}  ({len(lr)} arrays)")
    for d in lr_dirs:
        print(f"    {d}{'   [test]' if is_test_dir(d) else ''}")
    print(f"matched pairs         : {len(paired)}")
    print(f"NoisyLR without GT    : {len(unpaired)}  (treated as test; "
          f"{n_test_files} came from a test-named folder)")
    if not paired:
        raise SystemExit(
            "No GT/NoisyLR pairs found. Check --root, and make sure the "
            "macOS '._' sidecar files were deleted before upload."
        )
    return gt, lr, paired, unpaired


def getarr(idx, stem):
    """Index values are either file paths or in-memory/memmapped arrays."""
    v = idx[stem]
    if isinstance(v, np.ndarray):
        return np.asarray(v)
    return np.load(v)


def collect_stacks(stacks_dir):
    """Same interface as collect(), but backed by pack_data.py output."""
    with open(os.path.join(stacks_dir, "stems.json")) as fh:
        meta = json.load(fh)
    paired, test = meta["paired"], meta.get("test", [])
    gt_arr = np.load(os.path.join(stacks_dir, "gt_train.npy"), mmap_mode="r")
    lr_arr = np.load(os.path.join(stacks_dir, "lr_train.npy"), mmap_mode="r")
    gt = {s: gt_arr[i] for i, s in enumerate(paired)}
    lr = {s: lr_arr[i] for i, s in enumerate(paired)}
    tpath = os.path.join(stacks_dir, "lr_test.npy")
    if test and os.path.exists(tpath):
        te = np.load(tpath, mmap_mode="r")
        for i, s in enumerate(test):
            lr[s] = te[i]
    print(f"loaded packed stacks: {len(paired)} pairs, {len(test)} test")
    return gt, lr, paired, test


# ----------------------------------------------------------------------
# noise-model fitting
# ----------------------------------------------------------------------
def block_downsample(x, factor):
    """Non-overlapping block mean — the reference operator for fitting."""
    h, w = x.shape
    hh, ww = h // factor, w // factor
    return x[:hh * factor, :ww * factor].reshape(
        hh, factor, ww, factor).mean(axis=(1, 3))


def fit_noise(y, x, nbins=25):
    """
    Fit  Var(y - d) = sigma_g^2 + sigma_s^2 * d^2   where d = downsample(x).

    Returns (sigma_g, sigma_s, R2). Binning by intensity before regressing
    keeps a handful of bright outliers from dominating the fit.
    """
    factor = max(1, round(x.shape[0] / y.shape[0]))
    d = block_downsample(x, factor)
    h = min(d.shape[0], y.shape[0])
    w = min(d.shape[1], y.shape[1])
    d, yy = d[:h, :w], y[:h, :w]
    r = yy - d

    edges = np.percentile(d, np.linspace(0, 100, nbins + 1))
    mu, var = [], []
    for i in range(nbins):
        m = (d >= edges[i]) & (d < edges[i + 1])
        if m.sum() > 80:
            mu.append(d[m].mean())
            var.append(r[m].var())
    if len(mu) < 4:
        return np.nan, np.nan, np.nan, factor
    mu = np.array(mu)
    var = np.array(var)
    A = np.vstack([np.ones_like(mu), mu ** 2]).T
    coef, *_ = np.linalg.lstsq(A, var, rcond=None)
    pred = A @ coef
    denom = ((var - var.mean()) ** 2).sum()
    r2 = 1 - ((var - pred) ** 2).sum() / denom if denom > 0 else np.nan
    return (np.sqrt(max(coef[0], 0.0)),
            np.sqrt(max(coef[1], 0.0)), r2, factor)


def gamma_shape(y, x, sigma_g, sigma_s):
    """
    Method-of-moments Gamma shape L from bright pixels only, where the
    additive term contributes little. Returns nan if too few qualify.
    """
    factor = max(1, round(x.shape[0] / y.shape[0]))
    d = block_downsample(x, factor)
    h = min(d.shape[0], y.shape[0])
    w = min(d.shape[1], y.shape[1])
    d, yy = d[:h, :w], y[:h, :w]

    thr = max(0.30, 3.0 * sigma_g / max(sigma_s, 1e-6))
    m = d > thr
    if m.sum() < 1500:
        return np.nan, np.nan
    u = (yy[m] / d[m])
    u = u[(u > 0) & np.isfinite(u)]
    if u.size < 1500:
        return np.nan, np.nan
    var = u.var()
    return (1.0 / var if var > 0 else np.nan), float(np.mean(u))


# ----------------------------------------------------------------------
# global kernel recovery via accumulated normal equations
# ----------------------------------------------------------------------
class KernelAccumulator:
    """
    Solves for K in  y[i,j] = sum_{u,v} K[u,v] * x[2i+u, 2j+v].

    Accumulates A^T W A and A^T W b across images so memory stays flat
    regardless of dataset size.
    """

    def __init__(self, offsets=KERNEL_SUPPORT):
        self.offs = list(offsets)
        p = len(self.offs)
        self.p = p
        self.AtA = np.zeros((p * p, p * p))
        self.Atb = np.zeros(p * p)
        self.n = 0

    def add(self, y, x, factor):
        if factor != 2:
            return
        h, w = y.shape
        lo, hi = 2, min(h, w) - 2
        if hi <= lo:
            return
        ii, jj = np.mgrid[lo:hi, lo:hi]
        cols = []
        for u in self.offs:
            for v in self.offs:
                yi = np.clip(2 * ii + u, 0, x.shape[0] - 1)
                xj = np.clip(2 * jj + v, 0, x.shape[1] - 1)
                cols.append(x[yi, xj].ravel())
        A = np.stack(cols, 1)
        b = y[ii, jj].ravel()

        d = block_downsample(x, 2)
        d = d[lo:hi, lo:hi].ravel()
        wt = 1.0 / (PRIOR_SG ** 2 + (PRIOR_SS * d) ** 2)

        self.AtA += A.T @ (A * wt[:, None])
        self.Atb += A.T @ (wt * b)
        self.n += 1

    def solve(self, shrink=0.0):
        """
        Least-squares solve with optional shrinkage toward the box average.

        Also returns the condition number: image content that is mostly
        smooth makes neighbouring GT pixels near-identical, which leaves
        this system close to singular and the recovered kernel meaningless.
        """
        if self.n == 0:
            return None, np.inf
        p = self.p
        scale = np.trace(self.AtA) / (p * p)
        try:
            cond = float(np.linalg.cond(self.AtA))
        except np.linalg.LinAlgError:
            cond = np.inf

        box = np.zeros((p, p))
        c = p // 2 - 1
        box[c:c + 2, c:c + 2] = 0.25
        prior = box.ravel()

        lam = (1e-6 + shrink) * scale
        A = self.AtA + lam * np.eye(p * p)
        b = self.Atb + lam * prior
        K = np.linalg.solve(A, b)
        return K.reshape(p, p), cond


def thumbnail(x, size=12):
    """Tiny normalised thumbnail — cheap scene signature."""
    f = max(1, x.shape[0] // size)
    t = block_downsample(x.astype(np.float64), f)[:size, :size]
    t = t - t.mean()
    n = np.linalg.norm(t)
    return (t / n).ravel() if n > 0 else t.ravel()


def group_scenes(sigs, stems, thresh=0.80):
    """
    Union-find over cosine similarity of thumbnails. Overlapping crops of
    one scene land in one group, so they can't straddle the train/val line.
    """
    n = len(stems)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    M = np.stack(sigs)
    for i in tqdm(range(n), desc="grouping scenes", total=n):
        sims = M[i + 1:] @ M[i]
        for off in np.nonzero(sims > thresh)[0]:
            union(i, i + 1 + off)

    groups = {}
    for i, stem in enumerate(stems):
        groups.setdefault(find(i), []).append(stem)
    return list(groups.values())


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="dataset root containing GT / NoisyLR folders")
    ap.add_argument("--stacks", help="directory of pack_data.py output "
                                     "(use instead of --root)")
    ap.add_argument("--out", default="./characterisation")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kernel-max", type=int, default=400,
                    help="cap images used for the kernel fit (it converges "
                         "long before the whole set is consumed)")
    args, _unknown = ap.parse_known_args(argv)
    if not args.root and not args.stacks:
        raise SystemExit("Pass either --root or --stacks.")
    os.makedirs(args.out, exist_ok=True)

    if args.stacks:
        gt, lr, paired, unpaired = collect_stacks(args.stacks)
    else:
        gt, lr, paired, unpaired = collect(args.root)

    rows = []
    acc = KernelAccumulator()
    sigs, stems = [], []
    gt_min_exact = gt_max_exact = 0

    print("\n--- fitting the noise model on every pair ---", flush=True)
    for idx, stem in enumerate(tqdm(paired, desc="pairs", total=len(paired))):
        try:
            y = getarr(lr, stem).astype(np.float64)
            x = getarr(gt, stem).astype(np.float64)
        except Exception as e:
            print(f"  skipping {stem}: {e}")
            continue
        y = np.squeeze(y)
        x = np.squeeze(x)
        if y.ndim != 2 or x.ndim != 2:
            continue

        sg, ss, r2, factor = fit_noise(y, x)
        L, umean = (np.nan, np.nan)
        if np.isfinite(sg) and np.isfinite(ss):
            L, umean = gamma_shape(y, x, sg, ss)

        xmin, xmax = float(x.min()), float(x.max())
        gt_min_exact += abs(xmin) < 1e-6
        gt_max_exact += abs(xmax - 1.0) < 1e-6

        rows.append(dict(stem=stem, scale=factor, sigma_g=sg, sigma_s=ss,
                         fit_r2=r2, gamma_L=L, speckle_mean=umean,
                         gt_min=xmin, gt_max=xmax,
                         lr_min=float(y.min()), lr_max=float(y.max()),
                         lr_frac_above_1=float((y > 1).mean()),
                         lr_frac_below_0=float((y < 0).mean())))

        if idx < args.kernel_max:
            acc.add(y, x, factor)
        sigs.append(thumbnail(x))
        stems.append(stem)

    if not rows:
        raise SystemExit("Every pair failed to load — check the data.")

    # ---- params.csv
    keys = list(rows[0].keys())
    csv_path = os.path.join(args.out, "params.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    def col(name):
        v = np.array([r[name] for r in rows], float)
        return v[np.isfinite(v)]

    sg_v, ss_v, L_v, r2_v = col("sigma_g"), col("sigma_s"), col("gamma_L"), col("fit_r2")
    n = len(rows)

    print(f"\n=== noise parameters over {n} pairs ===")
    for nm, v in (("sigma_g", sg_v), ("sigma_s", ss_v), ("gamma_L", L_v)):
        if v.size:
            q = np.percentile(v, [0, 1, 50, 99, 100])
            print(f"  {nm:9s} min {q[0]:.4f}  p1 {q[1]:.4f}  median {q[2]:.4f}"
                  f"  p99 {q[3]:.4f}  max {q[4]:.4f}")
    if r2_v.size:
        print(f"  fit R^2   median {np.median(r2_v):.3f}, "
              f"fraction above 0.9: {(r2_v > 0.9).mean():.2%}")
    print("  ^ SAMPLE FROM THESE RANGES (widened ~20%) when synthesising")

    print(f"\n=== GT normalisation check ===")
    print(f"  min exactly 0 : {gt_min_exact}/{n}  ({gt_min_exact/n:.1%})")
    print(f"  max exactly 1 : {gt_max_exact}/{n}  ({gt_max_exact/n:.1%})")
    if min(gt_min_exact, gt_max_exact) / n > 0.98:
        print("  => per-image min-max normalisation CONFIRMED; rescaling the "
              "prediction to [0,1] is a free gain")
    else:
        print("  => NOT universal. Do NOT rescale predictions to [0,1].")

    # ---- kernel
    K, cond = acc.solve()
    if K is not None:
        c = len(acc.offs) // 2 - 1
        centre = K[c:c + 2, c:c + 2]
        outside = np.abs(K).sum() - np.abs(centre).sum()
        print(f"\n=== fitted 6x6 stride-2 kernel ({acc.n} images) ===")
        np.set_printoptions(precision=4, suppress=True, linewidth=140)
        print(K)
        print(f"  sum {K.sum():.4f} | central 2x2 sum {centre.sum():.4f} | "
              f"|weight| outside centre {outside:.4f}")
        print(f"  condition number of the normal equations: {cond:.3e}")

        trust = cond < 1e6 and outside < 0.35
        if trust:
            np.save(os.path.join(args.out, "kernel.npy"), K)
            print("  => WELL CONDITIONED. This kernel is trustworthy; use it "
                  "as the forward operator.")
            if outside > 0.05:
                print("     Non-negligible side lobes: genuinely not a plain "
                      "box average.")
            else:
                print("     Side lobes negligible: it IS effectively a 2x2 "
                      "box average.")
        else:
            Kb, _ = acc.solve(shrink=1.0)
            np.save(os.path.join(args.out, "kernel.npy"), Kb)
            print("  => ILL CONDITIONED, or too much stray mass. Smooth image "
                  "content makes neighbouring GT pixels nearly identical, so "
                  "the free-form fit is not identifiable.")
            print("     DO NOT read the numbers above as the true kernel.")
            print("     Saved a box-shrunk kernel instead; treat the operator "
                  "as a 2x2 box average and move on. It is not worth more "
                  "time than that.")

    # ---- scene-grouped split
    print()
    groups = group_scenes(sigs, stems)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(groups))
    target = int(round(args.val_frac * len(stems)))
    val, taken = [], 0
    for gi in order:
        if taken >= target:
            break
        val.extend(groups[gi])
        taken += len(groups[gi])
    val_set = set(val)
    train = [s for s in stems if s not in val_set]
    split = dict(train=train, val=sorted(val_set), test=unpaired,
                 n_groups=len(groups), seed=args.seed,
                 note="grouped by scene similarity; crops of one scene never straddle the split")
    with open(os.path.join(args.out, "split.json"), "w") as fh:
        json.dump(split, fh, indent=1)

    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"\n=== scene-grouped split ===")
    print(f"  {len(stems)} images fall into {len(groups)} scene groups")
    print(f"  largest groups: {sizes[:8]}")
    print(f"  train {len(train)} | val {len(val_set)} | test {len(unpaired)}")
    if len(groups) < len(stems):
        print(f"  {len(stems) - len(groups)} images share a scene with another "
              "— a random split WOULD have leaked")

    # ---- train vs test distribution comparison
    if unpaired:
        print("\n--- comparing test NoisyLR against train NoisyLR ---", flush=True)

        def lr_stats(stem_list, label, cap=600):
            vals = []
            for s in tqdm(stem_list[:cap], desc=label, total=min(cap, len(stem_list))):
                try:
                    a = np.squeeze(getarr(lr, s).astype(np.float64))
                except Exception:
                    continue
                vals.append((a.mean(), a.std(), (a > 1).mean(), a.max()))
            return np.array(vals) if vals else None

        tr = lr_stats(train, "train LR")
        te = lr_stats(unpaired, "test LR")
        if tr is not None and te is not None:
            print("\n  statistic        train median    test median")
            for i, nm in enumerate(("mean", "std", "frac>1", "max")):
                print(f"  {nm:14s}   {np.median(tr[:, i]):10.4f}"
                      f"      {np.median(te[:, i]):10.4f}")
            drift = abs(np.median(tr[:, 1]) - np.median(te[:, 1]))
            if drift > 0.02:
                print("  => test noise level DIFFERS from train. Widen the "
                      "synthesis ranges accordingly.")
            else:
                print("  => test and train look like the same distribution.")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(args.out, 'split.json')}")
    if K is not None:
        print(f"Wrote {os.path.join(args.out, 'kernel.npy')}")


def run(root=None, stacks=None, out="./characterisation",
        val_frac=0.12, seed=1234, kernel_max=400):
    """Notebook-friendly entry point — no command line needed.

        import characterise
        characterise.run(root="/kaggle/input", out="/kaggle/working/char")
    """
    argv = []
    if root:
        argv += ["--root", str(root)]
    if stacks:
        argv += ["--stacks", str(stacks)]
    argv += ["--out", str(out), "--val-frac", str(val_frac),
             "--seed", str(seed), "--kernel-max", str(kernel_max)]
    return main(argv)


if __name__ == "__main__":
    main()
