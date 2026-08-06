"""Double-flat init: orthogonal W_E + isometric attention heads.

The ticket variable T_k = sum_h ||W_O^h W_V^h W_E|_k||^2 factorizes as
E_k (W_E spectrum) x G_k (frame-to-OV alignment). orthWE froze E_k and the
lottery ran on G_k (T50-T54). This experiment freezes BOTH: with
embed_init='orthogonal' and attn_init='isometric', T_k is identically flat
by construction (verified numerically per seed below, before training).

PRE-REGISTERED PREDICTIONS (printed before any training):
  P-DF1 (primary): every epoch-0 committee readout is at chance (AUC ~0.5)
        on these runs — raw W_E spectrum, closed-form T_k, forward-pass MLP
        readout, forward-pass align readout. Test: per-readout mean AUC over
        the 8 seeds, t vs 0.5, all ns and inside [0.40, 0.60].
  P-DF2: the runs still grok (acc 1.0) and form normal-looking committees:
        K in 3-6, above the margin floor, menu-closed. Erasing the readable
        lottery does not break selection - it only hides the selector.
  P-DF3 (falsification arm): if any readout stays significantly above 0.5,
        T_k is NOT the complete readable ticket and the residual carrier
        (candidate: W_in-feature alignment, so far null in every knockout)
        becomes the next object of study.

8 init seeds, cell (p=113, data_seed=2034), 20k epochs, spectra every 100.
Runs land in runs/doubleflat/p-113/seed2034/seed<iseed>.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
sys.path.insert(0, str(ROOT / "scripts"))
import mlx.core as mx
from grok.config import Config
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
from grok.train import train
from mask_lottery import committee_from_coeffs
from margin_analysis import relM_equal

P, DSEED = 113, 2034
SEEDS = [11285, 33428, 4242, 777, 1001, 1002, 1003, 1004,
         2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
EPOCHS = 20000
nf = P // 2
fourier = Fourier(P)

def freq_energy(M):        # per-frequency energy of (d x p) map M
    F = M @ fourier.basis.T
    E = (F ** 2).sum(0)
    return E[1::2][:nf] + E[2::2][:nf]

def T_k(model):
    W_E = np.array(model.W_E, dtype=np.float64)[:, :P]
    at = model.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)
    W_O = np.array(at.W_O, dtype=np.float64)
    h, dh, _ = W_V.shape
    t = np.zeros(nf)
    for i in range(h):
        OV = W_O[:, i * dh:(i + 1) * dh] @ W_V[i]
        t += freq_energy(OV @ W_E)
    return t

def auc(s, members):
    lab = np.zeros(nf, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(s))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

print(__doc__, flush=True)

def make_cfg(iseed):
    c = Config(p=P, data_seed=DSEED, init_seed=iseed,
               embed_init="orthogonal", attn_init="isometric",
               num_epochs=EPOCHS, save_every=2000)
    return c

# ---- init-time flatness verification (before any training) ----------------
print("=== init flatness check (rel sd of spectra; Gaussian init ~0.09) ===",
      flush=True)
for iseed in SEEDS:
    mx.random.seed(iseed)
    m = Transformer(make_cfg(iseed))
    e = freq_energy(np.array(m.W_E, dtype=np.float64)[:, :P])
    t = T_k(m)
    print(f"  seed {iseed}: W_E energy rel-sd {e.std()/e.mean():.2e}, "
          f"T_k rel-sd {t.std()/t.mean():.2e}", flush=True)

# ---- train -----------------------------------------------------------------
tokens, _ = make_dataset(make_cfg(SEEDS[0]))
fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
for iseed in SEEDS:
    run_dir = ROOT / "runs" / "doubleflat" / "p-113" / f"seed{DSEED}" / f"seed{iseed}"
    if (run_dir / "spectra.npz").exists():
        print(f"skip seed{iseed} (exists)", flush=True)
        continue
    print(f"\n=== training doubleflat seed {iseed} ===", flush=True)
    train(make_cfg(iseed), run_dir, spectra_every=100, log_every=4000)

# ---- evaluation battery ----------------------------------------------------
res = {"emb": [], "T_k": [], "fwd_mlp": [], "fwd_align": []}
rows = []
for iseed in SEEDS:
    run_dir = ROOT / "runs" / "doubleflat" / "p-113" / f"seed{DSEED}" / f"seed{iseed}"
    z = np.load(run_dir / "spectra.npz")
    acc = float(z["test_acc"][-1])
    comm = committee_from_coeffs(z["coeffs"][-1])
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    ge = int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1
    i3k = int(np.argmin(np.abs(z["epochs"] - 3000)))
    menu8 = set((np.argsort(np.abs(z["coeffs"][i3k]))[::-1][:8] + 1).tolist())
    m = Transformer(make_cfg(iseed))
    m.load_weights(str(run_dir / "checkpoints" / "epoch_00000.safetensors"))
    res["emb"].append(auc(freq_energy(
        np.array(m.W_E, dtype=np.float64)[:, :P]), comm))
    res["T_k"].append(auc(T_k(m), comm))
    _, cache = m.run_with_cache(tokens)
    acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
    centered = acts - acts.mean(0, keepdims=True)
    fa2 = fourier.fft2d(centered) ** 2
    per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)
    W_U = np.array(m.W_U, dtype=np.float64)[:, :P]
    W_out = np.array(m.blocks[0].mlp.W_out, dtype=np.float64)
    Fo = fourier.fft1d(W_out.T @ W_U) ** 2
    out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
    res["fwd_mlp"].append(auc(per_nk.sum(1), comm))
    res["fwd_align"].append(auc(np.sqrt(per_nk * out_nk).sum(1), comm))
    rM = relM_equal(sorted(comm), P)
    rows.append((iseed, acc, ge, sorted(comm), rM, set(comm) <= menu8))
    print(f"seed {iseed}: acc {acc:.3f} grok@{ge} committee {sorted(comm)} "
          f"relM {rM:.3f} menu-closed {set(comm) <= menu8}", flush=True)

from scipy.stats import ttest_1samp
print("\n=== P-DF1: epoch-0 readouts on double-flat runs (target: ~0.5) ===")
for name, v in res.items():
    v = np.array(v)
    t = ttest_1samp(v, 0.5)
    print(f"  {name:<9} mean AUC {v.mean():.3f}  sd {v.std():.3f}  "
          f"t-vs-0.5 p={t.pvalue:.3f}  per-seed {np.round(v, 2)}")
print("\nP-DF2:", "PASS" if all(r[1] >= 0.99 for r in rows) else "CHECK",
      "- grok status above; committees/K/floor/menu per row")
