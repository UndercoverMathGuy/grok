"""Post-farm evaluation of the double-flat runs (run after doubleflat.py).

Per run (8 seeds):
  readouts   emb spectrum, closed-form T_k, forward mlp, forward align
  knockouts  W_QK, attn-uniform, W_pos, W_in (3 draws each, crc32 seeds)
  He et al.  per-neuron winner-count readout: for each MLP neuron, the
             frequency with the largest product of its a-side and b-side
             arriving spectral magnitudes "wins" the neuron; score_k =
             #neurons won. Also the magnitude x phase-coherence variant
             (He et al. 2602.16849: winner = initial spectral magnitude +
             phase alignment). If these predict the committee here, their
             per-neuron lottery is the reactivated floor carrier.
  stats      committee, K, grok epoch, relM_equal, menu closure

Pooled: mean AUC per readout with t vs 0.5; knockout contrasts (paired).
"""
import sys, zlib
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
sys.path.insert(0, str(ROOT / "scripts"))
import mlx.core as mx
from grok.config import Config
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
from mask_lottery import committee_from_coeffs
from margin_analysis import relM_equal

P, DSEED = 113, 2034
SEEDS = sorted(
    int(d.name[4:])
    for d in (Path("/Users/ruhaanrajadhyaksha/projects/grok/runs/doubleflat"
                   ) / "p-113" / f"seed2034").glob("seed*")
    if (d / "spectra.npz").exists())
nf = P // 2
fourier = Fourier(P)
fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])

def auc(s, members):
    lab = np.zeros(nf, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(s))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

def freq_energy(M):
    F = M @ fourier.basis.T
    E = (F ** 2).sum(0)
    return E[1::2][:nf] + E[2::2][:nf]

def readouts(m, tokens, comm):
    _, cache = m.run_with_cache(tokens)
    acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
    centered = acts - acts.mean(0, keepdims=True)
    fa2 = fourier.fft2d(centered) ** 2
    per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)
    W_U = np.array(m.W_U, dtype=np.float64)[:, :P]
    W_out = np.array(m.blocks[0].mlp.W_out, dtype=np.float64)
    Fo = fourier.fft1d(W_out.T @ W_U) ** 2
    out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
    return (auc(per_nk.sum(1), comm),
            auc(np.sqrt(per_nk * out_nk).sum(1), comm))

def he_scores(m):
    """Per-neuron effective a-side / b-side spectra of W_in through the
    attention arrival map; winner-count readouts."""
    W_E = np.array(m.W_E, dtype=np.float64)[:, :P]
    at = m.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)
    W_O = np.array(at.W_O, dtype=np.float64)
    W_in = np.array(m.blocks[0].mlp.W_in, dtype=np.float64)
    h, dh, _ = W_V.shape
    OV = sum(W_O[:, i * dh:(i + 1) * dh] @ W_V[i] for i in range(h))
    eff = W_in @ (OV @ W_E)            # (m, p): neuron x token weight
    F = eff @ fourier.basis.T           # (m, p) fourier coeffs
    c = F[:, 1::2][:, :nf]; s = F[:, 2::2][:, :nf]
    mag = np.sqrt(c ** 2 + s ** 2)      # (m, nf) per-neuron magnitude
    win = np.argmax(mag, axis=1)
    cnt = np.bincount(win, minlength=nf).astype(float)
    # margin-weighted variant: winner weighted by (top - runner-up)
    srt = np.sort(mag, axis=1)
    marg = srt[:, -1] - srt[:, -2]
    wcnt = np.zeros(nf)
    np.add.at(wcnt, win, marg)
    return cnt, wcnt

res = {k: [] for k in ("emb", "T_k", "fwd_mlp", "fwd_align",
                       "he_count", "he_margin")}
ko = {k: [] for k in ("W_QK", "attn_uniform", "W_pos", "W_in")}
base_mlp = []
rows = []
tokens = None
for iseed in SEEDS:
    run_dir = ROOT / "runs" / "doubleflat" / "p-113" / f"seed{DSEED}" / f"seed{iseed}"
    z = np.load(run_dir / "spectra.npz")
    comm = committee_from_coeffs(z["coeffs"][-1])
    gi = int(np.argmax(z["test_acc"] >= 0.99))
    ge = int(z["epochs"][gi]) if z["test_acc"][gi] >= 0.99 else -1
    i3k = int(np.argmin(np.abs(z["epochs"] - 3000)))
    menu8 = set((np.argsort(np.abs(z["coeffs"][i3k]))[::-1][:8] + 1).tolist())
    cfg = Config.load(run_dir / "config.json")
    if tokens is None:
        tokens, _ = make_dataset(cfg)
    ck = str(run_dir / "checkpoints" / "epoch_00000.safetensors")
    m = Transformer(cfg)
    m.load_weights(ck)
    rng = np.random.default_rng(zlib.crc32(str(run_dir).encode()) % 2**31)

    res["emb"].append(auc(freq_energy(
        np.array(m.W_E, dtype=np.float64)[:, :P]), comm))
    W_V = np.array(m.blocks[0].attn.W_V, dtype=np.float64)
    W_O = np.array(m.blocks[0].attn.W_O, dtype=np.float64)
    h, dh, _ = W_V.shape
    W_E = np.array(m.W_E, dtype=np.float64)[:, :P]
    res["T_k"].append(auc(sum(freq_energy(
        (W_O[:, i * dh:(i + 1) * dh] @ W_V[i]) @ W_E)
        for i in range(h)), comm))
    a_m, a_a = readouts(m, tokens, comm)
    res["fwd_mlp"].append(a_m); res["fwd_align"].append(a_a)
    base_mlp.append(a_m)
    hc, hw = he_scores(m)
    res["he_count"].append(auc(hc, comm))
    res["he_margin"].append(auc(hw, comm))

    def rand_like(x):
        xn = np.array(x, dtype=np.float32)
        return mx.array(rng.normal(0, xn.std(), xn.shape).astype(np.float32))

    for variant in ko:
        vals = []
        reps = 1 if variant == "attn_uniform" else 3
        for _ in range(reps):
            m.load_weights(ck)
            at = m.blocks[0].attn
            if variant == "W_QK":
                at.W_Q = rand_like(at.W_Q); at.W_K = rand_like(at.W_K)
            elif variant == "attn_uniform":
                at.W_Q = mx.zeros(at.W_Q.shape)
            elif variant == "W_pos":
                m.pos_embed["W_pos"] = rand_like(m.W_pos)
            else:
                m.blocks[0].mlp.W_in = rand_like(m.blocks[0].mlp.W_in)
            vals.append(readouts(m, tokens, comm)[0])
        ko[variant].append(np.mean(vals))

    rows.append((iseed, float(z["test_acc"][-1]), ge, sorted(comm),
                 relM_equal(sorted(comm), P), set(comm) <= menu8))
    print(f"seed {iseed}: comm {sorted(comm)} K={len(comm)} grok@{ge} "
          f"fwd_mlp {a_m:.2f} he_margin {res['he_margin'][-1]:.2f}",
          flush=True)

print("\n=== double-flat cohort (n=8): epoch-0 readouts ===")
for name, v in res.items():
    v = np.array(v)
    t = ttest_1samp(v, 0.5)
    print(f"  {name:<10} mean {v.mean():.3f} sd {v.std():.3f} "
          f"t-vs-0.5 p={t.pvalue:.3f}  per-seed {np.round(v, 2)}")
print("\n=== knockouts of the forward-mlp readout (paired vs baseline) ===")
b = np.array(base_mlp)
print(f"  baseline   mean {b.mean():.3f}")
for variant, v in ko.items():
    v = np.array(v)
    t = ttest_rel(v, b)
    print(f"  {variant:<12} mean {v.mean():.3f}  d={v.mean()-b.mean():+.3f} "
          f"p={t.pvalue:.3f}")
print("\n=== run stats ===")
for r in rows:
    print(f"  seed {r[0]}: acc {r[1]:.3f} grok@{r[2]} {r[3]} "
          f"relM {r[4]:.3f} menu-closed {r[5]}")
