"""Knockout WITHIN the orth-flat cohort: where does the ticket live when
the W_E energy spectrum is identically flat?

For each orth-init run, the epoch-0 forward-pass readout predicts the final
committee (align AUC 0.657, ext_readout.py). Knock out one component at a
time (3 stable draws each) and re-score:

  W_E_qr   replace W_E with the QR of a FRESH Gaussian (a different
           orthonormal frame; downstream untouched)
  attn     randomize W_K/W_Q/W_V/W_O
  W_in     randomize the MLP input matrix

If W_E_qr collapses AUC to ~0.5 -> the ticket is (still) the specific W_E
frame, read through the fixed downstream pathway. If it survives while attn
kills it -> in flat runs the lottery migrates into the attention draw.
"""
import json, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
import mlx.core as mx
from grok.config import Config
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
from mask_lottery import committee_from_coeffs

ORTH = {"orthWE", "phase2-noise", "phase2-noise2", "phase2-tilt",
        "eff-A", "eff-B", "eff-C", "eff-D", "eff-E", "eff-G", "combined"}

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(scores))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

def rand_like(x, rng):
    xn = np.array(x, dtype=np.float32)
    return mx.array(rng.normal(0, xn.std(), xn.shape).astype(np.float32))

cache = {}
res = {v: [] for v in ("baseline", "W_E_qr", "attn", "W_in")}
for cj in sorted((ROOT / "runs").rglob("config.json")):
    d = cj.parent
    fam = str(d.relative_to(ROOT / "runs")).split("/")[0]
    if fam not in ORTH or not (d / "spectra.npz").exists():
        continue
    ck = d / "checkpoints" / "epoch_00000.safetensors"
    cfg = Config.load(cj)
    z = np.load(d / "spectra.npz")
    if not ck.exists() or float(z["test_acc"][-1]) < 0.99:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(z["coeffs"][-1])
    key = (p, cfg.data_seed)
    if key not in cache:
        fourier = Fourier(p)
        tokens, _ = make_dataset(cfg)
        fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
        cache[key] = (tokens, fourier, fidx)
    tokens, fourier, fidx = cache[key]
    rng = np.random.default_rng(zlib.crc32(str(d).encode()) % 2**31)

    model = Transformer(cfg)

    def score(m):
        _, c_ = m.run_with_cache(tokens)
        acts = np.array(c_["blocks.0.mlp.post"][:, -1], dtype=np.float64)
        centered = acts - acts.mean(0, keepdims=True)
        fa2 = fourier.fft2d(centered) ** 2
        per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)
        W_U = np.array(m.W_U, dtype=np.float64)[:, :p]
        W_out = np.array(m.blocks[0].mlp.W_out, dtype=np.float64)
        Fo = fourier.fft1d(W_out.T @ W_U) ** 2
        out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
        return (auc(per_nk.sum(1), final, nf),
                auc(np.sqrt(per_nk * out_nk).sum(1), final, nf))

    model.load_weights(str(ck))
    res["baseline"].append(score(model))
    for variant in ("W_E_qr", "attn", "W_in"):
        aa = []
        for _ in range(3):
            model.load_weights(str(ck))
            if variant == "W_E_qr":
                d_, v_ = model.W_E.shape
                g = rng.normal(size=(d_, v_))
                q, _ = np.linalg.qr(g)
                model.embed["W_E"] = mx.array(q.astype(np.float32))
            elif variant == "attn":
                at = model.blocks[0].attn
                for w in ("W_K", "W_Q", "W_V", "W_O"):
                    setattr(at, w, rand_like(getattr(at, w), rng))
            else:
                model.blocks[0].mlp.W_in = rand_like(
                    model.blocks[0].mlp.W_in, rng)
            aa.append(score(model))
        res[variant].append(tuple(np.mean(aa, axis=0)))
    print(f"done {d.relative_to(ROOT / 'runs')}", flush=True)

from scipy.stats import ttest_rel
base = np.array(res["baseline"])
print(f"\n=== orth-cohort knockout (n={len(base)}; mean AUC mlp | align) ===")
for v, rows in res.items():
    a = np.array(rows)
    line = f"{v:<8} mlp {a[:,0].mean():.3f}   align {a[:,1].mean():.3f}"
    if v != "baseline":
        t = ttest_rel(a[:, 1], base[:, 1])
        line += f"   (align vs baseline: dt {a[:,1].mean()-base[:,1].mean():+.3f}, p={t.pvalue:.2g})"
    print(line)
