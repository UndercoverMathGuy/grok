"""Extended epoch-0 readout across the full run zoo (claim-1 re-review).

Cohorts:
  natural-normal   24 canonical runs (baseline; expect align AUC ~0.70)
  surgical-normal  surgery/surgery2/transplant (init edited, else normal)
  orth-flat        orthWE/phase2-*/eff-*/combined — per-frequency W_E energy
                   is IDENTICALLY FLAT at init (QR), so any epoch-0
                   predictability must be carried by init geometry, not the
                   energy tilt.

This is the make-or-break test of the 08-02 carrier revision:
  revision RIGHT -> orth-flat AUC well above 0.5 (directions carry).
  revision WRONG -> orth-flat AUC ~ 0.5 (tilt was the whole signal and the
                    tilt_carrier scram_dir drop was a coherence artifact).
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
import mlx.core as mx  # noqa: F401  (model runs on mlx)
from grok.config import Config
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
from mask_lottery import committee_from_coeffs

NATURAL = {"og_seed0", "seed0", "seed1", "seed2", "p-113", "p-127", "p-157"}
SURGICAL = {"surgery", "surgery2", "transplant"}
ORTH = {"orthWE", "phase2-noise", "phase2-noise2", "phase2-tilt",
        "eff-A", "eff-B", "eff-C", "eff-D", "eff-E", "eff-G", "combined"}

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(scores))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

cache = {}   # (p, data_seed) -> (tokens, fourier, fidx)
res = {"natural-normal": [], "surgical-normal": [], "orth-flat": []}
detail = []
for cj in sorted((ROOT / "runs").rglob("config.json")):
    d = cj.parent
    fam = str(d.relative_to(ROOT / "runs")).split("/")[0]
    grp = ("natural-normal" if fam in NATURAL else
           "surgical-normal" if fam in SURGICAL else
           "orth-flat" if fam in ORTH else None)
    if grp is None or not (d / "spectra.npz").exists():
        continue
    ck = d / "checkpoints" / "epoch_00000.safetensors"
    if not ck.exists():
        continue
    cfg = Config.load(cj)
    z = np.load(d / "spectra.npz")
    if float(z["test_acc"][-1]) < 0.99:
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

    model = Transformer(cfg)
    model.load_weights(str(ck))
    _, c_ = model.run_with_cache(tokens)
    acts = np.array(c_["blocks.0.mlp.post"][:, -1], dtype=np.float64)
    centered = acts - acts.mean(0, keepdims=True)
    fa2 = fourier.fft2d(centered) ** 2
    per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)
    W_U = np.array(model.W_U, dtype=np.float64)[:, :p]
    W_out = np.array(model.blocks[0].mlp.W_out, dtype=np.float64)
    Fo = fourier.fft1d(W_out.T @ W_U) ** 2
    out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
    a_mlp = auc(per_nk.sum(1), final, nf)
    a_al = auc(np.sqrt(per_nk * out_nk).sum(1), final, nf)
    # raw W_E per-freq energy readout (should be ~chance in orth cohort by
    # construction — sanity check that flatness holds)
    W_E = np.array(model.W_E, dtype=np.float64)[:, :p]
    Fe = (fourier.fft1d(W_E) ** 2).sum(0)
    a_emb = auc(Fe[1::2][:nf] + Fe[2::2][:nf], final, nf)
    res[grp].append((a_mlp, a_al, a_emb))
    detail.append((grp, str(d.relative_to(ROOT / "runs")), a_mlp, a_al, a_emb))
    print(f"done {d.relative_to(ROOT / 'runs')} [{grp}] "
          f"mlp {a_mlp:.3f} align {a_al:.3f} emb {a_emb:.3f}", flush=True)

print("\n=== epoch-0 readout AUC by cohort (mlp | align | raw-emb-energy) ===")
for grp, rows in res.items():
    a = np.array(rows)
    if not len(a):
        continue
    t_al = ttest_1samp(a[:, 1], 0.5)
    print(f"{grp:<16} n={len(a):>2}  mlp {a[:,0].mean():.3f}  "
          f"align {a[:,1].mean():.3f} (t vs 0.5: p={t_al.pvalue:.2g})  "
          f"emb {a[:,2].mean():.3f}")
