"""Closed-form ticket variable on both cohorts (no forward pass):

  T_k = sum_h || (W_O^h W_V^h W_E) restricted to frequency k ||^2

the per-frequency energy of the embedding as transmitted through the
attention OV circuit at epoch 0. AUC vs the run's final committee:
natural-normal cohort ~0.724 (reproduces T38), orth-flat cohort ~0.661
(~= the full forward-pass readout, ext_readout.py's 0.657) — i.e. this one
scalar per frequency carries essentially the whole init-lottery signal in
both regimes.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
import mlx.core as mx  # noqa: F401
from grok.config import Config
from grok.model import Transformer
from grok.fourier import Fourier
from mask_lottery import committee_from_coeffs

ORTH = {"orthWE", "phase2-noise", "phase2-noise2", "phase2-tilt",
        "eff-A", "eff-B", "eff-C", "eff-D", "eff-E", "eff-G", "combined"}
NAT = {"og_seed0", "seed0", "seed1", "seed2", "p-113", "p-127", "p-157"}

def auc(s, members, nf):
    lab = np.zeros(nf, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(s))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

out = {"orth-flat": [], "natural": []}
F_ = {}
for cj in sorted((ROOT / "runs").rglob("config.json")):
    d = cj.parent
    fam = str(d.relative_to(ROOT / "runs")).split("/")[0]
    grp = ("orth-flat" if fam in ORTH else "natural" if fam in NAT else None)
    if grp is None or not (d / "spectra.npz").exists():
        continue
    ck = d / "checkpoints" / "epoch_00000.safetensors"
    z = np.load(d / "spectra.npz")
    if not ck.exists() or float(z["test_acc"][-1]) < 0.99:
        continue
    cfg = Config.load(cj)
    p = cfg.p; nf = p // 2
    if p not in F_:
        F_[p] = Fourier(p)
    fourier = F_[p]
    final = committee_from_coeffs(z["coeffs"][-1])
    m = Transformer(cfg)
    m.load_weights(str(ck))
    W_E = np.array(m.W_E, dtype=np.float64)[:, :p]
    at = m.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)
    W_O = np.array(at.W_O, dtype=np.float64)
    h, dh, _ = W_V.shape
    ov = np.zeros(nf)
    for i in range(h):
        OV = W_O[:, i * dh:(i + 1) * dh] @ W_V[i]
        Fp = (OV @ W_E) @ fourier.basis.T
        E = (Fp ** 2).sum(0)
        ov += E[1::2][:nf] + E[2::2][:nf]
    out[grp].append(auc(ov, final, nf))
    print(f"done {d.relative_to(ROOT / 'runs')} [{grp}]", flush=True)

print("\n=== closed-form T_k = OV-transmitted embedding energy at init ===")
for g, v in out.items():
    v = np.array(v)
    print(f"{g:<10} AUC mean {v.mean():.3f}  (n={len(v)}, "
          f"t vs 0.5 p={ttest_1samp(v, 0.5).pvalue:.2g})")
