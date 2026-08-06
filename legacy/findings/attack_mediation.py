"""Adversarial re-analysis of P2/T22 ("amplitude head start, NOT growth-rate
advantage").

The claim rests on: corr(init W_E energy, log c50) = 0.33 vs
corr(init, exponent 50->150) = 0.15. But the committee signal in logit
amplitudes goes from chance (AUC 0.56) at e0 to 0.95 at e50 — i.e. the
*entire* differentiation happens through growth during epochs 0-50, a window
the 50->150 exponent never sees. If corr(init, growth 0->50) is as large as
corr(init, c50), then the data cannot distinguish "head start" from "init
energy sets the early growth rate", and P2's dichotomy (and the S3 anti-He
framing "head start, not best-gradient-fastest") is unsupported.

Also: interval-resolved corr(init, log-amplitude growth per 50-epoch bin) and
the attenuation structure (exponent from a 2-snapshot difference is noisy;
its correlation is biased toward 0 relative to a level variable).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
from grok.model import Transformer
from grok.fourier import Fourier
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(scores))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

rows = []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    model = Transformer(cfg)
    model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
    fourier = Fourier(p)
    F = fourier.fft1d(np.array(model.W_E, dtype=np.float64)[:, :p]) ** 2
    e = F.sum(0)[1::2][:nf] + F.sum(0)[2::2][:nf]   # init W_E energy per freq

    ep = z["epochs"]; c = np.abs(z["coeffs"])
    idx = {t: int(np.argmin(np.abs(ep - t))) for t in (0, 50, 100, 150, 200)}
    lc = {t: np.log(c[i] + 1e-12) for t, i in idx.items()}

    row = dict(
        c_lc0=spearmanr(e, lc[0]).statistic,
        c_lc50=spearmanr(e, lc[50]).statistic,
        c_g0_50=spearmanr(e, lc[50] - lc[0]).statistic,
        c_g50_150=spearmanr(e, lc[150] - lc[50]).statistic,
        c_g50_100=spearmanr(e, lc[100] - lc[50]).statistic,
        c_g100_150=spearmanr(e, lc[150] - lc[100]).statistic,
        c_g150_200=spearmanr(e, lc[200] - lc[150]).statistic,
        auc0=auc(lc[0], final, nf),
        auc50=auc(lc[50], final, nf),
        auc_g0_50=auc(lc[50] - lc[0], final, nf),
        auc_init=auc(e, final, nf),
    )
    rows.append(row)
    print(".", end="", flush=True)

print()
r = {k: np.array([row[k] for row in rows]) for k in rows[0]}
for k, v in r.items():
    print(f"{k:<12} mean {v.mean():+.3f}  sd {v.std():.3f}")

print("\nkey comparison:")
print(f"  corr(init, log c50)        = {r['c_lc50'].mean():+.3f}   <- their 'head start'")
print(f"  corr(init, growth 0->50)   = {r['c_g0_50'].mean():+.3f}   <- unmeasured in their T22")
print(f"  corr(init, growth 50->150) = {r['c_g50_150'].mean():+.3f}   <- their 'exponent'")
print(f"  corr(init, log c0)         = {r['c_lc0'].mean():+.3f}   (logit amp at init)")
print(f"  AUC(growth 0->50 -> final) = {r['auc_g0_50'].mean():.3f}  vs AUC(c50) = {r['auc50'].mean():.3f}, AUC(c0) = {r['auc0'].mean():.3f}")
