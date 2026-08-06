"""C4 (effective-path readout) + C7 (phase anomaly) + C6 (grok correlates).
Runs on CPU to avoid contending with the GPU training job.

C4 primary (pre-specified): rank-combined align+mlp+OV beats align+mlp+emb
(the 0.725 combined score). Everything else exploratory.

C7: anomaly ratio = CE_actual / CE_sym, where CE_sym is the CE of the
translation-averaged logits L(x) applied uniformly. Correlate log-ratio with
(a) phase-unlocked fraction at committee freqs (1 - coeff^2/energy from the
final spectra snapshot), (b) reconfiguration status.

C6: grok epoch vs blind additive count / reconf status (numpy only).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

import mlx.core as mx
mx.set_default_device(mx.cpu)

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "findings"))
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(scores))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

def fold(x, p):
    x = x % p
    return min(x, p - x)

def n_additive(comm, p):
    s = set(comm); c = 0
    for i in range(len(comm)):
        for j in range(i + 1, len(comm)):
            if fold(comm[i] + comm[j], p) in s or fold(comm[i] - comm[j], p) in s:
                c += 1
    return c

C4 = {k: [] for k in ("emb", "ov", "mlp", "align", "comb_old", "comb_new",
                       "comb_all")}
C7 = []
C6 = []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    fourier = Fourier(p)
    tokens, labels = make_dataset(cfg)
    fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
    model = Transformer(cfg)

    # ---------------- C4 at epoch 0
    model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
    _, cache = model.run_with_cache(tokens)
    acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
    W_E = np.array(model.W_E, dtype=np.float64)[:, :p]
    W_U = np.array(model.W_U, dtype=np.float64)[:, :p]
    W_out = np.array(model.blocks[0].mlp.W_out, dtype=np.float64)

    def freq_energy_cols(W):
        F = fourier.fft1d(W) ** 2
        E = F.sum(0)
        return E[1::2][:nf] + E[2::2][:nf]

    emb = freq_energy_cols(W_E)
    at = model.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)
    W_O = np.array(at.W_O, dtype=np.float64)
    h, dh, _ = W_V.shape
    ov = np.zeros(nf)
    for i in range(h):
        OV = W_O[:, i*dh:(i+1)*dh] @ W_V[i]
        ov += freq_energy_cols(OV @ W_E)
    centered = acts - acts.mean(0, keepdims=True)
    fa2 = fourier.fft2d(centered) ** 2
    per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)
    mlp = per_nk.sum(1)
    Fo = fourier.fft1d(W_out.T @ W_U) ** 2
    out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
    align = np.sqrt(per_nk * out_nk).sum(1)
    rk = lambda v: np.argsort(np.argsort(v)) / (nf - 1)
    comb_old = rk(align) + rk(mlp) + rk(emb)
    comb_new = rk(align) + rk(mlp) + rk(ov)
    comb_all = comb_old + rk(ov)
    for nm, sc in [("emb", emb), ("ov", ov), ("mlp", mlp), ("align", align),
                   ("comb_old", comb_old), ("comb_new", comb_new),
                   ("comb_all", comb_all)]:
        C4[nm].append(auc(sc, final, nf))

    # ---------------- C7 at final checkpoint
    ckpt = sorted((d / "checkpoints").glob("epoch_*.safetensors"))[-1]
    model.load_weights(str(ckpt))
    from grok.metrics import all_logits, cross_entropy_high_precision
    logits = all_logits(model, tokens)
    labels_np = np.array(labels)
    ce_actual = cross_entropy_high_precision(logits, labels_np)
    a_, b_ = np.divmod(np.arange(p * p), p)
    x = (a_[:, None] + b_[:, None] - np.arange(p)[None, :]) % p
    Lx = np.zeros(p)
    np.add.at(Lx, x.ravel(), logits.ravel())
    Lx /= p * p
    sym_logits = Lx[x]
    ce_sym = cross_entropy_high_precision(sym_logits, labels_np)
    ratio = float(ce_actual / ce_sym)
    # phase-unlocked fraction at committee freqs (final spectra snapshot)
    cf = z["coeffs"][-1]; en = z["energy"][-1]
    plf = (cf ** 2) / np.maximum(en, 1e-30)
    w = cf[np.array(final) - 1] ** 2
    unlocked = float(1 - (plf[np.array(final) - 1] * w).sum() / w.sum())
    # C6 pieces
    K = len(final)
    i3000 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    blind = sorted((np.argsort(np.abs(z["coeffs"][i3000]))[::-1][:K] + 1).tolist())
    reconf = set(blind) != set(final)
    gi = np.argmax(z["test_acc"] >= 0.99)
    grok = int(z["epochs"][gi])
    C7.append((ratio, unlocked, reconf))
    C6.append((grok, n_additive(blind, p), reconf))
    print(f"done {d.relative_to(ROOT/'runs')}: ratio {ratio:.1f} "
          f"unlocked {unlocked:.4f}", flush=True)

print("\n=== C4: epoch-0 AUC (n=%d) ===" % len(C4["emb"]))
for nm in ("emb", "ov", "mlp", "align", "comb_old", "comb_new", "comb_all"):
    a = np.array(C4[nm])
    print(f"  {nm:<9} mean {a.mean():.3f} sd {a.std():.3f}")
d_new = np.array(C4["comb_new"]) - np.array(C4["comb_old"])
w = wilcoxon(d_new)
print(f"PRIMARY comb_new - comb_old: mean {d_new.mean():+.4f}, "
      f"wilcoxon p = {w.pvalue:.4f}")
d_all = np.array(C4["comb_all"]) - np.array(C4["comb_old"])
print(f"exploratory comb_all - comb_old: mean {d_all.mean():+.4f}, "
      f"p = {wilcoxon(d_all).pvalue:.4f}")

print("\n=== C7: anomaly ratio ===")
r = np.array([c[0] for c in C7]); u = np.array([c[1] for c in C7])
rc = np.array([c[2] for c in C7])
print(f"ratio range {r.min():.1f} - {r.max():.1f}")
rho, pv = spearmanr(np.log(r), u)
print(f"spearman(log ratio, unlocked frac): rho {rho:+.3f} p {pv:.4f}")
mw = mannwhitneyu(r[rc], r[~rc], alternative="two-sided")
print(f"ratio: reconf median {np.median(r[rc]):.1f} vs loyal "
      f"{np.median(r[~rc]):.1f}, MW p = {mw.pvalue:.4f}")

print("\n=== C6: grok epoch ===")
g = np.array([c[0] for c in C6]); nadd = np.array([c[1] for c in C6])
rc6 = np.array([c[2] for c in C6])
rho2, pv2 = spearmanr(nadd, g)
print(f"spearman(blind additive count, grok epoch): rho {rho2:+.3f} p {pv2:.4f}")
mw2 = mannwhitneyu(g[rc6], g[~rc6], alternative="greater")
print(f"grok: reconf mean {g[rc6].mean():.0f} vs loyal {g[~rc6].mean():.0f}, "
      f"MW one-sided p = {mw2.pvalue:.4f}")
