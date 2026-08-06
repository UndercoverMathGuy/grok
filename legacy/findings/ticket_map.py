"""Where does the lottery ticket live? Component x epoch AUC map.

For each run (24 with spectra), at epochs 0/100/200/400, score every
frequency at five circuit stages and compute AUC vs (a) final committee,
(b) e3000 audition menu (top-8).  Stages:
  emb    energy of W_E cols at freq k
  unemb  energy of W_U cols at freq k
  mlp    neuron post-act 2D Fourier energy at freq k (summed over neurons)
  align  He-et-al-style: sum_n sqrt(in_k(n) * out_k(n))
  logit  |cos(w(a+b-c)) coeff|  (known baseline: ~chance at e0)
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
from grok.metrics import freq_coeffs_and_energy
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

EPOCHS = [0, 100, 200, 400]

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(scores))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

results = {}   # (stage, epoch) -> list of (auc_final, auc_menu)
runs_used = 0
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    i3000 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    menu = (np.argsort(np.abs(z["coeffs"][i3000]))[::-1][:8] + 1).tolist()
    fourier = Fourier(p)
    tokens, _ = make_dataset(cfg)
    model = Transformer(cfg)
    fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
    runs_used += 1
    for ep in EPOCHS:
        ck = d / "checkpoints" / f"epoch_{ep:05d}.safetensors"
        if not ck.exists():
            continue
        model.load_weights(str(ck))
        _, cache = model.run_with_cache(tokens)
        acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
        logits = np.array(cache["blocks.0.resid_post"][:, -1] @ model.W_U,
                          dtype=np.float64)[:, :p]
        W_E = np.array(model.W_E, dtype=np.float64)[:, :p]
        W_U = np.array(model.W_U, dtype=np.float64)[:, :p]
        W_out = np.array(model.blocks[0].mlp.W_out, dtype=np.float64)

        def col_energy(W):   # (d, p) -> (nf,)
            F = fourier.fft1d(W)             # (d, p)
            E = (F ** 2).sum(0)
            return E[1::2][:nf] + E[2::2][:nf]

        emb, unemb = col_energy(W_E), col_energy(W_U)

        centered = acts - acts.mean(0, keepdims=True)
        fa = fourier.fft2d(centered)          # (p^2, d_mlp)
        fa2 = fa ** 2
        per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)  # (nf, d_mlp)
        mlp = per_nk.sum(1)

        out_logit = W_out.T @ W_U             # (d_mlp, p)
        Fo = fourier.fft1d(out_logit) ** 2    # (d_mlp, p)
        out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T  # (nf, d_mlp)
        align = np.sqrt(per_nk * out_nk).sum(1)

        lcoef, _ = freq_coeffs_and_energy(logits, fourier)
        stage_scores = dict(emb=emb, unemb=unemb, mlp=mlp, align=align,
                            logit=np.abs(lcoef))
        for st, sc in stage_scores.items():
            results.setdefault((st, ep), []).append(
                (auc(sc, final, nf), auc(sc, menu, nf)))
    print(f"done {d.relative_to(ROOT/'runs')}", flush=True)

print(f"\n=== mean AUC across {runs_used} runs (final committee | menu) ===")
print(f"{'stage':<8}" + "".join(f"{f'e{ep}':>16}" for ep in EPOCHS))
for st in ("emb", "unemb", "mlp", "align", "logit"):
    row = f"{st:<8}"
    for ep in EPOCHS:
        v = results.get((st, ep))
        if v:
            a = np.array(v)
            row += f"  {a[:,0].mean():.2f}|{a[:,1].mean():.2f} (n{len(v):2d})"
        else:
            row += " " * 16
    print(row)

# significance of the epoch-0 numbers vs 0.5 (t-test across runs)
from scipy.stats import ttest_1samp
print("\nepoch-0 AUCs vs chance (final-committee labels):")
for st in ("emb", "unemb", "mlp", "align", "logit"):
    v = results.get((st, 0))
    if v:
        a = np.array(v)[:, 0]
        t = ttest_1samp(a, 0.5)
        print(f"  {st:<6} mean {a.mean():.3f}  sd {a.std():.3f}  p={t.pvalue:.4f}")
