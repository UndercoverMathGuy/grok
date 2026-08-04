"""CLAIM 1a — the lottery ticket is readable at step zero, and its two
ingredients are the whole readable content.

For EVERY compatible run with an epoch-0 checkpoint, computes four
committee predictors from the untrained weights and scores each by AUC
(probability a random eventual winner outranks a random loser; 0.5 =
chance) against the run's own final committee:

  emb        raw per-frequency W_E energy (ingredient 1 alone)
  T_k        closed-form arrival loudness: sum_h ||W_O^h W_V^h W_E|_k||^2
             (ingredients 1 x 2; three matrix multiplies, no forward pass)
  align      T_k / emb — ingredient 2 ALONE, in-place (divides the embedding
             energy back out, so any signal is attention fit, not size)
  fwd_mlp    per-frequency energy of the MLP activations (full forward)
  fwd_align  fwd_mlp combined with output-side alignment

Statistics are reported at two levels: per-run (descriptive) and per
independent init cluster (primary — dynamics variants / surgical arms that
share an epoch-0 lottery draw are one unit of evidence, see
common.cluster_key).

Expected (SEMIFINAL.md): natural-normal T_k ~0.72; orth-flat (ingredient 1
erased) predictable from ingredient 2 with emb ~0.5; double-flat (both
erased) ALL ~0.5.
"""
import numpy as np
from scipy.stats import ttest_1samp

from common import (discover, auc, freq_energy, fourier, cluster_key,
                    tokens_and_fidx, mlp_freq_energy)
from grok.model import Transformer

res = {}
for r in discover(require_e0=True):
    p = r["cfg"].p
    nf = p // 2
    f = fourier(p)
    tokens, fidx = tokens_and_fidx(r["cfg"])
    m = Transformer(r["cfg"])
    m.load_weights(str(r["e0"]))
    comm = r["committee"]

    W_E = np.array(m.W_E, dtype=np.float64)[:, :p]
    e_emb = freq_energy(W_E, p)
    a_emb = auc(e_emb, comm, nf)
    at = m.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)
    W_O = np.array(at.W_O, dtype=np.float64)
    h, dh, _ = W_V.shape
    tk = sum(freq_energy((W_O[:, i * dh:(i + 1) * dh] @ W_V[i]) @ W_E, p)
             for i in range(h))
    a_tk = auc(tk, comm, nf)
    a_al = auc(tk / e_emb, comm, nf)
    per_nk = mlp_freq_energy(m, tokens, fidx, p)
    W_U = np.array(m.W_U, dtype=np.float64)[:, :p]
    W_out = np.array(m.blocks[0].mlp.W_out, dtype=np.float64)
    Fo = f.fft1d(W_out.T @ W_U) ** 2
    out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
    a_fm = auc(per_nk.sum(1), comm, nf)
    a_fa = auc(np.sqrt(per_nk * out_nk).sum(1), comm, nf)
    res.setdefault(r["cohort"], []).append(
        (cluster_key(r), a_emb, a_tk, a_al, a_fm, a_fa))
    print(f"{r['rel']:<42} [{r['cohort']:<14}] emb {a_emb:.2f} T_k {a_tk:.2f} "
          f"al {a_al:.2f} fwd {a_fm:.2f}/{a_fa:.2f}", flush=True)

NAMES = ("emb", "T_k", "align", "fwd_mlp", "fwd_align")
print("\n=== CLAIM 1a: epoch-0 readout AUC by cohort ===")
for cohort in ("natural-normal", "surgical", "orth-flat", "double-flat",
               "other-normal"):
    if cohort not in res:
        continue
    rows = res[cohort]
    a = np.array([x[1:] for x in rows])
    clusters = {}
    for x in rows:
        clusters.setdefault(x[0], []).append(x[1:])
    cm = np.array([np.mean(v, axis=0) for v in clusters.values()])
    print(f"{cohort} n={len(a)} runs, {len(cm)} independent inits/masks")
    for i, name in enumerate(NAMES):
        t = ttest_1samp(a[:, i], 0.5)
        tc = (ttest_1samp(cm[:, i], 0.5) if len(cm) > 1 else t)
        print(f"   {name:<9} run-level {a[:, i].mean():.3f} (p={t.pvalue:.1e})"
              f"  cluster-level {cm[:, i].mean():.3f} (p={tc.pvalue:.1e})")
print("""
Backs SEMIFINAL claim 1. Cluster-level numbers are primary: dynamics
variants / surgical arms sharing one epoch-0 init are one unit of evidence.
'align' is ingredient 2 measured in untouched natural runs — the in-place
counterpart of the orth-flat erasure argument.""")
