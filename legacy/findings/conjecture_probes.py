"""Two quick probes to seed the conjecture program.

Probe 1 (C-readout): at epoch 0, does the OV-transformed embedding energy
(sum over heads of per-freq energy of W_O^h W_V^h W_E) predict the final
committee better than raw W_E energy? And does combining them beat the
current 0.725 combined score?

Probe 2 (C-grok): do reconfiguring runs grok later than loyal runs
(repair costs time)? Plus: does the blind draw's additive-relation count
predict reconfiguration (the candidate replacement trigger for dead T23)?
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, ttest_1samp

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

a_emb, a_ov, a_both = [], [], []
grok_rc, grok_ok = [], []
add_rc, add_ok = [], []
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    fourier = Fourier(p)
    model = Transformer(cfg)
    model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
    W_E = np.array(model.W_E, dtype=np.float64)[:, :p]

    def freq_energy(W):          # (d, p) -> (nf,)
        F = fourier.fft1d(W) ** 2
        E = F.sum(0)
        return E[1::2][:nf] + E[2::2][:nf]

    emb = freq_energy(W_E)
    at = model.blocks[0].attn
    W_V = np.array(at.W_V, dtype=np.float64)   # (h, dh, d)
    W_O = np.array(at.W_O, dtype=np.float64)   # (d, h*dh)
    h, dh, dm = W_V.shape
    ov = np.zeros(nf)
    for i in range(h):
        OV = W_O[:, i*dh:(i+1)*dh] @ W_V[i]    # (d, d)
        ov += freq_energy(OV @ W_E)
    rk = lambda v: np.argsort(np.argsort(v)) / (nf - 1)
    a_emb.append(auc(emb, final, nf))
    a_ov.append(auc(ov, final, nf))
    a_both.append(auc(rk(emb) + rk(ov), final, nf))

    # probe 2
    K = len(final)
    i3000 = int(np.argmin(np.abs(z["epochs"] - 3000)))
    blind = sorted((np.argsort(np.abs(z["coeffs"][i3000]))[::-1][:K] + 1).tolist())
    reconf = set(blind) != set(final)
    gi = np.argmax(z["test_acc"] >= 0.99)
    grok = int(z["epochs"][gi])
    nadd = n_additive(blind, p)
    (grok_rc if reconf else grok_ok).append(grok)
    (add_rc if reconf else add_ok).append(nadd)
    print(".", end="", flush=True)

print()
print(f"Probe 1 (n={len(a_emb)}): AUC emb {np.mean(a_emb):.3f}  "
      f"OV {np.mean(a_ov):.3f}  emb+OV {np.mean(a_both):.3f}")
t = ttest_1samp(np.array(a_ov), 0.5)
print(f"  OV vs chance: p = {t.pvalue:.4g}")
t2 = ttest_1samp(np.array(a_both) - np.array(a_emb), 0)
print(f"  (emb+OV) - emb paired: mean {np.mean(np.array(a_both)-np.array(a_emb)):+.3f}, p = {t2.pvalue:.4f}")

print(f"\nProbe 2a grok time: reconf mean {np.mean(grok_rc):.0f} (n={len(grok_rc)}) "
      f"vs loyal {np.mean(grok_ok):.0f} (n={len(grok_ok)}), "
      f"MW one-sided(greater) p = {mannwhitneyu(grok_rc, grok_ok, alternative='greater').pvalue:.4f}")
print(f"Probe 2b blind additive count as trigger: reconf mean {np.mean(add_rc):.2f} "
      f"vs loyal {np.mean(add_ok):.2f}, "
      f"MW one-sided(greater) p = {mannwhitneyu(add_rc, add_ok, alternative='greater').pvalue:.4f}")
print(f"  reconf counts {sorted(add_rc)}  loyal counts {sorted(add_ok)}")
