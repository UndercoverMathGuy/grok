"""Head start vs exponent: does the W_E init tilt predict early amplitude
c(50) (head start) rather than the growth exponent 50->150?"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
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
    fourier = Fourier(p)
    tokens, _ = make_dataset(cfg)
    fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
    model = Transformer(cfg)
    model.load_weights(str(d / "checkpoints" / "epoch_00000.safetensors"))
    _, cache = model.run_with_cache(tokens)
    acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
    centered = acts - acts.mean(0, keepdims=True)
    fa2 = fourier.fft2d(centered) ** 2
    init = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1).sum(1)  # mlp energy

    ep = z["epochs"]; c = np.abs(z["coeffs"])
    i50 = int(np.argmin(np.abs(ep - 50))); i150 = int(np.argmin(np.abs(ep - 150)))
    lc50, lc150 = np.log(c[i50] + 1e-12), np.log(c[i150] + 1e-12)
    lam = lc150 - lc50
    rows.append((spearmanr(init, lc50).statistic,
                 spearmanr(init, lc150).statistic,
                 spearmanr(init, lam).statistic,
                 spearmanr(lc50, lam).statistic,
                 auc(lc50, final, nf), auc(lc150, final, nf),
                 auc(init, final, nf)))
    print(".", end="", flush=True)

r = np.array(rows)
names = ["corr(init, log c50)", "corr(init, log c150)", "corr(init, exponent)",
         "corr(log c50, exponent)", "AUC(c50 -> final)", "AUC(c150 -> final)",
         "AUC(init -> final)"]
print()
for i, nm in enumerate(names):
    print(f"{nm:<26} mean {r[:,i].mean():+.3f}  sd {r[:,i].std():.3f}")
