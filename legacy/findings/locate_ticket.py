"""Locate the init-lottery ticket by component knockout + verify the
He-et-al gradient chain (init alignment -> early growth rate -> winner)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/ruhaanrajadhyaksha/projects/grok")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx
from grok.model import Transformer
from grok.fourier import Fourier
from grok.data import make_dataset
from mask_lottery import discover, final_coeffs_and_acc, committee_from_coeffs

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(scores))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

def rand_like(x, rng):
    xn = np.array(x, dtype=np.float32)
    return mx.array(rng.normal(0, xn.std(), xn.shape).astype(np.float32))

res = {}     # variant -> list of (auc_mlp, auc_align)
chain = []   # per run: (corr(init,lam), auc_lam, auc_init)
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    fourier = Fourier(p)
    tokens, _ = make_dataset(cfg)
    fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
    rng = np.random.default_rng(hash(str(d)) % 2**31)

    def scores(model):
        _, cache = model.run_with_cache(tokens)
        acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
        centered = acts - acts.mean(0, keepdims=True)
        fa2 = fourier.fft2d(centered) ** 2
        per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)   # (nf, m)
        W_U = np.array(model.W_U, dtype=np.float64)[:, :p]
        W_out = np.array(model.blocks[0].mlp.W_out, dtype=np.float64)
        Fo = fourier.fft1d(W_out.T @ W_U) ** 2
        out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
        return per_nk.sum(1), np.sqrt(per_nk * out_nk).sum(1), per_nk

    model = Transformer(cfg)
    ck = d / "checkpoints" / "epoch_00000.safetensors"
    model.load_weights(str(ck))
    mlp0, align0, per_nk0 = scores(model)
    res.setdefault("baseline", []).append((auc(mlp0, final, nf),
                                           auc(align0, final, nf)))
    # knockouts requiring a re-forward
    for variant in ("W_E", "W_in", "attn"):
        a_m, a_a = [], []
        for rep in range(3):
            model.load_weights(str(ck))
            if variant == "W_E":
                model.embed["W_E"] = rand_like(model.W_E, rng)
            elif variant == "W_in":
                model.blocks[0].mlp.W_in = rand_like(model.blocks[0].mlp.W_in, rng)
            else:
                at = model.blocks[0].attn
                for w in ("W_K", "W_Q", "W_V", "W_O"):
                    setattr(at, w, rand_like(getattr(at, w), rng))
            m_, a_, _ = scores(model)
            a_m.append(auc(m_, final, nf)); a_a.append(auc(a_, final, nf))
        res.setdefault(variant, []).append((np.mean(a_m), np.mean(a_a)))
    # out-side knockouts: no re-forward needed (acts unchanged)
    model.load_weights(str(ck))
    W_out_t = np.array(model.blocks[0].mlp.W_out, dtype=np.float64)
    W_U_t = np.array(model.W_U, dtype=np.float64)[:, :p]
    for variant, (Wo, Wu) in {
        "W_out": (None, W_U_t), "W_U": (W_out_t, None)}.items():
        a_a = []
        for rep in range(3):
            Wo_ = rng.normal(0, W_out_t.std(), W_out_t.shape) if Wo is None else Wo
            Wu_ = rng.normal(0, W_U_t.std(), W_U_t.shape) if Wu is None else Wu
            Fo = fourier.fft1d(Wo_.T @ Wu_) ** 2
            out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
            a_a.append(auc(np.sqrt(per_nk0 * out_nk).sum(1), final, nf))
        res.setdefault(variant, []).append((np.nan, np.mean(a_a)))

    # gradient chain from existing spectra: growth rate over epochs 50->150
    ep = z["epochs"]; c = np.abs(z["coeffs"])
    i50 = int(np.argmin(np.abs(ep - 50))); i150 = int(np.argmin(np.abs(ep - 150)))
    lam = np.log(c[i150] + 1e-12) - np.log(c[i50] + 1e-12)
    from scipy.stats import spearmanr
    chain.append((spearmanr(align0, lam).statistic,
                  auc(lam, final, nf), auc(align0, final, nf)))
    print(f"done {d.relative_to(ROOT/'runs')}", flush=True)

print("\n=== component knockout (mean AUC: mlp-energy | align readout) ===")
for v in ("baseline", "W_E", "W_in", "attn", "W_out", "W_U"):
    a = np.array(res[v])
    m = "  -  " if np.isnan(a[:, 0]).all() else f"{np.nanmean(a[:,0]):.3f}"
    print(f"{v:<9} mlp {m}   align {np.nanmean(a[:,1]):.3f}")

ch = np.array(chain)
print(f"\n=== He et al. gradient chain (n={len(ch)}) ===")
print(f"corr(init align, growth rate 50-150): mean {ch[:,0].mean():.3f}")
print(f"AUC(growth rate -> final): mean {ch[:,1].mean():.3f}")
print(f"AUC(init align -> final):  mean {ch[:,2].mean():.3f}")
