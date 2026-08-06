"""Follow-up on the epoch-0 init-lottery signal: ramp (e0 -> e1000), combined
score, and per-run AUC (are the low-AUC runs the reconfigurers?)."""
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

# known reconfigurers (committee != e3000 amplitude top-K), from margin work
RECONF = {"og_seed0/seed1", "og_seed0/seed51224", "og_seed0/seed63523",
          "og_seed0/seed71539"}

def auc(scores, members, nfreq):
    lab = np.zeros(nfreq, bool); lab[np.array(members) - 1] = True
    r = np.argsort(np.argsort(scores))
    n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

per_run = []
ramp = {0: [], 1000: []}
for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    fourier = Fourier(p)
    tokens, _ = make_dataset(cfg)
    model = Transformer(cfg)
    fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
    name = str(d.relative_to(ROOT / "runs"))
    for ep in (0, 1000):
        ck = d / "checkpoints" / f"epoch_{ep:05d}.safetensors"
        if not ck.exists():
            continue
        model.load_weights(str(ck))
        _, cache = model.run_with_cache(tokens)
        acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
        W_E = np.array(model.W_E, dtype=np.float64)[:, :p]
        W_U = np.array(model.W_U, dtype=np.float64)[:, :p]
        W_out = np.array(model.blocks[0].mlp.W_out, dtype=np.float64)
        centered = acts - acts.mean(0, keepdims=True)
        fa2 = fourier.fft2d(centered) ** 2
        per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)
        out_logit = W_out.T @ W_U
        Fo = fourier.fft1d(out_logit) ** 2
        out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
        align = np.sqrt(per_nk * out_nk).sum(1)
        F = fourier.fft1d(W_E) ** 2
        emb = F.sum(0)[1::2][:nf] + F.sum(0)[2::2][:nf]
        # combined: mean of within-run rank-percentiles
        rk = lambda v: np.argsort(np.argsort(v)) / (nf - 1)
        comb = rk(align) + rk(per_nk.sum(1)) + rk(emb)
        a_comb = auc(comb, final, nf)
        ramp[ep].append(a_comb)
        if ep == 0:
            per_run.append((name, a_comb, len(final), name in RECONF))
print(f"combined-score AUC: e0 mean {np.mean(ramp[0]):.3f} "
      f"(n={len(ramp[0])}), e1000 mean {np.mean(ramp[1000]):.3f}")
print("\nper-run e0 combined AUC (sorted; * = known reconfigurer):")
for name, a, K, rec in sorted(per_run, key=lambda t: t[1]):
    print(f"  {'*' if rec else ' '} {a:.3f}  K={K}  {name}")
r_auc = [a for _, a, _, rec in per_run if rec]
o_auc = [a for _, a, _, rec in per_run if not rec]
from scipy.stats import mannwhitneyu
if r_auc:
    u = mannwhitneyu(r_auc, o_auc, alternative="less")
    print(f"\nreconfigurers mean {np.mean(r_auc):.3f} vs others "
          f"{np.mean(o_auc):.3f}, one-sided p = {u.pvalue:.3f}")
