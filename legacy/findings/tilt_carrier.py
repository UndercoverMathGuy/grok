"""Decompose the W_E init ticket: is the per-frequency ENERGY spectrum the
carrier of the committee signal, or the frequency-subspace DIRECTIONS?

Motivation: the orthWE necessity test (QR-flatten -> committees change) is
confounded twice: (a) QR removes energy tilt AND all column geometry at once,
and (b) committee identity is chaotic to sub-threshold init perturbations
(T45), so "the committee changed" carries almost no information about WHICH
init variable was the carrier. This script asks the question at the readout
level instead, where it is well-posed: the epoch-0 forward-pass readouts
(per-freq MLP energy, align score) predict the final committee at AUC ~0.70
(T18/T20). We surgically edit W_E in Fourier space and measure which edit
destroys / preserves that predictive signal.

Variants (per run, 3 seeded draws each where random):
  baseline      untouched epoch-0 weights
  perm_energy   permute the 56 per-freq energies across frequencies; each
                frequency keeps its own (d x 2) subspace DIRECTIONS, rescaled
                to the permuted energy. Marginal energy distribution intact,
                tilt information destroyed.
                -> if energy is the necessary carrier: AUC -> chance.
  scram_dir     redraw each frequency's (d x 2) block as fresh Gaussian,
                rescaled to exactly the ORIGINAL energy. Directions destroyed,
                energy spectrum intact.
                -> if energy is the sufficient carrier: AUC ~ baseline.
  flat_energy   set every frequency's energy to the mean energy (keep
                directions) — the analysis-level analog of the orthWE
                intervention, but surgical: only the tilt is removed.
                -> AUC -> chance iff the tilt (not geometry) was the signal.

DC column (F[:,0]) and the '=' token column (W_E[:, p]) are never touched.
Stable per-run seeds (crc32, not salted hash) so the output is exactly
reproducible run-to-run.
"""
import sys
import zlib
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

VARIANTS = ("baseline", "perm_energy", "scram_dir", "flat_energy")
res = {v: [] for v in VARIANTS}

for d, cfg in discover():
    coeffs, acc, z = final_coeffs_and_acc(d, cfg)
    if acc < 0.99 or z is None:
        continue
    p = cfg.p; nf = p // 2
    final = committee_from_coeffs(coeffs)
    fourier = Fourier(p)
    tokens, _ = make_dataset(cfg)
    fidx = np.stack([fourier.freq_indices_2d(k) for k in range(1, nf + 1)])
    rng = np.random.default_rng(zlib.crc32(str(d).encode()) % 2**31)

    def scores(model):
        _, cache = model.run_with_cache(tokens)
        acts = np.array(cache["blocks.0.mlp.post"][:, -1], dtype=np.float64)
        centered = acts - acts.mean(0, keepdims=True)
        fa2 = fourier.fft2d(centered) ** 2
        per_nk = fa2[fidx.reshape(-1)].reshape(nf, 8, -1).sum(1)
        W_U = np.array(model.W_U, dtype=np.float64)[:, :p]
        W_out = np.array(model.blocks[0].mlp.W_out, dtype=np.float64)
        Fo = fourier.fft1d(W_out.T @ W_U) ** 2
        out_nk = (Fo[:, 1::2][:, :nf] + Fo[:, 2::2][:, :nf]).T
        return per_nk.sum(1), np.sqrt(per_nk * out_nk).sum(1)

    model = Transformer(cfg)
    ck = d / "checkpoints" / "epoch_00000.safetensors"
    model.load_weights(str(ck))
    W_E0 = np.array(model.W_E, dtype=np.float64)            # (d, p+1)
    F0 = W_E0[:, :p] @ fourier.basis.T                       # (d, p)
    E0 = (F0 ** 2).sum(0)[1::2][:nf] + (F0 ** 2).sum(0)[2::2][:nf]

    def with_WE(Fmod):
        W = W_E0.copy()
        W[:, :p] = Fmod @ fourier.basis
        model.load_weights(str(ck))
        model.embed["W_E"] = mx.array(W.astype(np.float32))
        return model

    def edit(mode, rng):
        Fm = F0.copy()
        if mode == "perm_energy":
            perm = rng.permutation(nf)
            for k in range(nf):
                s = np.sqrt(E0[perm[k]] / E0[k])
                Fm[:, 2 * k + 1] *= s
                Fm[:, 2 * k + 2] *= s
        elif mode == "scram_dir":
            for k in range(nf):
                blk = rng.normal(size=(F0.shape[0], 2))
                blk *= np.sqrt(E0[k] / (blk ** 2).sum())
                Fm[:, 2 * k + 1] = blk[:, 0]
                Fm[:, 2 * k + 2] = blk[:, 1]
        elif mode == "flat_energy":
            for k in range(nf):
                s = np.sqrt(E0.mean() / E0[k])
                Fm[:, 2 * k + 1] *= s
                Fm[:, 2 * k + 2] *= s
        return Fm

    m0, a0 = scores(with_WE(F0))
    res["baseline"].append((auc(m0, final, nf), auc(a0, final, nf)))
    for mode in ("perm_energy", "scram_dir", "flat_energy"):
        reps = 1 if mode == "flat_energy" else 3
        am, aa = [], []
        for _ in range(reps):
            m_, a_ = scores(with_WE(edit(mode, rng)))
            am.append(auc(m_, final, nf)); aa.append(auc(a_, final, nf))
        res[mode].append((np.mean(am), np.mean(aa)))
    print(f"done {d.relative_to(ROOT / 'runs')}", flush=True)

from scipy.stats import ttest_rel
print("\n=== W_E carrier decomposition (mean AUC over runs: mlp | align) ===")
base = np.array(res["baseline"])
for v in VARIANTS:
    a = np.array(res[v])
    line = f"{v:<12} mlp {a[:,0].mean():.3f}   align {a[:,1].mean():.3f}"
    if v != "baseline":
        t = ttest_rel(a[:, 1], base[:, 1])
        line += f"   (paired vs baseline, align: dt {a[:,1].mean()-base[:,1].mean():+.3f}, p={t.pvalue:.2g})"
    print(line)
print(f"n = {len(base)} runs")
print("""
Reading: energy is the carrier iff perm_energy ~ chance (0.5) AND
scram_dir ~ baseline. If scram_dir drops partway, the directions carry a
share of the signal and the orthWE result overstates what "the tilt" does.""")
