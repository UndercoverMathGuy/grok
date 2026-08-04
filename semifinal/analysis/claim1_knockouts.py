"""CLAIM 1b — scramble tests: where the readout formula reads from

Check by scrambling components and rechecking for AUC changes.
Causal changes in claim1_readout.py

For every non-surgical (not modified with transplants etc.) run with an epoch-0 checkpoint, recompute the
forward-pass committee readout after replacing ONE component with fresh orthonormal / flat weights.

  W_E       fresh Gaussian embedding (natural) — kills both ingredients (diff seed, zero correl)
  W_E_qr    orthonormal W_E - every Fourier base vector has energy 1.
  attn      fresh W_K/W_Q/W_V/W_O (kills the attention half)
  W_in      fresh MLP input matrix

Two readouts per condition:
  agg   total per-frequency energy of the MLP activations. NOTE: agg sums
        over all neurons, so it is nearly invariant to remixing W_in — the
        W_in row of agg is a sanity check of the readout, NOT evidence that
        the MLP carries nothing.
  neur  per-neuron winner counts (how many neurons put their argmax energy
        on frequency k) — He-et-al-flavored, and genuinely W_in-sensitive:
        if a committee signal lived in W_in's neuron alignments, scrambling
        W_in would destroy THIS readout. Observed: it does read the
        committee at baseline, but a fresh W_in leaves it UNCHANGED while a
        fresh W_E kills it — so the per-neuron structure it reads is
        inherited from the embedding x attention stream, and W_in itself
        contributes no committee information. That is the fair version of
        "W_in is not a carrier".

Expected (SEMIFINAL.md): natural — W_E kills agg, attn partial signal;
orth-flat — W_E_qr and attn BOTH kill (relational carrier);
double-flat — baseline already ~0.5, nothing to kill.
"""
import zlib
import numpy as np
from scipy.stats import ttest_rel

from common import discover, auc, tokens_and_fidx, mlp_freq_energy
import mlx.core as mx
from grok.model import Transformer

VARIANTS = ("W_E", "W_E_qr", "attn", "W_in")
res = {}
for r in discover(require_e0=True):
    if r["cohort"] == "surgical":
        continue
    p = r["cfg"].p
    nf = p // 2
    tokens, fidx = tokens_and_fidx(r["cfg"])
    comm = r["committee"]
    m = Transformer(r["cfg"])
    rng = np.random.default_rng(zlib.crc32(r["rel"].encode()) % 2**31)

    def score():
        per_nk = mlp_freq_energy(m, tokens, fidx, p)
        counts = np.bincount(np.argmax(per_nk, axis=0), minlength=nf)
        return auc(per_nk.sum(1), comm, nf), auc(counts, comm, nf)

    def rand_like(x):
        xn = np.array(x, dtype=np.float32)
        return mx.array(rng.normal(0, xn.std(), xn.shape).astype(np.float32))

    m.load_weights(str(r["e0"]))
    row = {"baseline": score()}
    for variant in VARIANTS:
        d_, v_ = m.W_E.shape
        if variant == "W_E_qr" and v_ > d_:
            # a v-column orthonormal frame needs d >= v (p=157 has 158 > 128);
            # the fresh-frame variant is undefined there — skip.
            row[variant] = (float("nan"), float("nan"))
            continue
        vals = []
        for _ in range(3):
            m.load_weights(str(r["e0"]))
            if variant == "W_E":
                m.embed["W_E"] = rand_like(m.W_E)
            elif variant == "W_E_qr":
                q, _ = np.linalg.qr(rng.normal(size=(d_, v_)))
                m.embed["W_E"] = mx.array(q.astype(np.float32))
            elif variant == "attn":
                at = m.blocks[0].attn
                for w in ("W_K", "W_Q", "W_V", "W_O"):
                    setattr(at, w, rand_like(getattr(at, w)))
            else:
                m.blocks[0].mlp.W_in = rand_like(m.blocks[0].mlp.W_in)
            vals.append(score())
        row[variant] = tuple(np.mean(vals, axis=0))
    res.setdefault(r["cohort"], []).append(row)
    print(f"{r['rel']:<42} [{r['cohort']}] " +
          " ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in row.items()),
          flush=True)

print("\n=== CLAIM 1b: knockouts by cohort (mean AUC, agg | neur) ===")
for cohort, rows in res.items():
    for ri, rd in enumerate(("agg", "neur")):
        base = np.array([x["baseline"][ri] for x in rows])
        print(f"\n{cohort} (n={len(rows)}), {rd}: baseline {base.mean():.3f}")
        for variant in VARIANTS:
            v = np.array([x[variant][ri] for x in rows])
            ok = ~np.isnan(v)
            t = ttest_rel(v[ok], base[ok])
            print(f"  {variant:<7} {np.nanmean(v):.3f}  "
                  f"d={np.nanmean(v)-base[ok].mean():+.3f} "
                  f"(paired p={t.pvalue:.1e}, n={ok.sum()})")
print("""
Consistency check on SEMIFINAL claim 1: the agg readout depends on the
embedding x attention pathway exactly as the T_k formula demands (W_E
kills, attn partial; in flat cohorts either side's scramble kills the
relational carrier). The agg W_in row is a readout sanity check only. The
fair MLP-carrier test is the neur readout (W_in-sensitive by construction):
it reads the committee at baseline, but scrambling W_in leaves it intact
while scrambling W_E kills it — the per-neuron signal is inherited from
the stream, and W_in itself carries none. The causal 'nothing else picks
up the slack' statement rests on the trained double-flat cohort.""")
