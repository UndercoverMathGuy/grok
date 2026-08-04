"""Margin analysis for the matrix farm (rebuilds the scratchpad tooling).

Reproduces, on the p-<prime>/seed<data>/seed<init> runs, the core tests of
runs/eviction/MARGIN_HYPOTHESIS.md:

  committee(run)      largest-log-gap detection on final |phase-locked coeff|
  relM(S, alloc)      1 - max_{x!=0} sum_k a_k cos(w_k x) / sum_k a_k
  lp_relM(S)          relM at the max-min-optimal amplitude allocation (an LP)
  percentile(S)       LP-optimal relM of S vs size-matched random K-subsets
  counterfactual      chosen committee vs top-K-by-amplitude committee
  homeostat(run)      minGap = A_tot * relM read off the final logits, in nats

Everything is a function so other scripts / a REPL can import it; running the
file prints the per-run table and the pooled statistics.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

RUNS = Path("runs")


# --------------------------------------------------------------------------- #
# run discovery
# --------------------------------------------------------------------------- #

def matrix_runs():
    """Every completed matrix run as (p, data_seed, init_seed, dir), sorted."""
    out = []
    for pdir in sorted(RUNS.glob("p-*")):
        p = int(pdir.name.split("-")[1])
        for ddir in sorted(pdir.glob("seed*")):
            data_seed = int(ddir.name[4:])
            for rdir in sorted(ddir.glob("seed*")):
                if (rdir / "spectra.npz").exists() and (rdir / "metrics.json").exists():
                    out.append((p, data_seed, int(rdir.name[4:]), rdir))
    return out


# --------------------------------------------------------------------------- #
# committee detection
# --------------------------------------------------------------------------- #

def committee(coeffs_final):
    """Frequencies (1-indexed) above the largest log-gap in sorted |coeff|."""
    a = np.abs(coeffs_final)
    order = np.argsort(a)[::-1]
    sorted_a = a[order]
    logs = np.log(sorted_a + 1e-12)
    gaps = logs[:-1] - logs[1:]
    # search only the plausible committee range (top ~12) for the cliff
    cut = int(np.argmax(gaps[:12])) + 1
    members = np.sort(order[:cut] + 1)
    return members.tolist(), a[order[:cut]]


# --------------------------------------------------------------------------- #
# the margin functional
# --------------------------------------------------------------------------- #

def _cos_matrix(freqs, p):
    """(p-1, K) matrix cos(2 pi k x / p) for x=1..p-1, k in freqs."""
    x = np.arange(1, p)[:, None]
    k = np.asarray(freqs)[None, :]
    return np.cos(2 * np.pi * k * x / p)

def relM_equal(freqs, p):
    """Relative min-margin at equal amplitudes: 1 - max_{x!=0} mean_k cos."""
    C = _cos_matrix(freqs, p)
    return 1.0 - C.mean(axis=1).max()

def lp_relM(freqs, p):
    """Max-min-optimal relative margin over amplitude allocations (an LP).

    max m  s.t.  sum_k a_k cos(w_k x) <= 1 - m  for all x!=0,
                 sum_k a_k = 1,  a_k >= 0.
    Variables ordered [a_1..a_K, m]; we minimize -m.
    """
    C = _cos_matrix(freqs, p)          # (p-1, K)
    K = C.shape[1]
    # -m + sum_k a_k cos <= 1  ->  [C | +1] . [a; m] <= 1
    A_ub = np.hstack([C, np.ones((C.shape[0], 1))])
    b_ub = np.ones(C.shape[0])
    A_eq = np.hstack([np.ones((1, K)), np.zeros((1, 1))])
    b_eq = np.array([1.0])
    c = np.zeros(K + 1); c[-1] = -1.0
    bounds = [(0, None)] * K + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return float(res.x[-1]), res.x[:K]


# --------------------------------------------------------------------------- #
# null distributions / percentiles
# --------------------------------------------------------------------------- #

def null_percentile(freqs, p, score_fn, n=4000, rng=None):
    """Percentile of score_fn(freqs) among random K-subsets of {1..p//2}."""
    rng = rng or np.random.default_rng(0)
    K = len(freqs)
    pool = np.arange(1, p // 2 + 1)
    obs = score_fn(freqs, p)
    if isinstance(obs, tuple):
        obs = obs[0]
    null = np.empty(n)
    for i in range(n):
        s = rng.choice(pool, size=K, replace=False)
        v = score_fn(s.tolist(), p)
        null[i] = v[0] if isinstance(v, tuple) else v
    return 100.0 * (null < obs).mean(), obs, null


# --------------------------------------------------------------------------- #
# homeostat: amplitudes off the final logits -> minGap in nats
# --------------------------------------------------------------------------- #

def homeostat(run_dir, p):
    """(A_tot, relM, minGap) from the translation-averaged final logits.

    Loads the final checkpoint, builds full-grid logits, averages over the
    p 'miss' diagonals x=a+b-c to get L(x), reads cosine amplitudes a_k, and
    returns minGap = min_{x!=0} sum_k a_k (1 - cos(w_k x)) in nats.
    """
    import mlx.core as mx
    from grok.config import Config
    from grok.model import Transformer
    from grok.metrics import all_logits

    cfg = Config.load(run_dir / "config.json")
    ckpts = sorted((run_dir / "checkpoints").glob("epoch_*.safetensors"))
    model = Transformer(cfg)
    model.load_weights(str(ckpts[-1]))
    logits = all_logits(model, _tokens(cfg))          # (p^2, p)

    a, b = np.divmod(np.arange(p * p), p)
    c = np.arange(p)
    x = (a[:, None] + b[:, None] - c[None, :]) % p     # (p^2, p) miss per entry
    Lx = np.zeros(p)
    np.add.at(Lx, x.ravel(), logits.ravel())
    Lx /= p * p                                        # mean logit at miss x
    ks = np.arange(1, p // 2 + 1)
    amp = (2.0 / p) * (Lx[None, :] * np.cos(2 * np.pi * ks[:, None]
                                            * np.arange(p)[None, :] / p)).sum(axis=1)
    gap = (amp[:, None] * (1 - np.cos(2 * np.pi * ks[:, None]
                                      * np.arange(1, p)[None, :] / p))).sum(axis=0)
    A_tot = amp[amp > 0].sum()
    return float(A_tot), float(gap.min() / max(A_tot, 1e-9)), float(gap.min())

def _tokens(cfg):
    from grok.data import make_dataset
    t, _ = make_dataset(cfg)
    return t


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def analyze():
    runs = matrix_runs()
    rng = np.random.default_rng(0)
    rows = []
    print(f"{len(runs)} completed runs\n")
    header = ("p   dseed  init    committee                    K  "
              "relM_eq  lp_relM  pctile  amp-fav-lp  cf")
    print(header)
    print("-" * len(header))
    for p, dseed, init, d in runs:
        z = np.load(d / "spectra.npz")
        acc = float(z["test_acc"][-1])
        coeffs_final = z["coeffs"][-1]
        comm, _ = committee(coeffs_final)
        rM = relM_equal(comm, p)
        lpM, _ = lp_relM(comm, p)
        pct, _, _ = null_percentile(comm, p, lp_relM, n=3000, rng=rng)

        # counterfactual: top-K by |coeff| at MID-AUDITION (rich-get-richer
        # set, before consolidation/eviction) vs the final committee.
        K = len(comm)
        e3000 = int(np.argmin(np.abs(z["epochs"] - 3000)))
        amp_fav = np.sort(
            np.argsort(np.abs(z["coeffs"][e3000]))[::-1][:K] + 1).tolist()
        lp_fav, _ = lp_relM(amp_fav, p)
        same = set(amp_fav) == set(comm)
        cf = "same" if same else ("WIN" if lpM > lp_fav else "LOSS")

        rows.append(dict(p=p, dseed=dseed, init=init, acc=acc, comm=comm, K=K,
                         relM=rM, lp=lpM, pct=pct, amp_fav=amp_fav,
                         lp_fav=lp_fav, cf=cf))
        cstr = "{" + ",".join(map(str, comm)) + "}"
        print(f"{p:<4}{dseed:<7}{init:<8}{cstr:<28} {K}  "
              f"{rM:6.3f}   {lpM:6.3f}   {pct:5.1f}   {lp_fav:6.3f}    {cf}")

    _pooled(rows)
    return rows

def _pooled(rows):
    pcts = np.array([r["pct"] for r in rows])
    print("\n=== pooled ===")
    print(f"LP-optimal relM percentile: mean {pcts.mean():.1f}  "
          f"(min {pcts.min():.1f}, max {pcts.max():.1f}), n={len(pcts)}")
    # uniform-null test on the mean percentile (each ~Uniform(0,100), var 833.3)
    z = (pcts.mean() - 50) / (np.sqrt(833.33 / len(pcts)))
    from scipy.stats import norm
    print(f"  vs uniform-null mean 50: z={z:.2f}, one-sided p={norm.sf(z):.2e}")
    cf = [r["cf"] for r in rows]
    inf = [c for c in cf if c != "same"]
    wins = sum(c == "WIN" for c in inf)
    print(f"counterfactual: {cf.count('same')} same, {wins}/{len(inf)} WIN "
          f"among informative")
    if inf:
        from scipy.stats import binomtest
        print(f"  sign test (p=0.5): p={binomtest(wins, len(inf)).pvalue:.4f}")
    byp = {}
    for r in rows:
        byp.setdefault(r["p"], []).append(r["pct"])
    print("by prime:  " + "   ".join(
        f"p{p}: {np.mean(v):.1f} (n{len(v)})" for p, v in sorted(byp.items())))


if __name__ == "__main__":
    analyze()
