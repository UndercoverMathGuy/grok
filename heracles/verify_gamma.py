"""VERIFY the last clause of the claim: "hence K* is exactly the feature count".

The theorem's equality case forces the max-margin network's clean spectrum (|G_k|) to be, after
normalisation, a MAXIMISER of the free-amplitude programme

    Gamma_s(p) = max { min_{D != 0} sum_k x_k (1 - cos 2 pi k D / p) : x >= 0, sum_k x_k^{1/s} <= 1 },
    s = nu/q,

so the "feature count" is the SUPPORT SIZE of that maximiser.  Two things must hold for the clause
to be well posed:
  (i)  the maximiser's support size is unique (no tie between different K);
  (ii) K* as the paper computes it (kstar.py's EQUAL-amplitude ladder argmax_K M_K K^{-s}) equals
       that support size.
PROOF_AUDIT F2 already reported that (ii) fails in some cells.  This script re-measures both,
exhaustively where possible: for every support S we solve the inner free-amplitude problem
      max_y min_D sum_{k in S} y_k^s g_k(D),  y >= 0, sum y <= 1
by multistart SLSQP (the inner problem is a small nonconvex max-min; this is COMPUTER-ASSISTED and
carries no certificate -- that is itself part of the verdict).
"""
import itertools, json, os
import numpy as np
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
os.makedirs(OUT, exist_ok=True)


def Gmat(p):
    k = np.arange(1, (p - 1) // 2 + 1); d = np.arange(1, p)
    return 1 - np.cos(2 * np.pi * np.outer(k, d) / p)


def best_on_support(G, S, s, restarts, rng):
    """max over y >= 0, sum y <= 1 of min_D sum_{k in S} y_k^s g_k(D)."""
    Gs = G[list(S)]
    K = len(S)
    best = Gs.sum(0).min() * K ** (-s)          # equal amplitudes: a feasible floor
    besty = np.full(K, 1.0 / K)
    if K == 1:
        return best, besty

    def negt(z):
        return -z[-1]

    def cons(z):
        y = np.abs(z[:-1])
        return np.concatenate([(y ** s) @ Gs - z[-1], [1.0 - y.sum()]])

    for t in range(restarts):
        y0 = besty.copy() if t == 0 else rng.dirichlet(np.ones(K))
        z0 = np.append(y0, float(((y0 ** s) @ Gs).min()))
        try:
            res = minimize(negt, z0, method="SLSQP",
                           constraints=[dict(type="ineq", fun=cons)],
                           bounds=[(0, 1)] * K + [(None, None)],
                           options=dict(maxiter=400, ftol=1e-14))
        except Exception:
            continue
        y = np.abs(res.x[:-1])
        if y.sum() > 1:
            y = y / y.sum()
        v = float(((y ** s) @ Gs).min())
        if v > best:
            best, besty = v, y
    return best, besty


def main():
    rng = np.random.default_rng(3)
    rep = []
    for p in [11, 23, 31, 47]:
        G = Gmat(p)
        nf = G.shape[0]
        for nu, q in [(3, 2), (4, 2), (5, 2)]:
            s = nu / q
            byK = {}
            Kmax = min(nf, 6 if p <= 31 else 5)
            for K in range(1, Kmax + 1):
                bv, bS, by = -1, None, None
                for S in itertools.combinations(range(nf), K):
                    if 0 not in S:       # WLOG 1 in S: the unit group acts transitively
                        continue
                    v, y = best_on_support(G, S, s, restarts=(25 if K <= 3 else 12), rng=rng)
                    if v > bv:
                        bv, bS, by = v, S, y
                # equal-amplitude ladder value for comparison (kstar.py's object)
                mk = max(G[list(S)].sum(0).min() for S in itertools.combinations(range(nf), K) if 0 in S)
                byK[K] = dict(Gamma_K=bv, support=[k + 1 for k in bS],
                              shares=[float(t) for t in np.sort(by)[::-1]],
                              ladder=mk * K ** (-s))
            Kstar_free = max(byK, key=lambda K: byK[K]["Gamma_K"])
            Kstar_ladder = max(byK, key=lambda K: byK[K]["ladder"])
            vals = sorted(((byK[K]["Gamma_K"], K) for K in byK), reverse=True)
            gap = (vals[0][0] - vals[1][0]) / vals[0][0] if len(vals) > 1 else float("nan")
            sh = byK[Kstar_free]["shares"]
            unequal = (max(sh) - min(sh)) / max(sh)
            print(f"p={p:<4} nu={nu} q={q}  Gamma_s={vals[0][0]:.9f}  K*(free)={Kstar_free}  "
                  f"K*(equal-amp ladder)={Kstar_ladder}  {'MATCH' if Kstar_free==Kstar_ladder else '*** DIFFER ***'}"
                  f"   runner-up K={vals[1][1]} at {vals[1][0]/vals[0][0]:.6f} of the max"
                  f"   share spread={unequal:.4f}")
            print(f"        per-K free values: " +
                  "  ".join(f"K={K}:{byK[K]['Gamma_K']:.7f}" for K in sorted(byK)))
            rep.append(dict(p=p, nu=nu, q=q, Gamma=vals[0][0], Kfree=Kstar_free,
                            Kladder=Kstar_ladder, runnerup_K=vals[1][1],
                            runnerup_ratio=vals[1][0] / vals[0][0],
                            shares=sh, support=byK[Kstar_free]["support"],
                            perK={str(K): byK[K]["Gamma_K"] for K in byK}))
    with open(os.path.join(OUT, "gamma.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("wrote", os.path.join(OUT, "gamma.json"))


if __name__ == "__main__":
    main()
