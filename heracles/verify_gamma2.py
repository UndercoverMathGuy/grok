"""VERIFY X3, done properly: the SUPPORT SIZE of the Gamma_s maximiser vs kstar.py's K*.

Gamma_K(p,s) := max over supports S of size K of max{ min_{D != 0} sum_{k in S} x_k g_k(D) :
                x >= 0, sum x_k^{1/s} <= 1 }  is NON-DECREASING in K (a size-K optimum may put zero
weight on a frequency), so "the maximiser's support size" is the SMALLEST K with
Gamma_K = Gamma_s.  My first pass reported argmax_K, which is confounded by that monotonicity and by
optimiser noise; this script reports the smallest K within a relative tolerance, together with the
optimal shares there (which is what decides whether the attaining block has to be SHARE-WEIGHTED).

kstar.py's object is different: K*_ladder = argmax_K M_K(p) K^{-s} with
M_K(p) = max_{|S|=K} min_{D != 0} sum_{k in S} g_k(D)  -- the EQUAL-amplitude restriction.
"""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_gamma import Gmat, best_on_support   # noqa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")
TOL = 1e-7


def main():
    rng = np.random.default_rng(5)
    rep = []
    print(f"{'p':>4}{'nu':>4}{'q':>3} | {'Gamma_s':>12} {'K(free support)':>16} "
          f"{'K*(kstar ladder)':>17} {'share spread':>13}  shares at the free optimum")
    for p in [11, 23, 31]:
        G = Gmat(p)
        nf = G.shape[0]
        for nu in [3, 4, 5]:
            s = nu / 2.0
            GK, SK, YK, MK = {}, {}, {}, {}
            Kmax = min(nf, 6)
            for K in range(1, Kmax + 1):
                bv, bS, by = -1, None, None
                for S in itertools.combinations(range(nf), K):
                    if 0 not in S:
                        continue
                    v, y = best_on_support(G, S, s, restarts=(20 if K <= 3 else 8), rng=rng)
                    if v > bv:
                        bv, bS, by = v, S, y
                GK[K], SK[K], YK[K] = bv, bS, by
                MK[K] = max(G[list(S)].sum(0).min()
                            for S in itertools.combinations(range(nf), K) if 0 in S) * K ** (-s)
            gmax = max(GK.values())
            Kfree = min(K for K in GK if GK[K] >= gmax * (1 - TOL))
            Kladder = max(MK, key=lambda K: MK[K])
            y = np.sort(YK[Kfree])[::-1]
            y = y[y > 1e-9]
            spread = (y.max() - y.min()) / y.max() if len(y) > 1 else 0.0
            print(f"{p:>4}{nu:>4}{2:>3} | {gmax:>12.9f} {Kfree:>16} {Kladder:>17} "
                  f"{spread:>13.4f}  {np.round(y, 6).tolist()}")
            rep.append(dict(p=p, nu=nu, q=2, Gamma=gmax, K_free=Kfree, K_ladder=Kladder,
                            spread=float(spread), shares=[float(t) for t in y],
                            support=[k + 1 for k in SK[Kfree]],
                            GammaK={str(K): GK[K] for K in GK},
                            ladderK={str(K): MK[K] for K in MK}))
    with open(os.path.join(OUT, "gamma2.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("\nwrote", os.path.join(OUT, "gamma2.json"))


if __name__ == "__main__":
    main()
