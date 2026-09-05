"""Theory-only P(K) and gamma band. NOTHING fitted to run data.

Ladder: gamma_hat(K) = M_K(p) / K^(nu/2) (equal-share closed form), r(K) = gamma_hat(K)/max.
Width: the theory's own resolution delta = max over K in {K*-1, K*, K*+1} of |gamma_free(K)/gamma_eq(K) - 1|, the
gap between the equal-share ladder and the free-amplitude optimum on the same support
(construct_gamma.json for relu4/relu6; kstar_table.unequal_mass for the pure ladder).
Two committees whose r differ by less than delta are indistinguishable BY THE THEORY, so
    P(K) = exp(-(1 - r(K)) / delta) / Z          over K = 1..12.
Predicted count: mode and the smallest set with >= 90% mass. Predicted gamma: the
band [min, max] of gamma_hat over that 90% set (relative to the ceiling).
Error metric reported: coverage = fraction of observed runs inside the predicted 90%
set, and the predicted-vs-observed relative gamma. Observed histograms are used ONLY
to score coverage, never to set anything.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "nudial"))
from kstar_table import best_subset, unequal_mass
KMAX = 12
OBS = {("relu4", 113): {4: 2, 5: 16, 6: 11, 7: 3}, ("relu6", 113): {3: 5, 4: 18, 5: 9},
       ("transformer nu6", 113): {3: 25, 4: 71, 5: 24}, ("transformer nu6", 97): {3: 5, 4: 18, 5: 1},
       ("transformer nu6", 127): {3: 1, 4: 15, 5: 7, 6: 1}, ("transformer nu6", 211): {3: 4, 4: 10, 5: 2}}
cg = json.loads((ROOT / "notes" / "construct_gamma.json").read_text())
def ladder(p, nu):
    M = [best_subset(p, K)[0] for K in range(1, KMAX + 1)]; S = [best_subset(p, K)[1] for K in range(1, KMAX + 1)]
    g = np.array([M[K - 1] / K ** (nu / 2) for K in range(1, KMAX + 1)])
    gfree = np.array([unequal_mass(p, S[K - 1], nu) for K in range(1, KMAX + 1)])
    return g, gfree
out = {}
for fam, nu in (("relu4", 4), ("relu6", 6), ("transformer nu6", 6)):
    for p in (113,) if fam != "transformer nu6" else (113, 97, 127, 211):
        g, gfree = ladder(p, nu)
        if fam in ("relu4", "relu6") and p == 113:      # constructive resolution in the actual architecture
            c = cg[str(nu)]; ks = max(c, key=lambda K: c[K]["gamma_eq"])
            delta = max(c[K]["gamma_free"] / c[K]["gamma_eq"] - 1 for K in c if abs(int(K) - int(ks)) <= 1)
            src = "construct_gamma free/eq"
        else:
            ks = int(np.argmax(g)); lo, hi = max(0, ks - 1), min(KMAX - 1, ks + 1)
            delta = float(np.max(gfree[lo:hi + 1] / g[lo:hi + 1] - 1)); src = "kstar_table free/eq at K*-1..K*+1"
        r = g / g.max(); P = np.exp(-(1 - r) / delta); P /= P.sum()
        order = np.argsort(-P); cum = np.cumsum(P[order]); set90 = sorted((order[: int(np.searchsorted(cum, 0.9)) + 1] + 1).tolist())
        obs = OBS.get((fam, p)); n = sum(obs.values()) if obs else 0
        cov = sum(v for k, v in obs.items() if k in set90) / n if obs else None
        Kobs = sum(k * v for k, v in obs.items()) / n if obs else None
        EK = float((np.arange(1, KMAX + 1) * P).sum()); sdK = float(np.sqrt(((np.arange(1, KMAX + 1) - EK) ** 2 * P).sum()))
        gband = [float(r[K - 1]) for K in set90]
        out[f"{fam} p{p}"] = dict(nu=nu, delta=delta, delta_source=src, P={K: round(float(P[K - 1]), 3) for K in range(1, KMAX + 1) if P[K - 1] > 1e-3},
                                  mode=int(P.argmax() + 1), E_K=EK, sd_K=sdK, set90=set90, gamma_rel_band=[min(gband), max(gband)], coverage90=cov, obs_mean_K=Kobs)
        print(f"{fam:16s} p={p:4d} nu={nu}: delta={delta:.3f} ({src}) | P(K) {out[f'{fam} p{p}']['P']} | mode {P.argmax()+1}, E[K]={EK:.2f}+-{sdK:.2f}, 90% set {set90}, gamma/ceiling band [{min(gband):.3f},{max(gband):.3f}]"
              + (f" | OBS mean K {Kobs:.2f}, coverage of 90% set {cov:.2f}" if obs else ""))
(ROOT / "notes" / "pk_theory.json").write_text(json.dumps(out, indent=1))
