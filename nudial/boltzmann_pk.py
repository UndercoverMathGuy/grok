"""Deterministic P(K): a Boltzmann weight on the margin ladder with a shaded
selection exponent.

  gamma_s(K; p) = M_K(p) / K^s,   r_s(K) = gamma_s(K) / max_J gamma_s(J)
  P(K) = exp(beta * r_s(K)) / Z        over K = 1..12

Two constants (beta, s). s = nu_sel/2 is the phenomenological selection
exponent (nudial_results round 8: nu_sel in (4.66, 4.83) places the argmax
at the observed mode at all four primes; the endpoint nu = 6 cannot, by the
monotone-w impossibility of seeded_ensemble_theory.md sec 3).
Protocol: fit (beta, s) by maximum likelihood on the p=113 pooled histogram
ONLY (base + wd0.3 + wd3 + f0.4 + f0.5 cells of P-N14, n=120, all
wd/frac-invariant), then PREDICT p97 / p127 / p211 frozen. Score = mean
-log P(K_obs) in nats vs the null (69-run vanilla histogram applied to every
cell, Laplace-smoothed) and the saturated per-cell floor. Also reports the
power-law variant P ~ r_s^beta, the joint 4-prime fit with leave-one-prime-
out CE, and the p=1009 prediction (P-N15 discriminator).
Histograms are the P-N14 cells recorded in notes/nudial_results.md round 6.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nudial"))
from kstar_table import best_subset

KMAX = 12
HIST = {  # notes/nudial_results.md round 6 (P-N14), grokked runs
    97:  {3: 5, 4: 18, 5: 1},
    113: {3: 4+5+5+4+7, 4: 15+15+15+13+13, 5: 5+4+4+7+4},   # base+wd0.3+wd3+f0.4+f0.5
    127: {3: 1, 4: 15, 5: 7, 6: 1},
    211: {3: 4, 4: 10, 5: 2},
}
NULL69 = {3: 11, 4: 45, 5: 11, 6: 2}   # vanilla 69-run baseline (round 5)


def MK(p):
    cache = ROOT / "notes" / "kstar_pscale.json"
    d = json.loads(cache.read_text()) if cache.exists() else {}
    if str(p) in d and all(str(K) in d[str(p)] for K in range(1, KMAX + 1)):
        return np.array([d[str(p)][str(K)]["M"] for K in range(1, KMAX + 1)])
    return np.array([best_subset(p, K)[0] for K in range(1, KMAX + 1)])


def counts(h):
    c = np.zeros(KMAX); [c.__setitem__(k - 1, n) for k, n in h.items()]; return c


def logP(M, beta, s, power=False):
    K = np.arange(1, KMAX + 1); g = M / K ** s; r = g / g.max()
    lw = beta * np.log(np.clip(r, 1e-300, None)) if power else beta * r
    lw -= lw.max(); return lw - np.log(np.exp(lw).sum())


def ce(M, c, beta, s, power=False):
    return float(-(c * logP(M, beta, s, power)).sum() / c.sum())


def fit(Ms, cs, power=False):
    def obj(x):
        return sum(-(c * logP(M, x[0], x[1], power)).sum() for M, c in zip(Ms, cs))
    best = None
    for b0 in (5, 20, 60):
        for s0 in (2.0, 2.4, 2.8):
            r = minimize(obj, [b0, s0], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=4000))
            if best is None or r.fun < best.fun: best = r
    return best.x


def sat(c):
    q = c / c.sum(); return float(-(c[c > 0] * np.log(q[c > 0])).sum() / c.sum())


def null_ce(c):
    n = counts(NULL69) + 0.5; q = n / n.sum(); return float(-(c * np.log(q)).sum() / c.sum())


M = {p: MK(p) for p in (97, 113, 127, 211, 1009)}
C = {p: counts(h) for p, h in HIST.items()}
for p in M:
    print(f"p={p}: M_K = {np.round(M[p], 3).tolist()}")
res = {}
for power in (False, True):
    tag = "power-law r^beta" if power else "Boltzmann exp(beta r)"
    beta, s = fit([M[113]], [C[113]], power)
    print(f"\n== {tag}: fit on p113 pooled (n={int(C[113].sum())}) -> beta={beta:.2f} s={s:.3f} (nu_sel={2*s:.2f})")
    print(f"{'p':>5} {'n':>4} {'CE_model':>9} {'CE_null':>8} {'CE_sat':>7}  mode_pred  P(3/4/5/6)")
    rr = {}
    for p in (97, 113, 127, 211):
        lp = logP(M[p], beta, s, power); P = np.exp(lp)
        rr[p] = dict(ce=ce(M[p], C[p], beta, s, power), null=null_ce(C[p]), sat=sat(C[p]), mode=int(P.argmax() + 1), P=P.tolist())
        print(f"{p:5d} {int(C[p].sum()):4d} {rr[p]['ce']:9.3f} {rr[p]['null']:8.3f} {rr[p]['sat']:7.3f}  {rr[p]['mode']:9d}  {np.round(P[2:6], 3).tolist()}")
    P9 = np.exp(logP(M[1009], beta, s, power))
    print(f" p1009 frozen prediction: mode {P9.argmax()+1}, P(4/5/6) = {np.round(P9[3:6], 3).tolist()}")
    # joint fit + leave-one-prime-out
    bj, sj = fit([M[p] for p in HIST], [C[p] for p in HIST], power)
    loo = {}
    for p in HIST:
        bl, sl = fit([M[q] for q in HIST if q != p], [C[q] for q in HIST if q != p], power)
        loo[p] = ce(M[p], C[p], bl, sl, power)
    print(f" joint fit: beta={bj:.2f} s={sj:.3f} (nu_sel={2*sj:.2f}); LOO CE: " + ", ".join(f"p{p} {v:.3f}" for p, v in loo.items())
          + f" | pooled LOO {np.average(list(loo.values()), weights=[C[p].sum() for p in HIST]):.3f} vs null {np.average([null_ce(C[p]) for p in HIST], weights=[C[p].sum() for p in HIST]):.3f}")
    res[tag] = dict(beta_p113=beta, s_p113=s, per_prime=rr, p1009=P9.tolist(), joint=dict(beta=bj, s=sj), loo_ce=loo)
(ROOT / "notes" / "boltzmann_pk.json").write_text(json.dumps(res, indent=1))
print("wrote notes/boltzmann_pk.json")
