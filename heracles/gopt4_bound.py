"""gopt4_bound.py -- the junk trade-off, settled numerically as one number per p.

Ingredients (both for pure single-frequency neurons, any number per frequency,
junk allowed -- the class in which (T) can be attained):

 (cost)   sum_{k,l} |c_lk|^r  >=  sum_k Cl2_r(eta_k) |G_k|^r,
          eta_k := sqrt( sum_{T != empty} |C_T^{(k)}|^2 ) / |G_k| ,
          Cl2_r the junk-relief function of gopt4_relief.py (MEASURED, not yet
          proved: the proved bound, agent 1's (C'), is the much weaker linear
          one  (n+1)^{1-r} (1 - (2^n - 1) eta)^r ).

 (margin) Lemma Q, per frequency and sign pattern, symmetrised over +-D:
          2^n margin <= min_{D != 0} [ sum_k |G_k| w_kD
                          - max_k |sin(theta_kD / 2)| max_T |C_T^{(k)}| ],
          w_kD = 1 - cos(2 pi k D / p),  |sin(theta/2)| = sqrt(w/2),
          and  max_T |C_T^{(k)}| >= eta_k |G_k| / sqrt(2^n - 1).

Putting them together, gamma_q / B is at most

  V(p) := max over x >= 0, eta >= 0 of
          min_{D != 0} [ sum_k x_k w_kD - (1/sqrt(2^n-1)) max_k eta_k x_k sqrt(w_kD/2) ]
          subject to   sum_k Cl2_r(eta_k) x_k^r  <=  1 ,

while (T) on this class asserts  gamma_q / B  <=  (n+1)^{-(1-r)/r} Gamma_{1/r}(p).
So  V(p) <= (n+1)^{-(1-r)/r} Gamma  CLOSES the junk gap on this class, and the
largest eta at which the two sides cross says how much junk the present pair of
tools can still not price.

All float64, local CPU.
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gopt4_relief import solve as relief_solve, subsets   # noqa: E402
from gopt4_trade import gmat, gamma_s                     # noqa: E402


def relief_table(n, q, etas, starts=24, seed=2):
    """Cl2_r(eta) / Cl2_r(0) on a grid, by the same SLSQP programme."""
    r = q / (n + 1)
    Ts = subsets(n)
    rng = np.random.default_rng(seed)
    tgt = (n + 1) ** (1 - r)
    vals = []
    for e in etas:
        v = min(relief_solve(N, n, r, Ts, 'l2', e, starts, rng)[0]
                for N in (n + 1, n + 2))
        vals.append(v / tgt)
    vals = np.minimum.accumulate(np.array(vals))   # enforce monotonicity
    return vals


def V_of_p(p, n, q, Cfun, etamax, nstart=250, seed=4):
    """the max defined in the module docstring."""
    r = q / (n + 1)
    g = gmat(p)                       # (nf, p-1), g[k-1, D-1] = 1 - cos(2pi kD/p)
    nf = g.shape[0]
    sq = np.sqrt(g / 2.0)             # |sin(theta_kD/2)|
    c2 = 1.0 / np.sqrt(2 ** n - 1)
    rng = np.random.default_rng(seed)

    def neg(z):
        x = z[:nf] ** 2
        eta = etamax / (1.0 + np.exp(-z[nf:]))       # eta in (0, etamax)
        cost = np.sum(Cfun(eta) * x ** r)
        if cost <= 0:
            return 0.0
        x = x / cost ** (1.0 / r)                     # scale onto cost = 1
        clean = g.T @ x                               # (p-1,)
        pen = c2 * np.max((eta * x)[:, None] * sq, axis=0)
        return -float(np.min(clean - pen))

    best, bz = -np.inf, None
    for s in range(nstart):
        z0 = np.concatenate([rng.standard_normal(nf),
                             rng.standard_normal(nf) * 2 - 3])
        if s < 3:
            z0 = np.concatenate([np.ones(nf), -12 * np.ones(nf)])
            z0[:nf] = rng.uniform(0.2, 1.2, nf)
        res = minimize(neg, z0, method='Nelder-Mead',
                       options={'maxiter': 60000, 'xatol': 1e-11,
                                'fatol': 1e-13})
        res = minimize(neg, res.x, method='Powell',
                       options={'maxiter': 60000, 'xtol': 1e-12,
                                'ftol': 1e-14})
        if -res.fun > best:
            best, bz = -res.fun, res.x
    eta = etamax / (1.0 + np.exp(-bz[nf:]))
    x = bz[:nf] ** 2
    x = x / np.sum(Cfun(eta) * x ** r) ** (1.0 / r)
    return best, x, eta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--p', type=int, nargs='+',
                    default=[11, 13, 17, 23, 29, 37, 41, 53, 71])
    ap.add_argument('--starts', type=int, default=150)
    ap.add_argument('--out', type=str, default='FINAL_RUNS/GOPT4/bound.json')
    args = ap.parse_args()
    n, q = args.n, args.q
    r = q / (n + 1)
    etamax = float(np.sqrt(2 ** n - 1))

    etas = np.concatenate([[0.0], np.geomspace(0.01, etamax, 26)])
    print(f"measuring Cl2_r on {len(etas)} junk levels, n={n} q={q} ...")
    tab = relief_table(n, q, etas)
    for e, v in zip(etas, tab):
        print(f"   eta={e:8.5f}  Cl2/Cl2(0) = {v:.10f}")

    def Cfun(e):
        return np.interp(np.clip(e, 0, etamax), etas, tab) * (n + 1) ** (1 - r)

    rows = []
    print(f"\n{'p':>5} {'Gamma':>13} {'target':>13} {'V(p)':>13} "
          f"{'V/target':>11}  argmax eta (nonzero x)")
    for p in args.p:
        G, _ = gamma_s(p, r, starts=300)
        target = (n + 1) ** (-(1 - r) / r) * G
        V, x, eta = V_of_p(p, n, q, Cfun, etamax, nstart=args.starts)
        sel = x > 1e-6 * x.max()
        print(f"{p:>5} {G:13.9f} {target:13.9f} {V:13.9f} {V/target:11.6f}  "
              + " ".join(f"{e:.3f}" for e in eta[sel]))
        rows.append({'p': p, 'Gamma': G, 'target': target, 'V': V,
                     'ratio': V / target,
                     'x': x.tolist(), 'eta': eta.tolist()})
    with open(args.out, 'w') as fh:
        json.dump({'n': n, 'q': q, 'etas': etas.tolist(),
                   'Cl2_rel': tab.tolist(), 'rows': rows}, fh, indent=1,
                  default=float)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
