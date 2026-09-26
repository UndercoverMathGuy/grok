"""p3a_single.py -- (T) at MAXIMUM junk: one neuron per frequency, every support.

Why this family is the extreme case of P3.  A single pure neuron at frequency k emits
the clean cosine and all 2^n - 1 junk patterns with EQUAL amplitude, so its relative
junk is eta = sqrt(2^n - 1), the largest possible, and its cost is exactly |c_k|^r --
a factor (n+1)^{1-r} cheaper than the (n+1)-block that carries the same clean
amplitude junk-free.  In margin terms the neuron therefore starts with the whole
head-room the target (T) allows, namely

    phi_max = (n+1)^{(1-r)/r} = (n+1)^{(nu-q)/q}      ( = sqrt 3 at nu=3, q=2 ),

and (T) holds on this family if and only if the junk costs back all of it.  Agent 4's
gopt4_trade.py searched junk-carrying networks only on the Gamma-OPTIMAL support; this
script sweeps EVERY support (every K, greedy + random + exhaustive for small nf), which
is where the pressure actually is: the clean value of a K-frequency equal-share profile
relative to the target grows as K grows (it reaches 1.38 x target at p = 11 with the
full support), so those supports are the ones junk has to be shown to spoil.

For a one-atom-per-frequency network the profile is  C_T^{(k)} = c_k prod_{j in T} v_jk,
|C_T^{(k)}| = |c_k| for every T, and the exact cost is sum_k |c_k|^r.  So

    ratio := Lambda / (sum_k |c_k|^r)^{1/r} / [ (n+1)^{-(1-r)/r} Gamma_{nu/q}(p) ]

is an EXACT test of (T) on this family -- no relief curve, no Lemma Q, nothing
conditional.  Anything above 1 refutes (T).

Blocks
  pressure : for every support, the CLEAN value of the best equal-share/optimised
             profile divided by the target -- i.e. how much margin junk must destroy.
  search   : maximise ratio over one-atom-per-frequency networks.

Usage
    uv run python p3a_single.py --block pressure --p 11 13 17 19 23 29
    uv run python p3a_single.py --block search   --p 11 13 17 --starts 200

All float64, local CPU, no pods.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3a_prof import OUT, Profile, gamma_s, gmat, subsets  # noqa: E402


def clean_value(p, r, S):
    """max over shares of min_D sum_{k in S} x_k w_kD  s.t.  sum x^r = 1  (support fixed)."""
    g = gmat(p)[[k - 1 for k in S]]                      # (K, p-1)

    def neg(z):
        x = z ** 2
        s = np.sum(x ** r)
        if s <= 0:
            return 0.0
        return -float((g.T @ (x / s ** (1.0 / r))).min())

    best = 0.0
    for z0 in (np.ones(len(S)), np.linspace(0.5, 1.5, len(S))):
        res = minimize(neg, z0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14})
        res = minimize(neg, res.x, method='Powell',
                       options={'maxiter': 20000, 'xtol': 1e-13, 'ftol': 1e-15})
        best = max(best, -res.fun)
    return best


def supports(p, kmax_exhaustive=14):
    nf = (p - 1) // 2
    ks = list(range(1, nf + 1))
    if nf <= kmax_exhaustive:
        for K in range(1, nf + 1):
            for S in itertools.combinations(ks, K):
                yield list(S)
    else:
        rng = np.random.default_rng(0)
        for K in range(1, nf + 1):
            yield ks[:K]
            for _ in range(20):
                yield sorted(rng.choice(ks, K, replace=False).tolist())


def block_pressure(args):
    n, q = args.n, args.q
    r = q / (n + 1)
    fac = (n + 1) ** (-(1 - r) / r)
    rows = []
    for p in args.p:
        G, _ = gamma_s(p, r, starts=args.gstarts)
        target = fac * G
        best = {}
        for S in supports(p):
            v = clean_value(p, r, S)
            K = len(S)
            if K not in best or v > best[K][0]:
                best[K] = (v, S)
        print(f"\np={p}  Gamma_{{{1/r:.4g}}} = {G:.9f}  target = {target:.9f}"
              f"   (junk head-room phi_max = {(n+1)**((1-r)/r):.6f})")
        print(f"  {'K':>3} {'best clean value':>18} {'/target':>10}  support")
        for K in sorted(best):
            v, S = best[K]
            print(f"  {K:>3} {v:18.9f} {v/target:10.6f}  {S}")
            rows.append(dict(p=p, K=K, clean=v, ratio=v / target, S=S,
                             Gamma=G, target=target))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'pressure_n{n}_q{q}.json').write_text(json.dumps(rows, indent=1))
    print('\nwrote', OUT / f'pressure_n{n}_q{q}.json')


class Single:
    """one pure neuron per frequency: profile C_T^{(k)} = c_k prod_{j in T} v_jk."""

    def __init__(self, p, n, S):
        self.pr = Profile(p, n, S)
        self.n, self.K = n, len(S)
        self.Ts = self.pr.Ts

    def prof(self, z):
        K, n = self.K, self.n
        c = z[:K] + 1j * z[K:2 * K]
        v = np.exp(1j * z[2 * K:].reshape(K, n))
        C = np.zeros((K, len(self.Ts)), dtype=complex)
        for t, T in enumerate(self.Ts):
            w = np.ones(K, dtype=complex)
            for j in T:
                w = w * v[:, j]
            C[:, t] = c * w
        return c, C

    def ratio(self, z, r, denom):
        c, C = self.prof(z)
        cost = float(np.sum(np.abs(c) ** r))
        if cost <= 1e-14:
            return -1e9
        return (self.pr.Lam(C) / cost ** (1.0 / r)) / denom

    def soft(self, z, r, denom, tau):
        c, C = self.prof(z)
        cost = float(np.sum(np.abs(c) ** r))
        if cost <= 1e-14:
            return 1e9
        M = self.pr.M(C)
        mn = M.min()
        lam = mn - tau * np.log(np.exp(-(M - mn) / tau).sum())
        return -(lam / cost ** (1.0 / r)) / denom


def block_search(args):
    n, q = args.n, args.q
    r = q / (n + 1)
    fac = (n + 1) ** (-(1 - r) / r)
    rng = np.random.default_rng(args.seed)
    rows = []
    for p in args.p:
        G, xopt = gamma_s(p, r, starts=args.gstarts)
        target = fac * G
        nf = (p - 1) // 2
        allk = list(range(1, nf + 1))
        Sopt = [allk[i] for i in np.nonzero(xopt > 1e-7 * xopt.max())[0]]
        cands = [Sopt, allk]
        for K in range(2, nf + 1):
            cands.append(allk[:K])
        seen = set()
        print(f"\np={p}  Gamma={G:.9f}  target={target:.9f}")
        pbest = -np.inf
        for S in cands:
            key = tuple(S)
            if key in seen:
                continue
            seen.add(key)
            sg = Single(p, n, S)
            K = len(S)
            best = -np.inf
            for s in range(args.starts):
                z = np.concatenate([rng.uniform(0.3, 1.2, K),
                                    rng.standard_normal(K) * 0.1,
                                    rng.uniform(0, 2 * np.pi, K * n)])
                for tau in (0.5, 0.2, 0.08, 0.03, 0.01, 0.003):
                    res = minimize(sg.soft, z, args=(r, target, tau),
                                   method='Powell',
                                   options={'maxiter': 3000, 'maxfev': 6000,
                                            'xtol': 1e-8, 'ftol': 1e-10})
                    z = res.x
                res = minimize(lambda w: -sg.ratio(w, r, target), z,
                               method='Nelder-Mead',
                               options={'maxiter': 8000, 'maxfev': 12000,
                                        'xatol': 1e-10, 'fatol': 1e-12})
                best = max(best, sg.ratio(res.x, r, target))
            cl = clean_value(p, r, S) / target
            print(f"   S={str(S):<28s} K={K:2d}  clean/target={cl:8.5f}   "
                  f"best ratio = {best:.9f}  {'**REFUTES (T)**' if best > 1 + 1e-9 else ''}")
            rows.append(dict(p=p, S=S, K=K, clean_ratio=cl, ratio=best))
            pbest = max(pbest, best)
        print(f"   ---> p={p}: best over all supports = {pbest:.9f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'single_n{n}_q{q}.json').write_text(json.dumps(rows, indent=1))
    print('\nwrote', OUT / f'single_n{n}_q{q}.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', default='pressure', choices=['pressure', 'search'])
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--p', type=int, nargs='+', default=[11, 13, 17, 19, 23])
    ap.add_argument('--starts', type=int, default=60)
    ap.add_argument('--gstarts', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.block == 'pressure':
        block_pressure(args)
    else:
        block_search(args)


if __name__ == '__main__':
    main()
