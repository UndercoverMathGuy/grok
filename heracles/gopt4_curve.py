"""gopt4_curve.py -- the junk trade-off CURVE on a K >= 2 support.

The S2 obstruction, as inherited, is a comparison of two rates: junk of relative
size rho was believed to buy Theta(rho) of cost relief (robust peeling) against
a margin penalty of only O(rho / p) (Lemma Q), so on a spread support the trade
looked winning for junk.  gopt4_relief.py shows the cost relief is in fact
Theta(rho^2).  This script measures the resulting curve directly at the network
level: for each forced junk level rho it maximises

    ratio(rho) := Lambda / [ (n+1)^{-(1-r)/r} Gamma_{nu/q}(p) ]

over pure-neuron networks whose per-frequency relative junk
    rho_k := sqrt( sum_{T != empty} |C_T^{(k)}|^2 ) / |C_empty^{(k)}|
is pinned at rho.  ratio(0) = 1 is the (n+1)-block.  ratio(rho) <= 1 for every
rho is exactly statement S2 on this class; the shape of ratio near 0 says which
of the two rates wins.

All float64, local CPU.
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gopt4_trade import gamma_s, Net, block_z   # noqa: E402


def junk_profile(net, c, v):
    """(C_empty per k, l2 junk per k) from the atom data."""
    K, N, n = net.K, net.N, net.n
    # C_T = sum_l c_l prod_{j in T} v_lj  -- enumerate all 2^n subsets
    Cs = np.zeros((K, 2 ** n), dtype=complex)
    for t in range(2 ** n):
        pr = np.ones((K, N), dtype=complex)
        for j in range(n):
            if (t >> j) & 1:
                pr = pr * v[:, :, j]
        Cs[:, t] = (c * pr).sum(axis=1)
    C0 = Cs[:, 0]
    Jk = np.sqrt((np.abs(Cs[:, 1:]) ** 2).sum(axis=1))
    return C0, Jk


def run_cell(p, n, q, extra, rho_list, starts, seed, verbose=True):
    nu = n + 1
    r = q / nu
    rng = np.random.default_rng(seed)
    G, xstar = gamma_s(p, r)
    S = [int(k) for k in np.nonzero(xstar > 1e-8)[0] + 1]
    denom = (n + 1) ** (-(1 - r) / r) * G
    net = Net(p, n, S, n + 1 + extra)
    A = xstar[np.array(S) - 1] * (n + 1) ** (-(1 - r) / r)
    z_blk = block_z(net, A, rng)
    if verbose:
        print(f"p={p} q={q}  Gamma={G:.10f}  support={S} (K={len(S)})  "
              f"N/freq={net.N}  block ratio={net.ratio(z_blk, r, denom):.12f}")

    rows = []
    for rho in rho_list:
        best = -np.inf
        bestinfo = None
        for s in range(starts):
            z = z_blk + (0.0 if s == 0 else 10.0 ** (-1 - (s % 5))) \
                * rng.standard_normal(z_blk.size)
            if s >= starts // 2:
                z = np.concatenate([
                    rng.standard_normal(net.K * net.N) / (net.K * net.N),
                    rng.standard_normal(net.K * net.N) / (net.K * net.N),
                    rng.uniform(0, 2 * np.pi, size=net.K * net.N * n)])
            for beta, mu in ((30., 30.), (100., 100.), (400., 400.),
                             (2000., 2000.), (8000., 2000.)):
                def neg(zz):
                    c, v = net.split(zz)
                    H = net.hvals(c, v)
                    M = np.einsum('kd,ka->ad', net.W, H).real
                    m0 = M.min()
                    soft = m0 - np.log(np.exp(-beta * (M - m0)).sum()) / beta
                    cst = np.sum((np.abs(c) ** 2 + 1e-30) ** (r / 2))
                    C0, Jk = junk_profile(net, c, v)
                    pen = np.sum((Jk - rho * np.abs(C0)) ** 2) / \
                        max(np.sum(np.abs(C0) ** 2), 1e-12)
                    return -(soft / cst ** (1.0 / r)) / denom + mu * pen
                res = minimize(neg, z, method='L-BFGS-B',
                               options={'maxiter': 4000, 'ftol': 1e-16,
                                        'gtol': 1e-13})
                z = res.x

            # exact-ratio polish: the annealed soft-min systematically
            # UNDER-reports the true ratio, so finish on the exact objective
            # with the junk level still pinned by a hard penalty.
            def negexact(zz):
                c, v = net.split(zz)
                C0, Jk = junk_profile(net, c, v)
                pen = np.sum((Jk - rho * np.abs(C0)) ** 2) / \
                    max(np.sum(np.abs(C0) ** 2), 1e-12)
                return -net.ratio(zz, r, denom) + 5e4 * pen
            res = minimize(negexact, z, method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-11,
                                    'fatol': 1e-13})
            z = res.x
            c, v = net.split(z)
            C0, Jk = junk_profile(net, c, v)
            rho_ach = float(np.max(Jk / np.maximum(np.abs(C0), 1e-14)))
            val = net.ratio(z, r, denom)
            if abs(rho_ach - rho) < 0.15 * max(rho, 1e-3) + 1e-4 and val > best:
                best, bestinfo = val, rho_ach
        rows.append({'rho_target': rho, 'rho_achieved': bestinfo,
                     'ratio': best})
        if verbose:
            print(f"   rho={rho:<6.3g} achieved={bestinfo}  "
                  f"max ratio = {best:.12f}")
    return {'p': p, 'q': q, 'Gamma': G, 'support': S, 'N_per_freq': net.N,
            'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, nargs='+', default=[23])
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--q', type=float, default=2.0)
    ap.add_argument('--extra', type=int, default=1)
    ap.add_argument('--starts', type=int, default=16)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', type=str,
                    default='heracles/FINAL_RUNS/GOPT4/curve.json')
    args = ap.parse_args()
    rhos = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    out = [run_cell(p, args.n, args.q, args.extra, rhos, args.starts,
                    args.seed) for p in args.p]
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
