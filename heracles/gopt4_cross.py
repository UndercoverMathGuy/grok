"""gopt4_cross.py -- the explicit junk threshold (JT), and how it depends on K.

Derivation (GLOBAL_OPT4 Sec 6.1).  For pure-neuron networks on a support of K
frequencies with relative l_2 junk eta_k <= eta_cap, the cost bound
sum_k Cl2_r(eta_k) x_k^r <= 1 and Lemma Q's margin penalty give, after the single
lossy step  max_k a_k >= (1/K) sum_k a_k  and the substitution
x_k = xtilde_k * phi(eta_k),  phi(eta) := (Cl2_r(0)/Cl2_r(eta))^{1/r},

   gamma_q / B  <=  (n+1)^{-(1-r)/r} Gamma_{nu/q}(p)      -- i.e. (T) holds --
   whenever      eta / (2 K sqrt(2^n - 1))  >=  1 - (Cl2_r(eta)/Cl2_r(0))^{1/r}
                 for every eta <= eta_cap.                                   (JT)

The left side is LINEAR in eta, the right side is QUADRATIC (Theorem J), so (JT)
holds up to an explicit crossing eta_cross(K) and fails above it.  Under the old,
linear, PROVED relief bound the right side would be linear too, with a slope
enormously larger than the left's -- which is exactly why the junk trade used to
look lost for every eta.

Note this threshold does not depend on p except through K, and it is
conservative: the max >= mean step is the only inequality spent.
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gopt4_relief import solve as relief_solve, subsets   # noqa: E402


def relief_curve(n, q, etas, starts=30, seed=2):
    r = q / (n + 1)
    Ts = subsets(n)
    rng = np.random.default_rng(seed)
    tgt = (n + 1) ** (1 - r)
    vals = [min(relief_solve(N, n, r, Ts, 'l2', float(e), starts, rng)[0]
                for N in (n + 1, n + 2)) / tgt for e in etas]
    return np.minimum.accumulate(np.array(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=str, nargs='+', default=['2:2', '2:2.5',
                                                             '3:2'])
    ap.add_argument('--Ks', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument('--starts', type=int, default=30)
    ap.add_argument('--out', type=str, default='FINAL_RUNS/GOPT4/cross.json')
    args = ap.parse_args()
    out = []
    for cell in args.cells:
        ns, qs = cell.split(':')
        n, q = int(ns), float(qs)
        r = q / (n + 1)
        emax = float(np.sqrt(2 ** n - 1))
        etas = np.concatenate([[0.0], np.linspace(0.01, emax, 40)])
        print(f"\n=== n={n} nu={n+1} q={q} r={r:.6f}   eta_max = sqrt(2^n-1) "
              f"= {emax:.6f}")
        rel = relief_curve(n, q, etas, starts=args.starts)
        # right-hand side of (JT)
        rhs = 1.0 - rel ** (1.0 / r)
        beta = (n + 1) ** 0 * (r * (1 - r) /
                               (1 - ((n - 1) / (n + 1)) ** 2 * (1 - r))) / (n + 1)
        print(f"   beta (Theorem J) = {beta:.10f};  "
              f"small-eta slope of RHS/eta^2 = {rhs[1]/etas[1]**2:.6f} "
              f"(predicted {beta/r:.6f})")
        row = {'n': n, 'q': q, 'r': r, 'beta': beta,
               'etas': etas.tolist(), 'rel': rel.tolist(), 'Ks': {}}
        for K in args.Ks:
            lhs = etas / (2.0 * K * np.sqrt(2 ** n - 1))
            ok = lhs >= rhs - 1e-15
            bad = np.nonzero(~ok)[0]
            ec = emax if len(bad) == 0 else float(etas[bad[0] - 1])
            row['Ks'][str(K)] = ec
            print(f"   K={K:<3d}  eta_cross = {ec:.4f}"
                  + ("  (holds for every eta)" if ec >= emax else
                     f"   [= {ec/emax:.3f} of the maximum possible junk]"))
        out.append(row)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
