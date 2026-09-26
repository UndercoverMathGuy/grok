r"""P1A: the r-free product route for S1 at n = 2, N = 4.

At N = n+1 = 3 the Key Lemma is proved r-free: the bijection identity plus
Hadamard give  prod_l |c_l| >= (|A|/3)^3, and AM-GM then yields
sum_l |c_l|^r >= 3^{1-r}|A|^r for every r in (0,1] at once.

The natural N = 4 analogue, tested here:

  (P3)  the product of the THREE LARGEST |c_l| is at least (|A|/3)^3.

(P3) would imply S1 at n=2, N=4 for every r at once, since
sum_l |c_l|^r >= sum over the three largest >= 3 (prod of the three largest)^{r/3}
>= 3 (1/27)^{r/3} |A|^r = 3^{1-r}|A|^r.

Also tested, as fallbacks:
  (P3k) prod of three largest >= kappa (|A|/3)^3 for the measured best kappa;
  (Pw)  a weighted version  x_(1)^{w1} x_(2)^{w2} x_(3)^{w3} >= 3^{-(w1+w2+w3)};
  (Pall) prod over ALL FOUR of max(x_l, x_(3)) etc.

Run:  uv run python p1a_product.py --starts 4000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

from p1a_core import bigM, dependency, points

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1A")


def profile(x):
    """sorted (descending) |c_l| with |A| = 1, from the 6 gauge-fixed angles."""
    alpha = np.concatenate(([0.0], x[:3]))
    beta = np.concatenate(([0.0], x[3:]))
    a, b = points(alpha, beta)
    e, D = dependency(a, b)
    S = abs(np.sum(e * a * b))
    if S < 1e-13:
        return None
    return np.sort(np.abs(D) / S)[::-1]


def top3_product(x):
    p = profile(x)
    if p is None:
        return 1e6
    return float(p[0] * p[1] * p[2])


def powersum(x, r):
    p = profile(x)
    if p is None:
        return 1e6
    return float(np.sum(p ** r))


def minimise(fun, starts, seed, method="Nelder-Mead"):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(starts):
        x0 = rng.uniform(0, 2 * np.pi, 6)
        res = minimize(fun, x0, method=method,
                       options=dict(maxiter=4000, xatol=1e-11, fatol=1e-14))
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tgt = 1.0 / 27.0
    v, x = minimise(top3_product, a.starts, a.seed)
    p = profile(x)
    print("(P3)  min over configurations of  x_(1) x_(2) x_(3)")
    print(f"   minimum        = {v:.12f}")
    print(f"   target 1/27    = {tgt:.12f}")
    print(f"   ratio          = {v / tgt:.12f}   <-- < 1 REFUTES (P3)")
    print(f"   profile at min = {np.array2string(p, precision=9)}")
    print(f"   sum x^(2/3)    = {np.sum(p ** (2/3)):.12f}  vs target {3 ** (1/3):.12f}"
          f"  (ratio {np.sum(p ** (2/3)) / 3 ** (1/3):.9f})")

    rec = dict(P3_min=v, P3_ratio=v / tgt, P3_profile=[float(q) for q in p])

    # If (P3) fails, how far does a weighted product get?  Minimise
    # w1 log x1 + w2 log x2 + w3 log x3 over configurations for a few weights.
    rows = []
    for w in [(1, 1, 1), (1.2, 1, 0.8), (1.5, 1, 0.5), (0.8, 1, 1.2), (0.5, 1, 1.5),
              (0, 1, 1), (1, 1, 0)]:
        w = np.array(w, dtype=float)

        def fw(x, w=w):
            p = profile(x)
            if p is None:
                return 1e6
            return float(np.sum(w * np.log(np.maximum(p[:3], 1e-300))))

        vv, xx = minimise(fw, max(200, a.starts // 6), a.seed + 1)
        lhs = vv
        rhs = -np.sum(w) * np.log(3.0)
        rows.append(dict(w=[float(q) for q in w], min_log=float(lhs),
                         block_log=float(rhs), slack=float(lhs - rhs)))
        print(f"   weights {w}: min sum w log x_(i) = {lhs:.9f}, block = {rhs:.9f}, "
              f"slack {lhs - rhs:+.9f}  {'OK' if lhs >= rhs - 1e-9 else 'FAILS'}")
    rec["weighted"] = rows

    with open(os.path.join(OUT, "product.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'product.json')}")


if __name__ == "__main__":
    main()
