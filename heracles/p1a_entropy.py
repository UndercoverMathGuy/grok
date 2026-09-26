r"""P1A: the ENTROPY route to S1 -- an r-free sufficient condition.

THE OBSERVATION (proved here, used as a reduction).
Fix a configuration and write x_l = |c_l| (|A| = 1), S = sum_l x_l >= 1,
q_l = x_l / S (a probability vector), and H(q) = - sum_l q_l log q_l its Shannon
entropy ("how spread the feature's amplitude is over its atoms").  Put

    h(r) := log( sum_l x_l^r )  -  (1 - r) log(n+1) ,

so S1 at exponent r is exactly h(r) >= 0.  Now

  * r |-> sum_l x_l^r is LOG-CONVEX (Cauchy-Schwarz), and (1-r) log(n+1) is
    affine, so h is CONVEX on [0, 1];
  * hence h lies above its tangent at r = 1:
        h(r) >= h(1) + (r - 1) h'(1) = r log S + (1 - r) ( H(q) - log(n+1) ).

  ==> if  H(q) >= log(n+1)  then h(r) >= 0 for EVERY r in [0,1] at once,
      because S >= 1.

So the r-free statement

    (H)   H(q) >= log(n+1)   for every (KL) system

would imply S1 for all r simultaneously.  At the (n+1)-block q = (1/(n+1),...,0)
and H = log(n+1) exactly, so (H) is sharp if true.

THIS SCRIPT TESTS (H) at n = 2, N = 4 (and n = 1..3, N = n+1..n+3 as a control),
and, when (H) fails, measures the two-term sufficient condition
    Suff(r) := r log S + (1-r)(H - log(n+1))  >= 0 .

Run:  uv run python p1a_entropy.py --starts 1200
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

from p1a_core import dependency, points

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1A")


# ---------------------------------------------------------------- n = 2, N = 4


def prof_n2N4(x):
    alpha = np.concatenate(([0.0], x[:3]))
    beta = np.concatenate(([0.0], x[3:]))
    a, b = points(alpha, beta)
    e, D = dependency(a, b)
    S = abs(np.sum(e * a * b))
    if S < 1e-13:
        return None
    return np.abs(D) / S


def entropy(p):
    s = p.sum()
    q = p / s
    q = q[q > 0]
    return float(-np.sum(q * np.log(q))), float(s)


def neg_entropy_n2N4(x):
    p = prof_n2N4(x)
    if p is None:
        return 10.0
    H, _ = entropy(p)
    return -(-H)  # minimise H  ->  return H


def min_entropy_n2N4(starts, seed):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(starts):
        x0 = rng.uniform(0, 2 * np.pi, 6)
        res = minimize(neg_entropy_n2N4, x0, method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-11, fatol=1e-14))
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


# ------------------------------------------------- general n, N: free-phase NLP
# (reuses the standard parametrisation: atoms c_l complex, phases v_lj on T^n,
#  constraints sum_l c_l prod_{j in T} v_lj = A [T empty].)


def general_min_entropy(n, N, starts=200, seed=0):
    """min H(q) over all (KL) systems with N atoms at n slots."""
    rng = np.random.default_rng(seed)
    subsets = []
    for mask in range(1 << n):
        subsets.append([j for j in range(n) if (mask >> j) & 1])

    def unpack(z):
        c = z[: 2 * N : 2] + 1j * z[1 : 2 * N : 2]
        ph = z[2 * N:].reshape(N, n)
        v = np.exp(1j * ph)
        return c, v

    def residual(z):
        c, v = unpack(z)
        out = []
        for T in subsets:
            val = np.sum(c * np.prod(v[:, T], axis=1)) if T else np.sum(c)
            tgt = 1.0 if not T else 0.0
            out += [val.real - tgt, val.imag]
        return np.array(out)

    def obj(z):
        c, _ = unpack(z)
        p = np.abs(c)
        if p.sum() <= 0:
            return 10.0
        H, _ = entropy(p)
        return H

    cons = [{"type": "eq", "fun": residual}]
    best = (np.inf, None)
    for _ in range(starts):
        z0 = np.concatenate((rng.normal(0, 0.5, 2 * N), rng.uniform(0, 2 * np.pi, N * n)))
        try:
            res = minimize(obj, z0, method="SLSQP", constraints=cons,
                           options=dict(maxiter=500, ftol=1e-12))
        except Exception:
            continue
        if not res.success or np.max(np.abs(residual(res.x))) > 1e-8:
            continue
        if res.fun < best[0]:
            c, _ = unpack(res.x)
            best = (float(res.fun), np.abs(c))
    return best


# ------------------------------------------------------- the two-term condition


def min_suff_n2N4(r, starts, seed):
    """min over configurations of  r log S + (1-r)(H - log 3)."""
    rng = np.random.default_rng(seed)

    def f(x):
        p = prof_n2N4(x)
        if p is None:
            return 10.0
        H, S = entropy(p)
        return r * np.log(S) + (1 - r) * (H - np.log(3.0))

    best = (np.inf, None)
    for _ in range(starts):
        x0 = rng.uniform(0, 2 * np.pi, 6)
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-11, fatol=1e-14))
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=600)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--blocks", nargs="*", default=["n2N4", "suff", "general", "rho"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rec = {}
    L3 = np.log(3.0)

    if "n2N4" in a.blocks:
        v, x = min_entropy_n2N4(a.starts, a.seed)
        p = prof_n2N4(x)
        H, S = entropy(p)
        print("(H) at n = 2, N = 4:  min over configurations of H(q)")
        print(f"   min H   = {v:.12f}")
        print(f"   log 3   = {L3:.12f}")
        print(f"   deficit = {v - L3:+.12f}   "
              f"{'(H) HOLDS so far' if v >= L3 - 1e-9 else '<-- (H) REFUTED'}")
        print(f"   profile |c| at the min = {np.array2string(np.sort(p)[::-1], precision=9)}")
        print(f"   S = sum|c| = {S:.9f},  sum|c|^(2/3) = {np.sum(p**(2/3)):.9f} "
              f"vs target {3**(1/3):.9f}")
        rec["n2N4_minH"] = dict(minH=v, log3=L3, deficit=v - L3,
                                profile=[float(t) for t in np.sort(p)[::-1]], S=S)

    if "suff" in a.blocks:
        rows = []
        for r in (2.0 / 3.0, 0.5, 5.0 / 6.0, 0.25):
            v, x = min_suff_n2N4(r, max(150, a.starts // 3), a.seed + 5)
            p = prof_n2N4(x)
            H, S = entropy(p)
            ok = v >= -1e-9
            print(f"   Suff(r={r:.4f}): min = {v:+.9f}  {'OK' if ok else '<-- fails'}"
                  f"   (at that config: H = {H:.6f}, S = {S:.6f},"
                  f" true h = {np.log(np.sum(p**r)) - (1-r)*L3:+.6f})")
            rows.append(dict(r=r, min_suff=float(v), H=H, S=S,
                             true_h=float(np.log(np.sum(p**r)) - (1 - r) * L3)))
        rec["suff"] = rows

    if "general" in a.blocks:
        print("\n  control: min H(q) over (KL) systems, general n and N")
        rows = []
        for n in (1, 2, 3):
            for N in (n + 1, n + 2, n + 3):
                out = general_min_entropy(n, N, starts=max(60, a.starts // 8), seed=a.seed + n * 10 + N)
                if out[1] is None:
                    print(f"    n={n} N={N}: no feasible point found")
                    continue
                tgt = np.log(n + 1)
                print(f"    n={n} N={N}: min H = {out[0]:.9f}  log(n+1) = {tgt:.9f}"
                      f"  deficit {out[0]-tgt:+.9f}"
                      f"  {'' if out[0] >= tgt - 1e-7 else '  <-- BELOW'}")
                rows.append(dict(n=n, N=N, minH=out[0], target=float(tgt),
                                 deficit=float(out[0] - tgt),
                                 profile=[float(t) for t in np.sort(out[1])[::-1]]))
        rec["general"] = rows

    if "rho" in a.blocks:
        rec["rho_scan"] = rho_scan(starts=max(80, a.starts // 4), seed=a.seed)

    with open(os.path.join(OUT, "entropy.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'entropy.json')}")




# ---------------------------------------------------------------------------
# the rho-scan: how much room does the entropy condition (E) have?
#
#   (E_rho)   H(q) + rho log S  >=  log(n+1),      rho := r/(1-r).
#
# (E_rho) at rho_0 implies S1 for EVERY r with r/(1-r) >= rho_0, i.e. for every
# r in [rho_0/(1+rho_0), 1].  Smaller rho = stronger statement.  This scan finds
# the smallest rho for which (E_rho) still holds numerically at n = 2, N = 4.
# ---------------------------------------------------------------------------


def min_E(rho, starts, seed, n=2):
    """min over configurations of  H(q) + rho log S - log(n+1)  at n = 2, N = 4."""
    rng = np.random.default_rng(seed)
    L = np.log(n + 1.0)

    def f(x):
        p = prof_n2N4(x)
        if p is None:
            return 10.0
        H, S = entropy(p)
        return H + rho * np.log(S) - L

    best = (np.inf, None)
    for _ in range(starts):
        x0 = rng.uniform(0, 2 * np.pi, 6)
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-11, fatol=1e-14))
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


def rho_scan(starts=150, seed=41):
    rows = []
    print("\n  rho-scan of (E_rho): H(q) + rho log S >= log 3   (n = 2, N = 4)")
    print("   rho   r = rho/(1+rho)    min over configurations    status")
    for rho in (10.0, 5.0, 4.88, 3.0, 2.0, 1.0, 0.5, 1 / 3, 0.2, 0.1, 0.05, 0.02):
        v, _ = min_E(rho, starts, seed + int(1000 * rho))
        ok = v >= -1e-9
        print(f"   {rho:6.3f}  {rho/(1+rho):.6f}      {v:+.10f}      "
              f"{'holds' if ok else 'FAILS'}")
        rows.append(dict(rho=float(rho), r=float(rho / (1 + rho)),
                         min=float(v), holds=bool(ok)))
    return rows


if __name__ == "__main__":
    main()
