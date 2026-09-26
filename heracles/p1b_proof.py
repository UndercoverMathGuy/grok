"""P1B: the two-branch proof of P1 (the sharp constant) for every number of atoms N.

THE CLAIM (P1).  For every n >= 1, every N, every unit-modulus v_lj and every c with
    (KL)   sum_l c_l prod_{j in T} v_lj = A [T empty]   for every T subset of {1..n},
one has   sum_l |c_l|^r >= (n+1)^{1-r} |A|^r,   r = q/nu in (0,1].

Normalise A = 1, write x_1 >= x_2 >= ... >= x_N >= 0 for the sorted moduli |c_l| and
    E := x_{n+2} + x_{n+3} + ... + x_N            ("the TAIL": everything past the n+1 largest).

BRANCH 1 (short tail).  Proposition B1 below:   cost >= (n+1)^{1-r} (1 - 2^n E)^r + E^r.
  This is the classical junk-robust Key Lemma (agent 1's (C')) applied with the tail playing the
  role of junk, PLUS the tail's own cost E^r -- which is the whole point: the relief the tail buys
  is LINEAR in E while the tail's own cost is E^r, and r < 1, so for small E the tail always
  loses.  Threshold eps0(n, r) := sup{ e : E^r >= (n+1)^{1-r} [1 - (1 - 2^n E)^r] on [0, e] }.

BRANCH 2 (long tail).  The peeling ladder of GLOBAL_OPT3 Theorem E is valid for every N:
    sum_{l > h} x_l >= D_h(n)   for h = 0..n,   D_h explicit (products of kappa_d = (1-2^-d)/S_d).
  Minimising the concave sum_l x_l^r over that polyhedron gives agent 3's L_r(n), which falls just
  SHORT of (n+1)^{1-r}.  Adding the single extra constraint  "tail >= delta"  lifts it: the ladder's
  minimising vertex uses exactly n+1 atoms, so a forced tail costs delta^r and saves only O(delta).
  Threshold eps1(n, r) := inf{ delta : L_r(n, delta) >= (n+1)^{1-r} }.

If eps1 <= eps0 the two branches cover every configuration and P1 follows.
"""

import argparse
import itertools
import json
import os

import numpy as np
from scipy.optimize import brentq, minimize

import p1b_core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1B")

# kappa_d = (1 - 2^-d)/S_d, S_d = max_{|z|=1} |1 - ((1+z)/2)^d|  (GLOBAL_OPT3 sec 3).
# d = 1,2,3 are exact:  kappa_1 = 1/2,  kappa_2 = 3 sqrt3/8,  kappa_3 = 49/72.
KAPPA = {1: 0.5, 2: 3.0 * np.sqrt(3.0) / 8.0, 3: 49.0 / 72.0,
         4: 0.678904577, 5: 0.666728299, 6: 0.652080868}


def D_h(n, h):
    """max over compositions d_1 + ... + d_h <= n of prod_i kappa_{d_i}  (D_0 = 1).

    NOTE (a correction to GLOBAL_OPT3 sec 3, which used  = n  rather than  <= n).  A kernel
    polynomial may be built on a PROPER subset J of the slots: Lambda_v(z_J) is still
    multi-affine in all n variables, still has constant coefficient 1, still vanishes at v
    (only z_J = v_J is needed) and its sup-norm is still S_{|J|}/(1 - 2^{-|J|}).  Since
    kappa_3 = 49/72 is the largest kappa_d, using fewer slots is often better:
    D_1(n) = 49/72 EXACTLY for every n >= 3, against agent 3's kappa_n (0.6789 at n = 4,
    0.6667 at n = 5).  This also makes every constant used here exact and algebraic
    (only kappa_1 = 1/2, kappa_2 = 3 sqrt3/8, kappa_3 = 49/72 appear)."""
    if h == 0:
        return 1.0
    best = 0.0
    for comp in _compositions(n, h):
        val = 1.0
        for d in comp:
            val *= KAPPA[d]
        best = max(best, val)
    return best


def _compositions(n, h):
    """all (d_1..d_h) with d_i >= 1 and sum d_i <= n.  Only d_i <= 3 is ever optimal:
    kappa_1 < kappa_2 < kappa_3 > kappa_4 > kappa_5 > ..., so replacing a part d_i >= 4 by 3
    both raises kappa_{d_i} and frees a slot.  Hence every D_h(n) is an exact algebraic number
    built from kappa_1 = 1/2, kappa_2 = 3 sqrt3/8, kappa_3 = 49/72 alone."""
    if h == 1:
        for d in range(1, min(n, 3) + 1):
            yield (d,)
        return
    for first in range(1, min(n - h + 1, 3) + 1):
        for rest in _compositions(n - first, h - 1):
            yield (first,) + rest


# ------------------------------------------------------------------ branch 1

def branch1_gap(E, n, r):
    """E^r - (n+1)^{1-r} [1 - (1 - 2^n E)^r]; branch 1 holds where this is >= 0."""
    t = 1.0 - (2 ** n) * E
    if t <= 0:
        return -np.inf
    return E ** r - (n + 1) ** (1.0 - r) * (1.0 - t ** r)


def branch1_threshold(n, r):
    """largest eps0 with branch1_gap >= 0 on the whole of (0, eps0]."""
    hi = 1.0 / 2 ** n
    lo = 1e-14
    # gap > 0 for small E (E^r beats the linear term); find the first sign change.
    f = lambda E: branch1_gap(E, n, r)
    grid = np.concatenate([np.geomspace(1e-12, hi * 1e-3, 4000),
                           np.linspace(hi * 1e-3, hi * 0.999999, 100000)])
    vals = np.array([f(E) for E in grid])
    neg = np.where(vals < 0)[0]
    if len(neg) == 0:
        return hi
    i = neg[0]
    if i == 0:
        return 0.0
    return float(brentq(f, grid[i - 1], grid[i], xtol=1e-16))


# ------------------------------------------------------------------ branch 2 (ladder programme)

def ladder_value(n, r, delta, N):
    """min sum_{l=1..N} x_l^r  s.t.  x_1 >= ... >= x_N >= 0,
       sum_{l>h} x_l >= D_h(n) for h = 0..n, and sum_{l>n+1} x_l >= delta.
    Concave objective on a pointed polyhedron -> minimum at a vertex; all vertices enumerated."""
    rows, rhs = [], []
    for l in range(N - 1):                    # x_l - x_{l+1} >= 0
        a = np.zeros(N); a[l] = 1.0; a[l + 1] = -1.0
        rows.append(a); rhs.append(0.0)
    a = np.zeros(N); a[N - 1] = 1.0           # x_N >= 0
    rows.append(a); rhs.append(0.0)
    for h in range(0, n + 1):                 # tail sums
        if h >= N:
            continue
        a = np.zeros(N); a[h:] = 1.0
        rows.append(a); rhs.append(D_h(n, h))
    if delta > 0 and n + 1 < N:
        a = np.zeros(N); a[n + 1:] = 1.0
        rows.append(a); rhs.append(delta)
    Aineq = np.array(rows); b = np.array(rhs)

    best = np.inf
    bestx = None
    for idx in itertools.combinations(range(len(b)), N):
        S = Aineq[list(idx)]
        if abs(np.linalg.det(S)) < 1e-12:
            continue
        x = np.linalg.solve(S, b[list(idx)])
        if np.any(x < -1e-10):
            continue
        x = np.maximum(x, 0.0)
        if np.any(Aineq @ x < b - 1e-9):
            continue
        val = float(np.sum(x ** r))
        if val < best:
            best, bestx = val, x
    return best, bestx


def branch2_threshold(n, r, N=None, tol=1e-12):
    if N is None:
        N = n + 4
    tgt = (n + 1) ** (1.0 - r)
    if ladder_value(n, r, 0.0, N)[0] >= tgt:
        return 0.0
    lo, hi = 0.0, 1.0 / (n + 2)
    while ladder_value(n, r, hi, N)[0] < tgt and hi < 0.5:
        hi *= 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ladder_value(n, r, mid, N)[0] >= tgt:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


# ------------------------------------------------------------------ numerical verification

def sample_rule(n, N, rng, noise=None):
    """One (KL) rule z on N atoms.  noise = None gives a uniformly random configuration;
    noise = s perturbs the (n+1)-block by angle s and adds random extra atoms, which is the
    only way to reach SMALL tails E (random configurations have E ~ O(1))."""
    if noise is None:
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
    else:
        vb = C.block(n) * np.exp(1j * rng.normal(0, noise, (n + 1, n)))
        extra = np.exp(1j * rng.uniform(0, 2 * np.pi, (max(N - n - 1, 0), n)))
        v = np.vstack([vb, extra])[:N]
    M = C.moment_matrix(v)
    rhs = np.zeros(M.shape[0], dtype=complex); rhs[0] = 1.0
    if N == M.shape[0]:
        try:
            z = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None
    else:
        z, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        if np.linalg.norm(M @ z - rhs) > 1e-9:
            return None
        K = C.nullspace(v)
        if K.shape[1] > 0 and rng.random() < 0.7:
            t = rng.normal(size=K.shape[1]) + 1j * rng.normal(size=K.shape[1])
            z = z + rng.choice([1e-3, 1e-2, 0.1, 0.4]) * K @ t / max(1.0, np.linalg.norm(t))
    return z if np.all(np.isfinite(z)) else None


def verify_branch1(n, N, trials=2000, r=2.0 / 3.0, seed=0):
    """Check on random (KL) systems:
       (a) the peeling ladder  sum_{l>h} x_l >= D_h(n)                (h = 0..n)
       (b) the product bound   prod_{l<=n+1} x_l >= (1 - 2^n E)^{n+1}/(n+1)^{n+1}
       (c) branch 1           cost >= (n+1)^{1-r}(1 - 2^n E)^r + E^r
       (d) P1 itself          cost >= (n+1)^{1-r}."""
    rng = np.random.default_rng(seed)
    worst_cost, worst_prod, worst_peel, worst_P1 = np.inf, np.inf, np.inf, np.inf
    used = act = viol = 0
    minE = np.inf
    modes = [None] + [s for s in (1e-3, 1e-2, 3e-2, 0.1, 0.3)]
    for i in range(trials):
        z = sample_rule(n, N, rng, modes[i % len(modes)])
        if z is None:
            continue
        x = np.sort(np.abs(z))[::-1]
        S = float(np.sum(x))
        if S > 1e8:
            continue
        used += 1
        cost = float(np.sum(x ** r))
        worst_P1 = min(worst_P1, cost / (n + 1) ** (1 - r))
        for h in range(0, min(n, N - 1) + 1):
            worst_peel = min(worst_peel, float(np.sum(x[h:])) / D_h(n, h))
        E = float(np.sum(x[n + 1:]))
        minE = min(minE, E)
        t = 1.0 - (2 ** n) * E
        if t > 0:
            act += 1
            bound = (n + 1) ** (1.0 - r) * t ** r + E ** r
            worst_cost = min(worst_cost, cost / bound)
            if cost < bound * (1 - 1e-9):
                viol += 1
            pb = t ** (n + 1) / (n + 1) ** (n + 1)
            prod = float(np.prod(x[:n + 1]))
            worst_prod = min(worst_prod, prod / pb)
            if prod < pb * (1 - 1e-9):
                viol += 1
    return dict(n=n, N=N, used=used, branch1_active=act, violations=viol, min_E=minE,
                min_peel_ratio=worst_peel, min_cost_over_branch1=worst_cost,
                min_prod_over_bound=worst_prod, min_cost_over_target=worst_P1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2", "3:2", "4:2", "5:2", "2:2.5", "3:3", "4:3"],
                    help="n:q pairs")
    ap.add_argument("--Nlad", type=int, default=0, help="atoms in the ladder programme (0 = n+4)")
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--tag", default="proof")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    out = {"cells": [], "verify": []}
    print("%-10s %-6s %-8s %-14s %-14s %-14s %-14s %s"
          % ("cell n:q", "r", "target", "L_r(delta=0)", "eps1 (branch2)", "eps0 (branch1)",
             "eps0/eps1", "CLOSES"))
    for cell in args.cells:
        ns, qs = cell.split(":")
        n = int(ns); q = float(qs); nu = n + 1; r = q / nu
        if r > 1:
            continue
        N = args.Nlad if args.Nlad else n + 4
        tgt = (n + 1) ** (1 - r)
        L0 = ladder_value(n, r, 0.0, N)[0]
        e1 = branch2_threshold(n, r, N)
        e0 = branch1_threshold(n, r)
        row = dict(n=n, q=q, r=r, N_ladder=N, target=tgt, L_r0=L0,
                   eps1=e1, eps0=e0, closes=bool(e1 <= e0))
        out["cells"].append(row)
        print("%-10s %-6.4f %-8.6f %-14.9f %-14.3e %-14.3e %-14.4g %s"
              % (cell, r, tgt, L0, e1, e0, (e0 / e1 if e1 > 0 else np.inf),
                 "YES" if e1 <= e0 else "no"))

    for (n, N) in [(2, 4), (2, 5), (2, 6), (3, 8), (3, 9), (1, 3)]:
        v = verify_branch1(n, N, trials=args.trials)
        out["verify"].append(v)
        print("verify n=%d N=%d: used=%d violations=%d  min cost/branch1=%.6f  min prod/bound=%.6f"
              % (n, N, v["used"], v["violations"], v["min_cost_over_branch1"], v["min_prod_over_bound"]))

    with open(os.path.join(OUT, args.tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()


# ------------------------------------------------------- the combined per-tail bound (Phi)

def M_r(n, r, eps):
    """min sum_{l=1}^{n+1} y_l^r  s.t.  y_1 >= ... >= y_{n+1} >= 0 and
       sum_{l=h+1}^{n+1} y_l >= D_h(n) - eps   (h = 0..n).

    This is the peeling ladder RESTRICTED to the top n+1 atoms of a profile whose tail has
    mass eps: sum_{l>h} x_l >= D_h and sum_{l>h} x_l = sum_{h<l<=n+1} x_l + eps.  It has n+1
    variables only, so unlike L_r it does not depend on N at all."""
    m = n + 1
    rows, rhs = [], []
    for l in range(m - 1):
        a = np.zeros(m); a[l] = 1.0; a[l + 1] = -1.0
        rows.append(a); rhs.append(0.0)
    a = np.zeros(m); a[m - 1] = 1.0
    rows.append(a); rhs.append(0.0)
    for h in range(0, n + 1):
        a = np.zeros(m); a[h:] = 1.0
        rows.append(a); rhs.append(max(D_h(n, h) - eps, 0.0))
    A = np.array(rows); b = np.array(rhs)
    best = np.inf
    for idx in itertools.combinations(range(len(b)), m):
        S = A[list(idx)]
        if abs(np.linalg.det(S)) < 1e-12:
            continue
        y = np.linalg.solve(S, b[list(idx)])
        if np.any(y < -1e-10):
            continue
        y = np.maximum(y, 0.0)
        if np.any(A @ y < b - 1e-9):
            continue
        best = min(best, float(np.sum(y ** r)))
    return best


def Phi(n, r, E):
    """cost >= Phi(E) for every (KL) system whose tail has mass E:
         Phi(E) = max( M_r(n,r,E), (n+1)^{1-r}(1 - 2^n E)_+^r ) + E^r.
    The two arguments of the max are two lower bounds on the SAME quantity, the cost of the
    top n+1 atoms: the ladder's and branch 1's."""
    b1 = (n + 1) ** (1.0 - r) * max(1.0 - 2 ** n * E, 0.0) ** r
    return max(M_r(n, r, E), b1) + E ** r


def combined_check(n, r, eps1, npts=4000):
    """P1 holds for every N as soon as Phi >= target on [0, eps1] (branch 2 covers E >= eps1)."""
    tgt = (n + 1) ** (1.0 - r)
    grid = np.concatenate([[0.0], np.geomspace(1e-9, eps1, npts)])
    vals = np.array([Phi(n, r, E) for E in grid])
    i = int(np.argmin(vals - tgt))
    return dict(ok=bool(np.all(vals >= tgt - 1e-12)),
                min_margin=float(np.min(vals - tgt)), at_E=float(grid[i]),
                rel_margin=float(np.min(vals) / tgt))
