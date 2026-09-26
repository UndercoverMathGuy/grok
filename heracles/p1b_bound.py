"""P1B: the two-branch attack on P1 at n = 2, N = 4.

Notation (n = 2, four atoms, A = 1):
  v[l] = (v_l1, v_l2) in T^2       atom l's two slot phases
  c[l]                             its complex amplitude; (KL): sum_l c_l prod_{j in T} v_lj = [T empty]
  x_m = |c_m|  (m = 1,2,3),  eps = |c_4|    atom 4 is the DESIGNATED (smallest) atom
  d^{(j)}_{lm} = |v_lj - v_mj|     slot-j chordal distance between atoms l and m
  A^{(j)}_m    = |v_mj - v_4j|     slot-j distance from big atom m to the designated atom
  V_j          = prod_{l<m<=3} d^{(j)}_{lm}      slot-j Vandermonde of the three big atoms

BRANCH 1 (small designated atom).  The exact kill identity
    sum_l c_l prod_j (u_j - v_lj) = prod_j u_j           (all u in C^2; this IS (KL))
at u = (v_{m1,1}, v_{m2,2}) with m1 != m2 in {1,2,3} annihilates atoms m1 and m2, leaving
    c_{l0} P + c_4 Q = v_{m1,1} v_{m2,2},
    P = d^{(1)}_{m1,l0} d^{(2)}_{m2,l0},   Q = A^{(1)}_{m1} A^{(2)}_{m2},
so x_{l0} >= (1 - eps Q)/P.  Two bijections per l0, max >= geometric mean, then the product over
the three l0 turns prod P into V_1^2 V_2^2 (each ordered pair (m1,m2) occurs exactly once):

    prod_{m<=3} x_m  >=  Gplus^{1/2} / (V_1 V_2),
    Gplus = prod over the 6 ordered pairs (m1 != m2) of  max(1 - eps A^{(1)}_{m1} A^{(2)}_{m2}, 0).

AM-GM then gives   cost >= 3 * [Gplus^{1/2}/(V_1 V_2)]^{r/3} + eps^r  =  3 W^{-r/6} + eps^r,
with W := (V_1 V_2)^2 / Gplus a purely geometric quantity of the four slot configurations.
So branch 1 holds for every eps with   3 * maxW(eps)^{-r/6} + eps^r >= 3^{1-r}.

BRANCH 2 (no small atom).  |A| <= sum_l |c_l| gives sum_l x_l >= 1; if additionally every
|c_l| >= delta then minimising the concave sum_l x_l^r over the polyhedron
{x >= delta, sum x >= 1} (minimum at a vertex) gives  cost >= min((1-3d)^r + 3 d^r, 4 d^r).

If branch 1 covers eps <= eps0 and branch 2 covers eps >= eps1 with eps1 <= eps0, P1 is proved
at n = 2, N = 4.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

import p1b_core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1B")


# ---------------------------------------------------------------- the chain, verified

def chain_quantities(v, l4=3):
    """Return (Gplus, V1, V2, W) for the designated atom l4 at eps given separately."""
    big = [m for m in range(4) if m != l4]
    A = np.abs(v[big, :] - v[l4, :][None, :])  # (3, 2): A[i, j] = A^{(j)}_{big[i]}
    V = np.ones(2)
    for j in range(2):
        for i in range(3):
            for k in range(i + 1, 3):
                V[j] *= abs(v[big[i], j] - v[big[k], j])
    return A, V


def branch1_bound(v, eps, r, l4=3):
    """3 * W^{-r/6} + eps^r for one concrete configuration."""
    A, V = chain_quantities(v, l4)
    G = 1.0
    for i1 in range(3):
        for i2 in range(3):
            if i1 == i2:
                continue
            G *= max(1.0 - eps * A[i1, 0] * A[i2, 1], 0.0)
    VV = V[0] * V[1]
    if VV == 0.0:
        return None
    return 3.0 * (np.sqrt(G) / VV) ** (r / 3.0) + eps ** r


def verify_chain(trials=4000, seed=0, r=2.0 / 3.0):
    """Check prod_{m<=3} x_m >= Gplus^{1/2}/(V1 V2) and the whole branch-1 bound on
    random 4-atom (KL) systems (the square case: c is determined by v)."""
    rng = np.random.default_rng(seed)
    worst_prod = np.inf
    worst_cost = np.inf
    bad = 0
    used = 0
    for _ in range(trials):
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, (4, 2)))
        try:
            z = C.rule_from_config(v)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(z)):
            continue
        l4 = int(np.argmin(np.abs(z)))
        eps = abs(z[l4])
        x = np.array([abs(z[m]) for m in range(4) if m != l4])
        A, V = chain_quantities(v, l4)
        G = 1.0
        for i1 in range(3):
            for i2 in range(3):
                if i1 != i2:
                    G *= max(1.0 - eps * A[i1, 0] * A[i2, 1], 0.0)
        VV = V[0] * V[1]
        if VV < 1e-9:
            continue
        used += 1
        lhs = float(np.prod(x))
        rhs = np.sqrt(G) / VV
        if rhs > 0:
            worst_prod = min(worst_prod, lhs / rhs)
            if lhs < rhs * (1 - 1e-9):
                bad += 1
        b = branch1_bound(v, eps, r, l4)
        cost = float(np.sum(np.abs(z) ** r))
        if b is not None and b > 0:
            worst_cost = min(worst_cost, cost / b)
            if cost < b * (1 - 1e-9):
                bad += 1
    return dict(trials=trials, used=used, violations=bad,
                min_prod_ratio=worst_prod, min_cost_ratio=worst_cost)


# ---------------------------------------------------------------- max W over configurations

def negW(theta, eps):
    """-log W for slot angles theta = (phi1[3], phi2[3]); w^{(j)} = 1 in both slots."""
    ph1 = theta[:3]
    ph2 = theta[3:]
    A1 = 2.0 * np.abs(np.sin(ph1 / 2.0))
    A2 = 2.0 * np.abs(np.sin(ph2 / 2.0))
    u1 = np.exp(1j * ph1)
    u2 = np.exp(1j * ph2)
    V1 = abs(u1[0] - u1[1]) * abs(u1[0] - u1[2]) * abs(u1[1] - u1[2])
    V2 = abs(u2[0] - u2[1]) * abs(u2[0] - u2[2]) * abs(u2[1] - u2[2])
    if V1 <= 1e-14 or V2 <= 1e-14:
        return 1e6
    logG = 0.0
    for i1 in range(3):
        for i2 in range(3):
            if i1 == i2:
                continue
            t = 1.0 - eps * A1[i1] * A2[i2]
            if t <= 1e-14:
                return -1e6  # W -> infinity
            logG += np.log(t)
    return -(2.0 * np.log(V1 * V2) - logG)


def max_logW(eps, starts=400, seed=1):
    rng = np.random.default_rng(seed)
    best = -np.inf
    bestth = None
    # structured starts: both slots equally spaced with an atom antipodal to w
    base = []
    for sh1 in (0.0, np.pi / 3):
        for sh2 in (0.0, np.pi / 3):
            for perm in range(3):
                t = np.concatenate([
                    (np.pi + sh1 + 2 * np.pi * np.arange(3) / 3.0),
                    (np.pi + sh2 + 2 * np.pi * (np.arange(3) + perm) / 3.0)])
                base.append(t)
    for i in range(starts + len(base)):
        th0 = base[i] if i < len(base) else rng.uniform(0, 2 * np.pi, 6)
        res = minimize(negW, th0, args=(eps,), method="Nelder-Mead",
                       options=dict(maxiter=8000, xatol=1e-12, fatol=1e-14))
        res = minimize(negW, res.x, args=(eps,), method="Powell",
                       options=dict(maxiter=8000, xtol=1e-12, ftol=1e-14))
        val = -res.fun
        if val > best:
            best, bestth = val, res.x.copy()
    return best, bestth


def branch1_threshold(r, grid, starts=200, seed=1):
    """For each eps in grid, the proved branch-1 bound 3*maxW^{-r/6} + eps^r vs 3^{1-r}."""
    tgt = 3.0 ** (1.0 - r)
    rows = []
    for eps in grid:
        lw, th = max_logW(eps, starts=starts, seed=seed)
        bound = 3.0 * np.exp(-r * lw / 6.0) + eps ** r
        rows.append(dict(eps=float(eps), maxW=float(np.exp(lw)), bound=float(bound),
                         target=float(tgt), ok=bool(bound >= tgt),
                         argmax=[float(t % (2 * np.pi)) for t in th]))
    return rows


# ---------------------------------------------------------------- branch 2

def branch2_value(delta, r, N=4):
    """min sum_{l=1..N} x_l^r over {x_l >= delta, sum x_l >= 1}: concave, so a vertex."""
    cands = []
    if N * delta >= 1.0:
        cands.append(N * delta ** r)
    if 1.0 - (N - 1) * delta >= delta:
        cands.append((1.0 - (N - 1) * delta) ** r + (N - 1) * delta ** r)
    return min(cands) if cands else N * delta ** r


def branch2_threshold(r, N=4):
    tgt = (N - 1) ** (1.0 - r)
    lo, hi = 0.0, 1.0 / N
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if branch2_value(mid, r, N) >= tgt:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, nargs="+", default=[2.0 / 3.0])
    ap.add_argument("--starts", type=int, default=200)
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--tag", default="bound")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    out = {}

    out["verify_chain"] = verify_chain(trials=args.trials)
    print("chain verification:", json.dumps(out["verify_chain"], indent=1))

    for r in args.r:
        e1 = branch2_threshold(r)
        grid = np.round(np.arange(0.02, 0.201, 0.01), 4)
        rows = branch1_threshold(r, grid, starts=args.starts)
        ok = [row["eps"] for row in rows if row["ok"]]
        e0 = max(ok) if ok else 0.0
        key = "r=%.6f" % r
        out[key] = dict(branch2_eps1=e1, branch1_rows=rows, branch1_eps0_grid=e0,
                        closes=bool(e0 >= e1))
        print("\nr = %.6f   target = %.12f" % (r, 3 ** (1 - r)))
        print("  branch 2 covers  min|c| >= %.6f" % e1)
        for row in rows:
            print("  eps=%.3f  maxW=%.6g  bound=%.9f  %s"
                  % (row["eps"], row["maxW"], row["bound"], "OK" if row["ok"] else "--"))
        print("  branch 1 covers  min|c| <= %.4f (grid)   =>  closes: %s" % (e0, e0 >= e1))

    with open(os.path.join(OUT, args.tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
