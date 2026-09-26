"""P1C / step 4b: the HYBRID bound, evaluated EXACTLY.

Everything is a lower bound on F = sum_l |c_l|^r for a (KL) system with |A| = 1.
Sorted moduli x_1 >= x_2 >= ...;  S = the n+1 largest atoms;  s := x_{n+1};
R := sum of the moduli of the atoms outside S ("tail mass").

PROVED INGREDIENTS (each derived in P1C.md; (a) is new, (b) is agent 3's, (c) is elementary)

 (a)  sum_{l in S} x_l^r  >=  A_G(R) := (n+1)^{1-r} max(1 - 2^n R, 0)^r
 (b)  sum_{h < l <= n+1} x_l + R  >=  D_h(n),   h = 0..n
 (c)  sum_{l > n+1} x_l^r  >=  max( R^r , R s^{r-1} )        (every tail atom is <= s)

Therefore, for the true (R, s) of any configuration,

   F  >=  Val(R, s) := max( A_G(R), Lad(R, s) ) + max( R^r, R s^{r-1} ),
   Lad(R, s) := min { sum_{l=1}^{n+1} x_l^r : x_1 >= ... >= x_n >= x_{n+1} = s,  (b) } .

Lad is the minimum of a CONCAVE function over a polyhedron, so it sits at a vertex and is
computed here by exact vertex enumeration (no local optimiser).  The bound is then

   F  >=  Lhyb := min over (R, s) in the box of Val(R, s),

and the box is justified: s >= 1/(n+1) already gives F >= (n+1) s^r >= (n+1)^{1-r}, and
R >= R2(n, r) already gives F >= (n+1) s^r + R s^{r-1} >= [r(n+1)/(1-r)]^{1-r} R^r / r
>= (n+1)^{1-r}.  So only (R, s) in [0, R2] x [0, 1/(n+1)] can be dangerous.
"""
import argparse, json, os
from itertools import combinations
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")

S_D = {1: 1.0, 2: 2.0 / np.sqrt(3.0), 3: 9.0 / 7.0,
       4: 1.380900987, 5: 1.452990673, 6: 1.509590371}
KAPPA = {d: (1.0 - 2.0 ** (-d)) / S_D[d] for d in S_D}


def D_h(h, n):
    if h == 0:
        return 1.0
    best = 0.0

    def rec(rem, parts, prod):
        nonlocal best
        if len(parts) == h:
            if rem == 0:
                best = max(best, prod)
            return
        for d in range(1, rem - (h - len(parts) - 1) + 1):
            rec(rem - d, parts + [d], prod * KAPPA[d])
    rec(n, [], 1.0)
    return best


def A_G(R, n, r):
    return (n + 1) ** (1 - r) * max(1.0 - 2 ** n * R, 0.0) ** r


def R2(n, r):
    """Smallest R for which  (n+1) s^r + R s^{r-1} >= (n+1)^{1-r}  for every s > 0."""
    # min over s of (n+1)s^r + R s^{r-1} equals  [r(n+1)/(1-r)]^{1-r} R^r / r
    K = (r * (n + 1) / (1.0 - r)) ** (1.0 - r) / r
    return ((n + 1) ** (1 - r) / K) ** (1.0 / r)


def Lad(R, s, n, r):
    """min sum_{l=1}^{n+1} x_l^r over x_1 >= ... >= x_n >= s, x_{n+1} = s, with the
    ladder-with-tail constraints.  Exact: concave objective on a polyhedron -> vertex."""
    m = n                                   # free variables x_1..x_n
    rows, bs = [], []
    for h in range(0, n + 1):
        a = np.zeros(m)
        if h < n:
            a[h:] = 1.0
        rhs = D_h(h, n) - R - s             # the x_{n+1} = s term always counts
        rows.append(a); bs.append(rhs)
    for i in range(m - 1):                  # x_i >= x_{i+1}
        a = np.zeros(m); a[i] = 1.0; a[i + 1] = -1.0
        rows.append(a); bs.append(0.0)
    a = np.zeros(m); a[m - 1] = 1.0         # x_n >= s
    rows.append(a); bs.append(s)
    A = np.array(rows); b = np.array(bs)
    # the h = n row has all-zero coefficients: it is a feasibility test, not a constraint
    feas_rows = [k for k in range(len(A)) if not np.any(A[k])]
    for k in feas_rows:
        if b[k] > 1e-12:
            return np.inf                   # s + R < D_n(n): infeasible
    keep = [k for k in range(len(A)) if np.any(A[k])]
    A, b = A[keep], b[keep]
    best = np.inf
    for idx in combinations(range(len(A)), m):
        M = A[list(idx)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, b[list(idx)])
        if np.any(A @ x < b - 1e-9):
            continue
        if np.any(x < -1e-12):
            continue
        best = min(best, float(np.sum(np.maximum(x, 0.0) ** r)))
    if not np.isfinite(best):
        return np.inf
    return best + s ** r


def Val(R, s, n, r):
    lad = Lad(R, s, n, r)
    if not np.isfinite(lad):
        return np.inf
    tail = max(R ** r, R * s ** (r - 1.0)) if s > 0 else (R ** r if R == 0 else np.inf)
    return max(A_G(R, n, r), lad) + tail


def sweep(n, q, nR=401, nS=401):
    nu = n + 1
    r = q / nu
    target = nu ** (1 - r)
    Rmax = R2(n, r)
    smax = 1.0 / nu
    Rs = np.linspace(0.0, Rmax, nR)
    ss = np.linspace(0.0, smax, nS)
    best, arg = np.inf, None
    for R in Rs:
        for s in ss:
            v = Val(R, s, n, r)
            if v < best:
                best, arg = v, (float(R), float(s))
    return dict(n=n, q=q, r=r, target=target, Rmax=float(Rmax), smax=float(smax),
                Lhyb=float(best), ratio=float(best / target), arg=arg,
                ladder_alone=float(min(Lad(0.0, s, n, r) for s in ss)),
                gridR=nR, gridS=nS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2", "2:2.5", "3:2", "3:2.5", "3:3", "4:2"])
    ap.add_argument("--nR", type=int, default=401)
    ap.add_argument("--nS", type=int, default=401)
    ap.add_argument("--tag", default="hybrid2")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cell in a.cells:
        n, q = cell.split(":")
        n, q = int(n), float(q)
        rec = sweep(n, q, a.nR, a.nS)
        rows.append(rec)
        print(f"n={n} nu={n+1} q={q} r={rec['r']:.5f}  target={rec['target']:.9f}  "
              f"ladder alone={rec['ladder_alone']:.9f} ({rec['ladder_alone']/rec['target']:.6f})  "
              f"HYBRID={rec['Lhyb']:.9f}  ratio={rec['ratio']:.9f}  "
              f"argmin (R,s)={rec['arg']}  box=[0,{rec['Rmax']:.4f}]x[0,{rec['smax']:.4f}]",
              flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
