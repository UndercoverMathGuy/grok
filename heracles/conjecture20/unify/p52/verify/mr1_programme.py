"""Math referee (mr): programme claims R1(i) [p=5] and R11(i) [p=7].

(a) The proofs in REPORT_R2 / tex use "by the power mean, sum_j y_j <= K^{1-s} when sum_j y_j^r <= 1".
    For r = 1/s < 1 this is FALSE (power mean runs the other way): y = e_1 gives sum y = 1 > K^{1-s}.
(b) Corrected proof: write y = (sum y) * yh with yh on the simplex; then sum y = (sum yh^r)^{-s} on the sphere,
    so  Phi_s(y) := min_d (K y)_d  <=  (S/2) K^{-s}  (S = sum_j kappa_j(d), d-independent)  iff
        Phi_1(yh) <= (S/K) * ... more precisely  Phi_1(yh) <= (S/2)/... see below
    For K features with constant row sum S:  (S) K^{-s} (sum yh^r)^s = S * M_r(yh) where M_r is the power mean.
    Since M_r >= M_{2/3} for r >= 2/3, it suffices to prove Phi_1(yh) <= S M_{2/3}(yh) on the simplex.
    Phi_1 is piecewise linear (min of linear), S M_{2/3} is concave, so on each linearity cell the difference
    S M_{2/3} - Phi_1 is concave and its minimum is at a vertex of the cell.  The cell vertices are intersections
    of two of the constraints {yh_j = 0} U {row_a = row_b} on the simplex.  We enumerate all of them in mpmath.
(c) Brute-force check on a fine grid of the z-simplex (y = z^s) at several s.
"""
import itertools
import numpy as np
import mpmath as mp

mp.mp.dps = 40


def kap(p, j, d):
    return 1 - mp.cos(2 * mp.pi * j * d / p)


def setup(p):
    F = list(range(1, (p - 1) // 2 + 1))
    D = list(range(1, (p - 1) // 2 + 1))     # delta and -delta give the same row
    K = mp.matrix(len(D), len(F))
    for a, d in enumerate(D):
        for b, j in enumerate(F):
            K[a, b] = kap(p, j, d)
    return F, D, K


def check_vertices(p):
    F, D, K = setup(p)
    m = len(F)
    S = sum(K[0, b] for b in range(m))
    # constraints as linear forms c . y = 0 (together with sum y = 1)
    cons = []
    for j in range(m):
        c = [mp.mpf(0)] * m; c[j] = mp.mpf(1); cons.append(('y%d=0' % (j + 1), c))
    for a, b in itertools.combinations(range(len(D)), 2):
        cons.append(('row%d=row%d' % (a + 1, b + 1), [K[a, k] - K[b, k] for k in range(m)]))
    pts = []
    for combo in itertools.combinations(cons, m - 1):
        A = mp.matrix(m, m); rhs = mp.matrix(m, 1)
        for i, (_, c) in enumerate(combo):
            for k in range(m):
                A[i, k] = c[k]
        for k in range(m):
            A[m - 1, k] = 1
        rhs[m - 1] = 1
        try:
            if abs(mp.det(A)) < mp.mpf(10) ** -30:
                continue
            y = mp.lu_solve(A, rhs)
        except ZeroDivisionError:
            continue
        if min(y) < -mp.mpf(10) ** -30:
            continue
        pts.append((tuple(n for n, _ in combo), [max(v, 0) for v in y]))
    # also the simplex vertices are covered (y_j = 0 for m-1 j's)
    worst = None
    out = []
    for names, y in pts:
        rows = [sum(K[a, k] * y[k] for k in range(m)) for a in range(len(D))]
        phi1 = min(rows)
        M23 = (sum(v ** (mp.mpf(2) / 3) for v in y) / m) ** (mp.mpf(3) / 2)
        diff = S * M23 - phi1
        out.append((names, [float(v) for v in y], float(phi1), float(S * M23), float(diff)))
        isuni = max(abs(v - mp.mpf(1) / m) for v in y) < mp.mpf(10) ** -25
        if not isuni:
            worst = diff if worst is None else min(worst, diff)
    return S, out, worst


def brute(p, s, N=600):
    F, D, K = setup(p)
    Kf = np.array(K.tolist(), dtype=float)
    m = len(F)
    best = -1; arg = None
    if m == 2:
        z1 = np.linspace(0, 1, 20001)
        Z = np.stack([z1, 1 - z1], 1)
    else:
        g = np.linspace(0, 1, N + 1)
        Z = np.array([(a, b, 1 - a - b) for a in g for b in g if a + b <= 1 + 1e-12])
        Z = np.clip(Z, 0, 1)
    Y = Z ** s
    vals = (Y @ Kf.T).min(axis=1)
    i = np.argmax(vals)
    return vals[i], Z[i], float(sum(Kf[0])) * float(m) ** (-s)


if __name__ == '__main__':
    # (a) the false inequality
    for p, K in ((5, 2), (7, 3)):
        for s in (1.25, 1.5):
            r = 1 / s
            y = np.zeros(K); y[0] = 1.0
            print(f"(a) p={p} s={s}: y=e1 has sum y^r = {np.sum(y**r):.3f} but sum y = {y.sum():.3f} > K^(1-s) = {K**(1-s):.4f}"
                  "  -> the 'power mean' step in R1(i)/R11(i) is false")
    # (b) corrected proof, vertex enumeration
    for p in (5, 7):
        S, out, worst = check_vertices(p)
        print(f"(b) p={p}: row sum S = {float(S):.12f};  cell vertices:")
        for names, y, phi1, rhs, diff in out:
            print(f"      {names}: y={np.round(y,6)}  Phi_1={phi1:.6f}  S*M_2/3={rhs:.6f}  diff={diff:.6f}")
        print(f"    min diff over non-uniform vertices = {float(worst):.6f}  (> 0 proves Gamma_s(p) = S K^-s with the unique uniform"
              f" maximiser for every s in [1, 3/2])")
    # (c) brute force
    for p in (5, 7):
        for s in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
            v, z, formula = brute(p, s)
            print(f"(c) p={p} s={s}: grid max = {v:.6f} at z={np.round(z,4)}   formula (S) K^-s = {formula:.6f}   ok={v <= formula + 1e-9}")
