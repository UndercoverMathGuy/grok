"""Exact certificate for the flip-enhanced transfer constant c(A*, A_s) (VERIFY.md 'New lemma').
Coordinates m = (m1, m2, m3) = moduli of the direction on patterns {1}, {2}, {1,2} at one frequency.
beta_sigma(m) = max( A_in x + A_out |y - A_in w| , A_out (y + A_in w) - A_in x ), roles (outer y, full x, coupled w).
F_lam(m) = sum_sigma lam_sigma beta_sigma(m) is linear on each cell of the line arrangement (all breakpoint lines) of the
simplex {m >= 0, sum m = 1}; F - c|m|_2 is concave on each cell, so F >= c|m|_2 on the simplex iff it holds at all
arrangement vertices. Exact rational arithmetic; the final comparison is F^2 >= c^2 |m|^2.
"""
from fractions import Fraction as Fr
import itertools
import numpy as np
from scipy.optimize import linprog

IDX = {'1': 0, '2': 1, 'F': 2}
ORDERS = [('1', 'F', '2', 'flip'), ('1', '2', 'F', 'flip'), ('2', 'F', '1', 'flip'), ('2', '1', 'F', 'flip'),
          ('F', '1', '2', 'plain'), ('F', '2', '1', 'plain'),
          ('1', None, None, 'pure'), ('2', None, None, 'pure'), ('F', None, None, 'pure')]
# 'pure' orders: aim at one junk pattern only (other slot / the difference a - b uniform), two-sided via the flip that
# negates that pattern; bound = A_s * |z_{k,T}| per frequency.

def params(order, As, Ao):
    o, f, c, kind = order
    if kind == 'pure':
        return IDX[o], None, None, As, None
    Ai, Aout = (As, Ao) if kind == 'flip' else (Ao, Ao)
    return IDX[o], IDX[f], IDX[c], Ai, Aout

def beta(m, order, As, Ao):
    o, f, c, Ai, Aout = params(order, As, Ao)
    if order[3] == 'pure':
        return Ai * m[o]
    x, y, w = m[f], m[o], m[c]
    return max(Ai * x + Aout * abs(y - Ai * w), Aout * (y + Ai * w) - Ai * x)

def lines(As, Ao):
    L = [(Fr(1), Fr(0), Fr(0)), (Fr(0), Fr(1), Fr(0)), (Fr(0), Fr(0), Fr(1))]    # m_i = 0
    for order in ORDERS:
        if order[3] == 'pure': continue
        o, f, c, Ai, Aout = params(order, As, Ao)
        for coef in ({o: Fr(1), c: -Ai}, {f: Fr(1), c: -Aout}, {f: Ai, o: -Aout}):
            v = [Fr(0)] * 3
            for i, a in coef.items(): v[i] += a
            L.append(tuple(v))
    return L

def vertices(As, Ao):
    V = set()
    Ls = lines(As, Ao)
    for l1, l2 in itertools.combinations(Ls, 2):
        M = [list(l1), list(l2), [Fr(1)] * 3]; rhs = [Fr(0), Fr(0), Fr(1)]
        # solve 3x3 exactly (Cramer)
        def det(A): return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
        D = det(M)
        if D == 0: continue
        sol = []
        for j in range(3):
            Mj = [row[:] for row in M]
            for i in range(3): Mj[i][j] = rhs[i]
            sol.append(det(Mj) / D)
        if all(s >= 0 for s in sol):
            V.add(tuple(sol))
    return sorted(V)

def certify(As, Ao, target):
    As, Ao, target = Fr(As), Fr(Ao), Fr(target)
    V = vertices(As, Ao)
    B = np.array([[float(beta(v, o, As, Ao)) for o in ORDERS] for v in V])
    nrm = np.array([np.sqrt(float(sum(x * x for x in v))) for v in V])
    k = len(ORDERS)
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.c_[-B, nrm], b_ub=np.zeros(len(V)),
                A_eq=[np.r_[np.ones(k), 0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)])
    lam = [Fr(float(x)).limit_denominator(10 ** 6) for x in r.x[:k]]
    lam = [max(l, Fr(0)) for l in lam]; s = sum(lam); lam = [l / s for l in lam]
    # exact min of F/|m| over vertices (report), and exact check against target
    worst = None; ok = True
    for v in V:
        F = sum(l * beta(v, o, As, Ao) for l, o in zip(lam, ORDERS))
        n2 = sum(x * x for x in v)
        ratio = float(F) / float(n2) ** 0.5
        worst = ratio if worst is None else min(worst, ratio)
        if not (F > 0 and F * F > target * target * n2):
            ok = False
    return worst, ok, lam, len(V)

if __name__ == '__main__':
    import sys
    As, Ao = float(sys.argv[1]), float(sys.argv[2])
    w, ok, lam, nv = certify(As, Ao, 0.272)
    print(f"A_s={As} A*={Ao}: exact min over {nv} vertices = {w:.6f}; > 0.272 certified: {ok}")
