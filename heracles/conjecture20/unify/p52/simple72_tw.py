"""Float exploration: the simple (7,2) two-feature bound with the phase-blind decoupled-twisted payment curve
F(x) = max(1 - c_1 x, 3^{1-s} max(1,x), F_tw(x)),  F_tw(x) = min over [0,x]^3 x [-1,1] of pay_tw2 (tabulated, then
interpolated using the right end point, which is a valid lower bound because F_tw is non-increasing).
Usage: python simple72_tw.py s"""
import sys
import numpy as np
from scipy.optimize import linprog, minimize
from tw2 import pay_tw2

s = float(sys.argv[1]); r = 1 / s; p = 7
c1 = 1 - 3 ** (1 - s)
kap = lambda j, d: 1 - np.cos(2 * np.pi * j * d / p)
XS = np.linspace(0, 1.0, 26)


def Ftw_point(x, nL=120):
    best = np.inf
    g = np.linspace(0, x, 3)
    for b in g:
        for c in g:
            for d in g:
                for cp in (-1.0, 0.0, 1.0):
                    best = min(best, pay_tw2(1.0, b, c, d, cp, s, nL=nL))
    return best


TAB = np.array([Ftw_point(x) for x in XS])
TAB = np.minimum.accumulate(TAB)


def F(x):
    i = np.searchsorted(XS, x)          # right end point XS[i] >= x
    ftw = TAB[i] if i < len(XS) else 0.0
    return max(1 - c1 * x, 3 ** (1 - s) * max(1, x), ftw)


def num(A1, A2, M):
    c = np.array([2 * (kap(1, d) * A1 + kap(2, d) * A2) for d in (1, 2, 3)] + [-3.5 * M])
    Aub = np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]])
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(3), A_eq=np.array([[2, 2, 2, 0]]), b_eq=[1], bounds=[(0, None)] * 4, method='highs')
    return res.fun


G = 3.5 * 3 ** (-s)


def ratio(v):
    A2 = abs(v[0]); M1 = abs(v[1]); M2 = abs(v[2])
    if A2 < 1e-9:
        return 0
    return num(1.0, A2, max(M1, M2)) / (G * (F(M1) ** r + (F(M2 / A2) * A2) ** r) ** s)


best = (-1, None)
for A2 in np.linspace(0.2, 3, 15):
    for M1 in np.linspace(0, 1.6, 9):
        for M2 in np.linspace(0, 3, 16):
            v = ratio([A2, M1, M2])
            if v > best[0]:
                best = (v, [A2, M1, M2])
rng = np.random.default_rng(0)
starts = [best[1]] + [list(rng.uniform([0.2, 0, 0], [3, 1.6, 3])) for _ in range(20)]
for v0 in starts:
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 3000})
    if -res.fun > best[0]:
        best = (-res.fun, list(np.abs(res.x)))
print(f"s={s}: sup ratio with twisted payment curve = {best[0]:.5f} at A2/A1, M1, M2 = {np.round(best[1], 4)};  F_tw table (x=0,0.2,..,1): {np.round(TAB[::5], 4)}")
