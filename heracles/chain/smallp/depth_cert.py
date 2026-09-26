"""Rigorous lower bounds on the one-sided (A*) and two-sided (A_s) K-mode depths of a frequency set T at prime p.

Body: C = conv{E_a : a in Z_p}, E_a = (w^{-2ka})_{k in T}  (two-sided: conv of E_a and -E_a).
Polar: P = {z : Re sum_k z_k E_ak <= 1 for all generators}. depth = 1 / max_{z in P} sum_k |z_k|.
Grid bound (HOLES_aiming sec 2): for unimodular grid directions d, U_d = max_{z in P} Re sum conj(d_k) z_k; then
depth >= cos(pi/L) / max_d U_d, where the grid either covers all K phases (L arbitrary) or fixes d_1 = 1 using the
cyclic row shift (valid only when L <= p; the shift rotates z_1 within pi/p <= pi/L).
Each U_d is bounded ABOVE rigorously from an approximate LP dual y >= 0:
  Re<d,z> = sum_a y_a H_a.x + (c - H^T y).x <= sum y_a + 2 ||c - H^T y||_1,
using |Re z_k|, |Im z_k| <= 2 on P (1 - f_z >= 0 has mean 1, so each Fourier coefficient z_k/2 has modulus <= 1).
The residual is evaluated in mpmath interval arithmetic with y converted exactly; cos/sin are interval-enclosed.
"""
import itertools, sys
import numpy as np
from scipy.optimize import linprog
from mpmath import iv, mpf
iv.dps = 30

def generators(p, T, sym):
    x = np.arange(p)
    E = np.exp(-2j * np.pi * 2 * np.outer(T, x) / p)            # K x p
    if sym:
        E = np.hstack([E, -E])
    return E

def iv_generators(p, T, sym):
    # interval enclosures of Re/Im of generators, as lists [K][n]
    re, im = [], []
    for k in T:
        rr, ii = [], []
        for a in range(p):
            ang = -2 * iv.pi * (2 * k * a % p) / p
            rr.append(iv.cos(ang)); ii.append(iv.sin(ang))
        if sym:
            rr = rr + [-v for v in rr]; ii = ii + [-v for v in ii]
        re.append(rr); im.append(ii)
    return re, im

def certify(p, T, L, sym, cyclic=True):
    T = list(T); K = len(T)
    E = generators(p, T, sym); n = E.shape[1]
    # H_a . x with x = (Re z, Im z):  Re(z E) = Re z Re E - Im z Im E
    H = np.vstack([E.real, -E.imag])                             # 2K x n  (columns are H_a)
    ivre, ivim = iv_generators(p, T, sym)
    Umax = mpf(0); worst = None
    # first coordinate: the row shift brings z_1's phase into [-pi/p, pi/p]; cover that window with grid points
    # j (2 pi / L), |j| <= ceil(L/(2p) - 1/2): every phase in the window is within pi/L of one of them.
    jmax = max(0, int(np.ceil(L / (2 * p) - 0.5))) if cyclic else None
    first = list(range(-jmax, jmax + 1)) if cyclic else list(range(L))
    idx_iter = itertools.product(first, *[range(L)] * (K - 1))
    for idx in idx_iter:
        ph = np.array(idx)
        d = np.exp(2j * np.pi * ph / L)
        c = np.r_[d.real, d.imag]
        r = linprog(np.ones(n), A_eq=H, b_eq=c, bounds=[(0, None)] * n, method='highs')
        y = np.clip(r.x if r.status == 0 else np.zeros(n), 0, None)
        # rigorous upper bound
        ysum = iv.mpf(0)
        yiv = [iv.mpf(float(v)) for v in y]
        for v in yiv: ysum += v
        res1 = iv.mpf(0)
        for j in range(K):
            ang = 2 * iv.pi * int(ph[j]) / L
            cre, cim = iv.cos(ang), iv.sin(ang)
            sre = iv.mpf(0); sim = iv.mpf(0)
            for a in range(n):
                if y[a] == 0: continue
                sre += yiv[a] * ivre[j][a]
                sim += yiv[a] * (-ivim[j][a])
            res1 += abs(cre - sre) + abs(cim - sim)
        U = mpf((ysum + 2 * res1).b.b if hasattr((ysum + 2 * res1).b, 'b') else (ysum + 2 * res1).b)
        if U > Umax:
            Umax, worst = U, ph
    q = iv.cos(iv.pi / L) / iv.mpf(Umax)                         # interval; take its lower endpoint
    lo = q.a
    return float(mpf(lo.a if hasattr(lo, 'a') else lo)) - 1e-15, worst

if __name__ == '__main__':
    p = int(sys.argv[1]); T = [int(t) for t in sys.argv[2].split(',')]; L = int(sys.argv[3])
    for sym in (False, True):
        lb, w = certify(p, T, L, sym)
        print(f"p={p} T={T} L={L} {'two' if sym else 'one'}-sided depth >= {lb:.6f}")
