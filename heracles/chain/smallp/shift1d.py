"""Depth of the shift-enlarged 1-D body at offset D:  conv{ ( m(s;k) w^{-2kc} )_{k in T} : s, c in Z_p },
m(s;k) = [cos(2 s th_k) - cos((1-2s) th_k)] / (1 - cos th_k),  th_k = 2 pi k D / p.
(s = 0 gives the plain body, s = (p+1)/2 the flip; so this contains the two-sided body.)
Valid aim: pattern-1-pure test distribution = rows (a + sD, b) at D and (a - sD, b) at -D, b uniform (VERIFY Vf.1 generalised,
checked in shift_check.py).  Grid-LP lower bound with the same rigorous dual repair as depth_cert.py (|z_k| bound: the s=0
generators alone force |z_k| <= 2).
"""
import itertools, sys, numpy as np
from scipy.optimize import linprog
from mpmath import iv, mpf
iv.dps = 30
def gens_np(p, T, D):
    T = np.array(T); th = 2 * np.pi * T * D / p; cov = 1 - np.cos(th)
    G = []
    for s in range(p):
        m = (np.cos(2 * s * th) - np.cos((1 - 2 * s) * th)) / cov
        for c in range(p):
            G.append(m * np.exp(-2j * np.pi * 2 * T * c / p))
    return np.array(G).T                         # K x n
def gens_iv(p, T, D):
    re, im = [], []
    for k in T:
        th = 2 * iv.pi * k * D / p; cov = 1 - iv.cos(th); rr, ii = [], []
        for s in range(p):
            m = (iv.cos(2 * s * th) - iv.cos((1 - 2 * s) * th)) / cov
            for c in range(p):
                ang = -2 * iv.pi * (2 * k * c % p) / p
                rr.append(m * iv.cos(ang)); ii.append(m * iv.sin(ang))
        re.append(rr); im.append(ii)
    return re, im
def certify(p, T, D, L):
    E = gens_np(p, T, D); K, n = E.shape; H = np.vstack([E.real, -E.imag]); ivre, ivim = gens_iv(p, T, D)
    jmax = max(0, int(np.ceil(L / (2 * p) - 0.5)))
    Umax = mpf(0)
    for idx in itertools.product(range(-jmax, jmax + 1), *[range(L)] * (K - 1)):
        ph = np.array(idx); d = np.exp(2j * np.pi * ph / L); c = np.r_[d.real, d.imag]
        r = linprog(np.ones(n), A_eq=H, b_eq=c, bounds=[(0, None)] * n, method='highs')
        y = np.clip(r.x, 0, None) if r.status == 0 else np.zeros(n)
        yiv = [iv.mpf(float(v)) for v in y]; ysum = iv.mpf(0)
        for v in yiv: ysum += v
        res1 = iv.mpf(0)
        for j in range(K):
            ang = 2 * iv.pi * int(ph[j]) / L; sre = iv.mpf(0); sim = iv.mpf(0)
            for a in range(n):
                if y[a] == 0: continue
                sre += yiv[a] * ivre[j][a]; sim += yiv[a] * (-ivim[j][a])
            res1 += abs(iv.cos(ang) - sre) + abs(iv.sin(ang) - sim)
        U = (ysum + 2 * res1).b
        U = mpf(U.b if hasattr(U, 'b') else U)
        Umax = max(Umax, U)
    q = (iv.cos(iv.pi / L) / iv.mpf(Umax)).a
    return float(mpf(q.a if hasattr(q, 'a') else q)) - 1e-15
if __name__ == '__main__':
    p = int(sys.argv[1]); T = [int(t) for t in sys.argv[2].split(',')]; L = int(sys.argv[3])
    for D in [int(x) for x in sys.argv[4].split(',')]:
        print(f"p={p} T={T} D={D} L={L}: shift-enlarged depth >= {certify(p, T, D, L):.6f}", flush=True)
