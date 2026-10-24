"""Primal of the channel-resolved relaxation at (7,2), L = {1,2}:
   fields Phi_j = own (fixed) + free (arbitrary) + paid (||.||_2 <= R);  g(x,d) = sum_j Re(Phi_j c_j(d)).
   h(R) = max min g;  ratio(R) = h(R) / (Gamma_s sqrt(F^2 + c_s R^2)).
Prints h(R), the ratio, and how the paid mass is distributed over characters at the optimum."""
import sys
import numpy as np
import cvxpy as cp
from n2lib import Grid, classify
from socp72 import G, p, X, LAB, own_from_atoms, Gamma7

s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
r = 1 / s
c = 3.0 ** (2 - 2 * s) / 4
NX = p * p
keys = sorted(LAB.keys())
idx_free = [k for k in keys if LAB[k] == 'free']
idx_paid = [k for k in keys if LAB[k] == 'paid']


def gmat(key):
    """real matrices (NX*6 x 2): contribution of z = (Re, Im) at (j, chi) to g(x,d)"""
    ji, c1, c2 = key
    j = G.F[ji]
    ph = G.rho ** ((c1 * X[:, 0] + c2 * X[:, 1]) % p)
    out = np.zeros((NX, 6, 2))
    for di, d in enumerate(G.DELS):
        cc = G.cfac(j, d) * ph            # Re(z cc) = Re z Re cc - Im z Im cc
        out[:, di, 0] = cc.real
        out[:, di, 1] = -cc.imag
    return out.reshape(NX * 6, 2)


GF = np.hstack([gmat(k) for k in idx_free])
GP = np.hstack([gmat(k) for k in idx_paid])


def own_g(A, BT):
    g = np.zeros(NX * 6)
    for ji, j in enumerate((1, 2)):
        z = {(0, 0): np.conj(A[ji])}
        for ti, T in enumerate(G.TS):
            z[((2 * j * T[0]) % p, (2 * j * T[1]) % p)] = np.conj(BT[ji][ti])
        for (c1, c2), val in z.items():
            M = gmat((ji, c1, c2))
            g += M[:, 0] * val.real + M[:, 1] * val.imag
    return g


def solve(A, BT, R):
    zf = cp.Variable(GF.shape[1]); zp = cp.Variable(GP.shape[1]); h = cp.Variable()
    g0 = own_g(A, BT)
    cons = [g0 + GF @ zf + GP @ zp >= h, cp.norm(zp, 2) <= R]
    pr = cp.Problem(cp.Maximize(h), cons)
    pr.solve(solver=cp.CLARABEL)
    return h.value, zp.value, zf.value


if __name__ == '__main__':
    A1 = 1.0; A2 = 2 * np.cos(np.pi / 7)
    xi = np.exp(1j * np.array([[0.3, 1.1]]))
    a1, bt1 = own_from_atoms(np.array([1.0 + 0j]), xi)
    P1 = 3 ** (1 - s); P2 = A2
    F = (P1 ** r + P2 ** r) ** s
    best = (0, None)
    for R in [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
        h, zp, zf = solve([a1, A2], [bt1, [0, 0, 0]], R)
        ratio = h / (Gamma7(s) * np.sqrt(F ** 2 + c * R ** 2))
        print(f"R={R:.2f}: h={h:.5f} ratio={ratio:.5f}", flush=True)
        if ratio > best[0]:
            best = (ratio, R, zp)
    ratio, R, zp = best
    print(f"best ratio {ratio:.5f} at R={R}")
    zc = zp[0::2] + 1j * zp[1::2]
    order = np.argsort(-np.abs(zc))
    print("largest paid coefficients (output j, chi): |z|")
    for o in order[:16]:
        ji, c1, c2 = idx_paid[o]
        j = G.F[ji]
        k = ((c1 - j) % p, (c2 - j) % p)
        print(f"   j={j} chi=({c1},{c2}) input k=({G.cls(k[0])},{G.cls(k[1])}) |z|={abs(zc[o]):.4f}")
