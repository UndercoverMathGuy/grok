"""Referee-3: explicit interference-free (IF) block networks as tight test cases.

IF block for feature j at n = 2: three neurons k = 0,1,2 with u_1 = u_2 = cos(2 pi j x/p - pi k/3) (so xi = eta = omega^k,
omega = e^{2 pi i/3}) and output w chosen so that every atom equals the same positive real a.  Then B = C = D = 0 for feature j.
(7,2): two-feature IF networks with amplitudes (a_1, a_2), maximised over a_2/a_1; compare gamma/gamma_IF (direct) with the
       simple R14 bound (LP over mu) and with the restricted-programme ratio of R16(a).
(5,2): single-feature IF block: gamma/gamma_IF should be kappa_a/Gamma_s (0.7818 at s = 3/2) and below the R9 bound.
"""
import numpy as np
from scipy.optimize import linprog, minimize_scalar


def make(p, amps):
    """amps: dict j -> atom amplitude a_j (each block has 3 atoms of size a_j)"""
    x = np.arange(p); rho = np.exp(2j * np.pi / p)
    U1 = []; U2 = []; W = []
    for j, a in amps.items():
        for k in range(3):
            u = np.cos(2 * np.pi * j * x / p - np.pi * k / 3)
            uh = np.sum(u * rho ** (-j * x))
            # beta = 2 p^{-3} conj(what(j)) uh^2 = a  -> what(j) = conj(a p^3 / (2 uh^2))
            t = np.conj(a * p ** 3 / (2 * uh * uh))
            w = (2 / p) * np.real(t * rho ** (j * x))
            n1 = np.linalg.norm(u); n3 = np.linalg.norm(w); g = (n1 * n1 * n3) ** (1 / 3)
            U1.append(u * g / n1); U2.append(u * g / n1); W.append(w * g / n3)
    return np.array(U1), np.array(U2), np.array(W)


def evaluate(p, U1, U2, W, s):
    X1, X2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); Y = (X1 + X2) % p
    f = np.einsum('ic,ia,ib->abc', W, U1, U2)
    h = min((f[X1, X2, Y] - f[X1, X2, (Y - d) % p]).min() for d in range(1, p))
    nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
    P = np.sum(nrm2 ** (1.5 / s))
    Bp = 2 * (2 * p * 3) ** -1.5
    Gam = 3.5 * 3 ** -s if p == 7 else 2.5 * 2 ** -s
    return h / P ** s / (3 ** (1 - s) * Bp * Gam)


def own(p, U1, U2, W, j):
    x = np.arange(p); rho = np.exp(2j * np.pi / p)
    hat = lambda v: v @ rho ** (-np.outer(x, x))
    w = hat(W); a = hat(U1); b = hat(U2); c = 2 / p ** 3
    A = np.sum(c * np.conj(w[:, j]) * a[:, j] * b[:, j])
    Ms = [abs(np.sum(c * np.conj(w[:, j]) * a[:, -j % p] * b[:, j])), abs(np.sum(c * np.conj(w[:, j]) * a[:, j] * b[:, -j % p])),
          abs(np.sum(c * np.conj(w[:, j]) * a[:, -j % p] * b[:, -j % p]))]
    return A, max(Ms)


kap = lambda j, d: 1 - np.cos(2 * np.pi * j * d / 7)


def simple_bound(A1, A2, M, x1, x2, s):
    c = [2 * (kap(1, d) * A1 + kap(2, d) * A2) for d in (1, 2, 3)] + [-3.5 * M]
    res = linprog(c, A_ub=[[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]], b_ub=[0, 0, 0], A_eq=[[2, 2, 2, 0]], b_eq=[1],
                  bounds=[(0, None)] * 4, method='highs')
    c1 = 1 - 3 ** (1 - s)
    F = lambda x: max(1 - c1 * x, 3 ** (1 - s) * max(1, x))
    return res.fun / (3.5 * 3 ** -s * ((F(x1) * A1) ** (1 / s) + (F(x2) * A2) ** (1 / s)) ** s)


print("(7,2) two-feature IF networks (features {1,2}):")
for s in (1.0, 1.1, 1.175, 1.182, 1.2, 1.5):
    def neg(lr):
        U1, U2, W = make(7, {1: 1.0, 2: np.exp(lr)})
        return -evaluate(7, U1, U2, W, s)
    r = minimize_scalar(neg, bounds=(-2, 2), method='bounded', options=dict(xatol=1e-10))
    U1, U2, W = make(7, {1: 1.0, 2: np.exp(r.x)})
    ratio = -r.fun
    A1, M1 = own(7, U1, U2, W, 1); A2, M2 = own(7, U1, U2, W, 2); A3, _ = own(7, U1, U2, W, 3)
    sc = abs(A1) + abs(A2)
    sb = simple_bound(abs(A1) / sc, abs(A2) / sc, max(M1, M2) / sc, M1 / abs(A1), M2 / abs(A2), s)
    print(f"  s={s:.3f} (b={3/s:.4f}): best IF ratio {ratio:.5f} at A2/A1={np.exp(r.x):.5f}; |B,C,D|max={max(M1,M2):.1e}, |beta_3|={abs(A3):.1e};"
          f" simple bound at that point {sb:.5f};  ok={ratio <= sb + 1e-9}")

print("(5,2) single-feature IF block:")
for s in (1.0, 1.25, 1.5):
    U1, U2, W = make(5, {1: 1.0})
    r = evaluate(5, U1, U2, W, s)
    print(f"  s={s}: gamma/gamma_IF = {r:.5f}   kappa_a/Gamma_s = {(1-np.cos(2*np.pi/5))/(2.5*2**-s):.5f}")
