"""(7,2), two features S = {1,2}, beta_3 = 0.  Free relaxation: own coefficients of Phi_1, Phi_2 fixed by atoms, Phi_3
has mean 0, every other Fourier coefficient free.  Ratio = h_free / (Gamma_s(7) 3^{1-s} (Q_1 + Q_2)^s)  (exact atom costs,
shared single-neuron budget sum_j Q_j <= C_b P).  ratio < 1 for all atom systems  =>  no maximiser with features {1,2}.
Usage: python free72.py s natoms restarts seed
"""
import sys, itertools
import numpy as np
from scipy.optimize import linprog, minimize

p = 7; n = 2
rho = np.exp(2j * np.pi / p)
X = np.array(list(itertools.product(range(p), repeat=2)))
N = p * p
DELS = list(range(1, p))
s = float(sys.argv[1]); m = int(sys.argv[2]); RS = int(sys.argv[3]); seed = int(sys.argv[4])
r = 1 / s


def Gamma7(s):
    from scipy.optimize import minimize as mn
    K = np.array([[1 - np.cos(2 * np.pi * j * d / p) for j in (1, 2, 3)] for d in (1, 2, 3)])
    best = 0
    for t in range(30):
        y0 = np.random.default_rng(t).uniform(0.1, 1, 3)
        f = lambda y: -np.min(K @ (np.abs(y) / np.sum(np.abs(y) ** r) ** s))
        res = mn(f, y0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 4000})
        best = max(best, -res.fun)
    return best


G7 = Gamma7(s)
OWN = {j: [((2 * j) % p * a, (2 * j) % p * b) for (a, b) in [(1, 0), (0, 1), (1, 1)]] for j in (1, 2)}


def build():
    nv = 6 * N + 1
    Aub = np.zeros((6 * N, nv))
    for di in range(6):
        for xi in range(N):
            Aub[di * N + xi, di * N + xi] = -1; Aub[di * N + xi, -1] = 1
    rows = []
    for di in range(6):
        rr = np.zeros(nv); rr[di * N:(di + 1) * N] = 1 / N; rows.append(rr)
    for j in (1, 2):
        for chi in OWN[j]:
            ph = rho ** (-(chi[0] * X[:, 0] + chi[1] * X[:, 1])) / N
            re = np.zeros(nv); im = np.zeros(nv)
            for di, d in enumerate(DELS):
                c = rho ** (j * d) * ph
                re[di * N:(di + 1) * N] = c.real; im[di * N:(di + 1) * N] = c.imag
            rows += [re, im]
    return Aub, np.array(rows), nv


Aub, Aeq, nv = build()


def h_free(A, bT):
    """A = (A1, A2) clean coefficients, bT[j] = sign flips of feature j (3 each); Phi_3 mean 0."""
    # means: E g(.,d) = sum_j Re(conj(A_j) (1 - rho^{-j d}))
    m_ = [sum(np.real(np.conj(A[j - 1]) * (1 - rho ** (-j * d))) for j in (1, 2)) for d in DELS]
    beq = list(m_)
    for j in (1, 2):
        for k in range(3):
            v = -(p / 2) * np.conj(bT[j - 1][k])
            beq += [v.real, v.imag]
    c = np.zeros(nv); c[-1] = -1
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(Aub.shape[0]), A_eq=Aeq, b_eq=np.array(beq), bounds=[(None, None)] * nv, method='highs')
    return res.x[-1] if res.status == 0 else -np.inf


def unpack(v):
    out = []
    for j in range(2):
        w = v[j * 4 * m:(j + 1) * 4 * m]
        beta = w[:m] * np.exp(1j * w[m:2 * m]); xi = np.exp(1j * w[2 * m:4 * m].reshape(m, 2))
        out.append((beta, xi))
    return out


def ratio(v):
    A = []; bT = []; Q = 0
    for (beta, xi) in unpack(v):
        A.append(beta.sum()); bT.append([(beta * xi[:, 0]).sum(), (beta * xi[:, 1]).sum(), (beta * xi[:, 0] * xi[:, 1]).sum()])
        Q += np.sum(np.abs(beta) ** r)
    if Q < 1e-9:
        return 0
    return h_free(A, bT) / (G7 * 3 ** (1 - s) * Q ** s)


rng = np.random.default_rng(seed)
best = (-1, None)
for k in range(RS):
    v0 = np.concatenate([np.concatenate([rng.uniform(0.2, 1, m), rng.uniform(0, 2 * np.pi, m), rng.uniform(0, 2 * np.pi, 2 * m)]) for _ in range(2)])
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'maxiter': 2000 * m, 'xatol': 1e-7, 'fatol': 1e-9})
    if -res.fun > best[0]:
        best = (-res.fun, res.x)
    print(f"start {k}: {-res.fun:.5f}  best {best[0]:.5f}", flush=True)
print(f"s={s} Gamma7={G7:.5f}: free-relaxation sup (two features, {m} atoms each) >= {best[0]:.5f}")
