"""Sanity checks of the conventions used in mrel.py on real networks.
(1) Phi_j computed from the logit table has, at character 2j1_T, the coefficient conj(beta_{j,T}) where beta_{j,T} is the
    paper's sign-flipped coefficient 2p^{-nu} fhat(k;-j) (k_t=-j on T, j off T).
(2) beta_{j,T} = sum_i beta_ij prod_{t in T} xi_itj with |xi|=1 (paper Lemma 6), for arbitrary inputs.
(3) The chain matrix of the field coefficients of ONE neuron is rank one with singular value nu |beta_ij|.
(4) g(x,delta) from the table equals sum_j Re(Phi_j(x) c_j(delta)).
(5) For the paper's IF block the own coefficients vanish except the clean one, and the margin equals Gamma-type value.
"""
import numpy as np, itertools
from mrel import Model, chain_matrix, subsets

rng = np.random.default_rng(3)


def table(U, W):
    m, n, p = U.shape
    f = np.zeros((p,) * n + (p,))
    for i in range(m):
        T = W[i]
        for t in reversed(range(n)):
            T = np.multiply.outer(U[i, t], T)
        f += T
    return f


def fields(f, p, n):
    rho = np.exp(2j * np.pi / p)
    X = np.array(list(itertools.product(range(p), repeat=n)))
    F = (p - 1) // 2
    Phi = {}
    for j in range(1, F + 1):
        vals = []
        for x in X:
            y = x.sum() % p
            fx = f[tuple(x)]
            vals.append(2 / p * rho ** (j * y) * np.sum(fx * rho ** (-j * np.arange(p))))
        Phi[j] = np.array(vals)
    return X, Phi


def coef(vals, X, chi, p):
    return np.mean(vals * np.exp(-2j * np.pi * (X @ np.array(chi)) / p))


for (p, n, m) in [(5, 2, 3), (7, 2, 2), (5, 3, 2)]:
    nu = n + 1
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
    f = table(U, W)
    X, Phi = fields(f, p, n)
    Fh = np.fft.fftn(f)
    err1 = err2 = err3 = 0
    for j in range(1, (p - 1) // 2 + 1):
        # atoms
        beta_i = []; xi_i = []
        for i in range(m):
            uh = [np.fft.fft(U[i, t]) for t in range(n)]; wh = np.fft.fft(W[i])
            b = 2 * p ** (-nu) * np.conj(wh[j]) * np.prod([uh[t][j] for t in range(n)])
            xi = np.array([np.conj(uh[t][j]) / uh[t][j] for t in range(n)])
            beta_i.append(b); xi_i.append(xi)
        for T in subsets(n):
            k = tuple((-j if t in T else j) % p for t in range(n)) + ((-j) % p,)
            bT = 2 * p ** (-nu) * Fh[k]
            bT_atoms = sum(b * np.prod([x[t] for t in T]) for b, x in zip(beta_i, xi_i))
            chi = tuple((2 * j % p) if t in T else 0 for t in range(n))
            aT = coef(Phi[j], X, chi, p)
            err1 = max(err1, abs(aT - np.conj(bT)))
            err2 = max(err2, abs(bT - bT_atoms))
        # one neuron chain matrix
        M = Model.__new__(Model); M.Ts = subsets(n)
        aj = {T: np.conj(beta_i[0] * np.prod([xi_i[0][t] for t in T])) for T in subsets(n)}
        H = chain_matrix(aj, n, U=None)
        # complete with the mirror: H = conj(beta) conj(m) conj(m)^*
        mvec = np.concatenate([[1], np.cumprod(xi_i[0])])
        Hfull = np.conj(beta_i[0]) * np.outer(np.conj(mvec), mvec)
        low = np.tril(Hfull)
        err3 = max(err3, np.abs(low - np.tril(H)).max())
        sv = np.linalg.svd(Hfull, compute_uv=False)
        assert abs(sv[0] - nu * abs(beta_i[0])) < 1e-12 and sv[1] < 1e-12
    # margin identity
    rho = np.exp(2j * np.pi / p)
    e4 = 0
    for xi, x in enumerate(X):
        y = x.sum() % p
        for d in range(1, p):
            g_tab = f[tuple(x)][y] - f[tuple(x)][(y - d) % p]
            g_fld = sum((Phi[j][xi] * (1 - rho ** (-j * d))).real for j in Phi)
            e4 = max(e4, abs(g_tab - g_fld))
    print(f"p={p} n={n}: |a_T - conj(beta_T)| {err1:.1e}; |beta_T - atoms| {err2:.1e}; chain lower {err3:.1e}; g identity {e4:.1e}")

# IF block (paper Prop 9) at p=5, n=2, j=1
p, n = 5, 2; nu = 3; lam = 1.0
a = np.arange(p)
U = np.array([[lam * np.cos(2 * np.pi * a / p + np.pi * l / nu) for t in range(n)] for l in range(nu)])
W = np.array([lam * np.cos(2 * np.pi * a / p + n * np.pi * l / nu) for l in range(nu)])
f = table(U, W)
X, Phi = fields(f, p, n)
print("IF block own coefficients (j=1):", [np.round(coef(Phi[1], X, tuple(2 if t in T else 0 for t in range(n)), p), 6) for T in subsets(n)])
print("IF block field range:", np.round(Phi[1].min(), 6), np.round(Phi[1].max(), 6), " Phi_2 max", np.abs(Phi[2]).max())
