# Independent referee library (does not import elib).
import numpy as np, itertools

def Bp(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)

def table(U, W):
    # U (m,n,p), W (m,p): f[x_1..x_n, c]
    m, n, p = U.shape
    f = np.zeros((p,) * n + (p,))
    for i in range(m):
        T = W[i]
        for t in reversed(range(n)):
            T = np.multiply.outer(U[i, t], T)
        f = f + T
    return f

def labels(p, n):
    return np.indices((p,) * n).sum(0) % p

def gaps(f, p, n):
    y = labels(p, n)
    ff = f.reshape(-1, p); yy = y.reshape(-1); r = np.arange(ff.shape[0])
    return np.stack([ff[r, yy] - ff[r, (yy - d) % p] for d in range(1, p)], 1)

def margin(U, W):
    m, n, p = U.shape
    return gaps(table(U, W), p, n).min()

def sqnorm(U, W):
    return (U ** 2).sum((1, 2)) + (W ** 2).sum(1)

def fields(f, p, n):
    """Phi[l-1] array over x for l in F; Phi_l(x) = (2/p) rho^{l y} sum_c f[x,c] rho^{-lc}"""
    F = (p - 1) // 2; y = labels(p, n)
    c = np.arange(p)
    out = []
    for l in range(1, F + 1):
        Fl = np.tensordot(f, np.exp(-2j * np.pi * l * c / p), axes=([n], [0]))
        out.append((2.0 / p) * np.exp(2j * np.pi * l * y / p) * Fl)
    return np.array(out)

def coef(Phi_l, chi, p, n):
    x = np.indices((p,) * n)
    ph = np.exp(-2j * np.pi * sum(chi[t] * x[t] for t in range(n)) / p)
    return np.mean(Phi_l * ph)

def atoms(U, W, j):
    """beta_ij and xi_itj from factor DFTs: beta_ij = 2 p^{-nu} conj(what_i(j)) prod_t uhat_it(j)."""
    m, n, p = U.shape; nu = n + 1
    Uh = np.fft.fft(U, axis=2)[:, :, j]; Wh = np.fft.fft(W, axis=1)[:, j]
    beta = 2 * p ** (-nu) * np.conj(Wh) * np.prod(Uh, axis=1)
    with np.errstate(all='ignore'):
        xi = np.conj(Uh) / Uh
    xi = np.where(np.isfinite(xi), xi, 1.0)
    return beta, xi

def renyi(w, a):
    w = np.asarray(w, float); w = w[w > 0]; w = w / w.sum()
    if abs(a - 1) < 1e-13: return float(-(w * np.log(w)).sum())
    return float(np.log((w ** a).sum()) / (1 - a))

def kap(p):
    F = (p - 1) // 2
    return np.array([[1 - np.cos(2 * np.pi * j * d / p) for j in range(1, F + 1)] for d in range(1, p)])

def programme(p, s, S=None):
    """Gamma_s^{(S)} by brute-force vertex enumeration of {v>=0 on S, K v >= 1}, min sum v^{1/s}."""
    F = (p - 1) // 2; r = 1 / s
    S = list(range(1, F + 1)) if S is None else list(S)
    K = kap(p)[:F][:, [j - 1 for j in S]]; k = len(S)
    A = np.vstack([K, np.eye(k)]); bb = np.concatenate([np.ones(F), np.zeros(k)])
    best = np.inf; vs = []
    for act in itertools.combinations(range(F + k), k):
        M = A[list(act)]
        if abs(np.linalg.det(M)) < 1e-12: continue
        v = np.linalg.solve(M, bb[list(act)])
        if v.min() < -1e-11 or (K @ v).min() < 1 - 1e-9: continue
        v = np.clip(v, 0, None); val = np.sum(v[v > 0] ** r)
        if val < best - 1e-12: best = val; vs = [v]
        elif abs(val - best) <= 1e-12: vs.append(v)
    G = best ** (-s)
    ys = []
    for v in vs:
        y = np.zeros(F); y[[j - 1 for j in S]] = v / best ** s
        ys.append(y)
    return G, ys

def if_block(p, n, j, lam=1.0, ph0=0.0):
    nu = n + 1; a = np.arange(p)
    U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for l in range(nu):
        al = np.pi * l / nu
        for t in range(n):
            U[l, t] = lam * np.cos(2 * np.pi * j * a / p + al + (ph0 if t == 0 else 0))
        W[l] = lam * np.cos(2 * np.pi * j * a / p + n * al + ph0)
    return U, W
