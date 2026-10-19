"""Independent referee library (does not import the agent's elib/splitlib).
Conventions (BRIEF.md): U[i,t,:] input vectors, W[i,:] output vector; f(x)[c] = sum_i W[i,c] prod_t U[i,t,x_t];
g(x,d) = f(x)[y] - f(x)[y-d], y = sum x_t mod p; vhat(k) = sum_a v(a) e^{-2 pi i k a/p} (numpy fft)."""
import itertools, math
import numpy as np


def F_of(p):
    return list(range(1, (p - 1) // 2 + 1))


def table(U, W):
    m, n, p = U.shape
    f = np.zeros((p,) * n + (p,))
    for i in range(m):
        T = W[i]
        for t in reversed(range(n)):
            T = np.multiply.outer(U[i, t], T)
        f = f + T
    return f


def gaps(f, p, n):
    """G[x_flat, d] for d=1..p-1 (column d-1)."""
    F2 = f.reshape(-1, p)
    idx = np.indices((p,) * n).reshape(n, -1)
    y = idx.sum(0) % p
    r = np.arange(F2.shape[0])
    return np.stack([F2[r, y] - F2[r, (y - d) % p] for d in range(1, p)], 1)


def margin(U, W):
    m, n, p = U.shape
    return gaps(table(U, W), p, n).min()


def sqnorm(U, W):
    return (U ** 2).sum((1, 2)) + (W ** 2).sum(1)


def shares(U, W, s):
    m, n, p = U.shape
    nu = n + 1; b = nu / s
    c = sqnorm(U, W) ** (b / 2)
    return c, c / c.sum()


def ent(w):
    w = np.asarray(w, float); w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def renyi(w, q):
    w = np.asarray(w, float); w = w[w > 0]
    if abs(q - 1) < 1e-14:
        return ent(w)
    return float(np.log((w ** q).sum()) / (1 - q))


def logF(U, W, s, h=None):
    m, n, p = U.shape
    if h is None:
        h = margin(U, W)
    c, w = shares(U, W, s)
    lg = math.log(h) - s * math.log(c.sum())
    H = ent(w)
    return lg, H, lg + (s - 1) * H


def atoms(U, W):
    m, n, p = U.shape
    nu = n + 1
    Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W, axis=-1)
    beta = {}; betaT = {}; at = {}
    for j in F_of(p):
        a = 2 * p ** (-nu) * np.conj(Wh[:, j]) * np.prod(Uh[:, :, j], axis=1)
        at[j] = a; beta[j] = a.sum()
        for r in range(1, n + 1):
            for T in itertools.combinations(range(n), r):
                pr = np.conj(Wh[:, j]).copy()
                for t in range(n):
                    pr = pr * (Uh[:, t, (-j) % p] if t in T else Uh[:, t, j])
                betaT[(j, T)] = 2 * p ** (-nu) * pr.sum()
    return beta, betaT, at


def proj(v, k, p):
    v = np.asarray(v, float)
    if k == 0:
        return np.repeat(v.mean(-1, keepdims=True), p, axis=-1)
    vh = np.fft.fft(v, axis=-1)
    mask = np.zeros(p); mask[k] = 1; mask[(-k) % p] = 1
    return np.real(np.fft.ifft(vh * mask, axis=-1))


def plane_energy(U, W):
    m, n, p = U.shape
    e = np.zeros((m, len(F_of(p)) + 1))
    for k in [0] + F_of(p):
        e[:, k] = (proj(U, k, p) ** 2).sum((1, 2)) + (proj(W, k, p) ** 2).sum(1)
    return e


def kappa(j, d, p):
    return 1 - np.cos(2 * np.pi * j * d / p)


def Bp(p, n):
    nu = n + 1
    return 2 * (2 * p * nu) ** (-nu / 2)


def block(p, n, j, lam=1.0, phase=0.0, m=None):
    """aligned IF block at frequency j with m >= nu equally spaced phases (m = nu: the paper's block)."""
    nu = n + 1
    if m is None:
        m = nu
    a = np.arange(p)
    U = np.zeros((m, n, p)); W = np.zeros((m, p))
    for l in range(m):
        al = np.pi * l / m + phase
        for t in range(n):
            U[l, t] = lam * np.cos(2 * np.pi * j * a / p + al)
        W[l] = lam * np.cos(2 * np.pi * j * a / p + n * al)
    return U, W
