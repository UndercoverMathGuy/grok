"""Core tools for the SPLIT angle (product networks on Z_p^n -> Z_p).

Conventions (as in the Note / paper):
  v_hat(k) = sum_a v(a) rho^{-k a},  rho = e^{2 pi i/p}.
  Network: U[i, t, :] (input vector of neuron i, slot t), W[i, :] (output vector).
  Table f[x_1..x_n, c] = sum_i W[i, c] prod_t U[i, t, x_t].
  Gap g(x, d) = f[x, y(x)] - f[x, y(x) - d],  y(x) = sum_t x_t mod p.  Margin h = min over x, d != 0.
  Cost P = sum_i (sum_t ||U[i,t]||^2 + ||W[i]||^2)^{b/2};  gamma = h / P^{nu/b},  nu = n + 1.
  beta_j = 2 p^{-nu} sum_i conj(What_i(j)) prod_t Uhat_it(j);   beta_{j,T}: Uhat_it(-j) on t in T.
  Frequency projections: P_0 = constants, P_k (k in F = 1..(p-1)/2) = span{cos(2 pi k ./p), sin(2 pi k ./p)}.
"""
import itertools, math
import numpy as np


def Fset(p):
    return list(range(1, (p - 1) // 2 + 1))


def proj(v, k, p):
    """Orthogonal projection of real vector(s) v (last axis = Z_p) onto frequency plane k (k=0: constants)."""
    v = np.asarray(v, float)
    if k == 0:
        return np.repeat(v.mean(-1, keepdims=True), p, axis=-1)
    a = np.arange(p)
    c = np.cos(2 * np.pi * k * a / p); s = np.sin(2 * np.pi * k * a / p)
    return np.multiply.outer(v @ c, c) / (c @ c) + np.multiply.outer(v @ s, s) / (s @ s)


def table(U, W):
    """Full logit table, shape (p,)*n + (p,).  Only for small p^n."""
    m, n, p = U.shape
    f = np.zeros((p,) * n + (p,))
    for i in range(m):
        T = W[i]
        for t in reversed(range(n)):
            T = np.multiply.outer(U[i, t], T)
        f += T
    return f


def gaps_from_table(f, p, n):
    """G[x..., d] = f[x, y] - f[x, y-d] for d = 0..p-1 (d = 0 column is 0)."""
    idx = np.indices((p,) * n).reshape(n, -1)
    y = idx.sum(0) % p
    F = f.reshape(-1, p)
    rows = np.arange(F.shape[0])
    G = np.zeros((F.shape[0], p))
    for d in range(1, p):
        G[:, d] = F[rows, y] - F[rows, (y - d) % p]
    return G


def margin_table(f, p, n):
    return gaps_from_table(f, p, n)[:, 1:].min()


def margin(U, W):
    m, n, p = U.shape
    return margin_table(table(U, W), p, n)


def cost(U, W, b):
    return float(((U ** 2).sum((1, 2)) + (W ** 2).sum(1)) ** (b / 2) @ np.ones(len(W)))


def gamma(U, W, b, h=None):
    m, n, p = U.shape
    nu = n + 1
    if h is None:
        h = margin(U, W)
    return h / cost(U, W, b) ** (nu / b)


def Bp(p, n):
    nu = n + 1
    return 2 * (2 * p * nu) ** (-nu / 2)


def split(U, W, keep_dead=True):
    """theta_split = {omega_i^{(k)} : i, k in F}: project every factor of every neuron on each frequency plane.
    DC components are dropped.  keep_dead=False drops split pieces that have a zero factor (they have zero table)."""
    m, n, p = U.shape
    Us, Ws = [], []
    for i in range(m):
        for k in Fset(p):
            u = np.array([proj(U[i, t], k, p) for t in range(n)]); w = proj(W[i], k, p)
            if not keep_dead:
                if min(np.linalg.norm(u, axis=1).min(), np.linalg.norm(w)) < 1e-14:
                    continue
            Us.append(u); Ws.append(w)
    return np.array(Us).reshape(-1, n, p), np.array(Ws).reshape(-1, p)


def flip(U, W, S, dc=None):
    """Global sign flip sigma in Sigma: S[k-1, t] in {+1,-1} multiplies the frequency-k component of slot t;
    the output frequency-k component is multiplied by prod_t S[k-1, t].  dc[t] in {+1,-1} optionally flips the
    DC component of input slot t (output DC untouched)."""
    m, n, p = U.shape
    U2 = U.copy(); W2 = W.copy()
    for k in Fset(p):
        for t in range(n):
            if S[k - 1, t] < 0:
                U2[:, t] -= 2 * proj(U[:, t], k, p)
        if np.prod(S[k - 1]) < 0:
            W2 -= 2 * proj(W, k, p)
    if dc is not None:
        for t in range(n):
            if dc[t] < 0:
                U2[:, t] -= 2 * proj(U[:, t], 0, p)
    return U2, W2


def flip_neuronwise(U, W, Ss):
    """Per-neuron flips: Ss[i] is the sign array (|F| x n) for neuron i."""
    U2 = U.copy(); W2 = W.copy()
    for i in range(len(W)):
        a, b_ = flip(U[i:i + 1], W[i:i + 1], Ss[i])
        U2[i] = a[0]; W2[i] = b_[0]
    return U2, W2


def own_coeffs(U, W):
    """beta[j] and beta_T[(j, T)] (T a tuple of slots) from the neurons' Fourier coefficients; also atoms."""
    m, n, p = U.shape
    Uh = np.fft.fft(U, axis=-1)            # Uh[i,t,k] = sum_a U(a) e^{-2 pi i k a/p} = v_hat(k)
    Wh = np.fft.fft(W, axis=-1)
    nu = n + 1
    beta = {}; betaT = {}; atoms = {}
    for j in Fset(p):
        at = 2 * p ** (-nu) * np.conj(Wh[:, j]) * np.prod(Uh[:, :, j], axis=1)
        atoms[j] = at
        beta[j] = at.sum()
        for r in range(1, n + 1):
            for T in itertools.combinations(range(n), r):
                prod = np.conj(Wh[:, j]).copy()
                for t in range(n):
                    prod = prod * (Uh[:, t, (-j) % p] if t in T else Uh[:, t, j])
                betaT[(j, T)] = 2 * p ** (-nu) * prod.sum()
    return beta, betaT, atoms


def field_coeffs(U, W, l):
    """Fourier coefficients phi_l(chi) of Phi_l(x) = (2/p) sum_i What_i(l) prod_t u_it(x_t) rho^{l x_t},
    in the convention Phi_l(x) = sum_chi phi_l(chi) rho^{chi . x}.  Returns array of shape (p,)*n."""
    m, n, p = U.shape
    Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W, axis=-1)
    # u(a) rho^{l a} = (1/p) sum_c uhat(c - l) rho^{c a}
    out = np.zeros((p,) * n, complex)
    for i in range(m):
        T = np.array(Wh[i, l] * 2 / p, complex)
        for t in range(n):
            vec = np.array([Uh[i, t, (c - l) % p] / p for c in range(p)])
            T = np.multiply.outer(T, vec)
        out += T
    return out


def kappa(j, d, p):
    return 1 - np.cos(2 * np.pi * j * d / p)


def gamma_IF(p, n, b, Gam):
    nu = n + 1; s = nu / b
    return nu ** (1 - s) * Bp(p, n) * Gam


def if_block(p, n, j, lam=1.0, phase0=0.0):
    """Paper Prop 9 block at frequency j: nu neurons u_t = lam cos(2 pi j a/p + alpha_l), w = lam cos(2 pi j c/p + n alpha_l)."""
    nu = n + 1
    a = np.arange(p)
    U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for l in range(nu):
        al = np.pi * l / nu + phase0
        for t in range(n):
            U[l, t] = lam * np.cos(2 * np.pi * j * a / p + al)
        W[l] = lam * np.cos(2 * np.pi * j * a / p + n * al)
    return U, W


def smooth_min(G, tau):
    """soft-min of entries of G with temperature tau (lower bound of min up to tau*log(count))."""
    gmin = G.min()
    return gmin - tau * np.log(np.exp(-(G - gmin) / tau).sum())
