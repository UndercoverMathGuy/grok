"""Referee's independent toolkit (does NOT import the split agent's lib.py).

Conventions: vhat(k) = sum_a v(a) rho^{-ka}, rho = exp(2 pi i/p).  Network = (U, W) with U[i,t,a], W[i,c].
Table f[x_1..x_n, c] = sum_i W[i,c] prod_t U[i,t,x_t].  Gap g(x,d) = f[x,y] - f[x,y-d], y = sum x_t mod p.
Cost P = sum_i (sum_t |U_it|^2 + |W_i|^2)^{b/2}.  gamma = h / P^{s}, s = nu/b  (||theta||^nu = P^{nu/b}).
Everything here is computed from the TABLE via numpy FFTs where possible (independent route from atom formulas).
"""
import itertools
import numpy as np


def F_(p):
    return list(range(1, (p - 1) // 2 + 1))


def fproj(v, ks, p):
    """Orthogonal projection of real array v (last axis Z_p) onto the sum of frequency planes in ks (0 = DC),
    computed by FFT masking (independent of the cos/sin route)."""
    vh = np.fft.fft(v, axis=-1)
    mask = np.zeros(p, bool)
    for k in ks:
        mask[k % p] = True
        mask[(-k) % p] = True
    return np.real(np.fft.ifft(np.where(mask, vh, 0), axis=-1))


def table(U, W):
    m, n, p = U.shape
    letters = 'abcdefghjk'[:n]
    expr = ','.join('i' + l for l in letters) + ',iz->' + letters + 'z'
    return np.einsum(expr, *[U[:, t, :] for t in range(n)], W, optimize=True)


def gaps(f):
    """G[x..., d] for d = 0..p-1 (d=0 column zero). f has shape (p,)*n + (p,)."""
    n = f.ndim - 1; p = f.shape[-1]
    y = np.indices((p,) * n).sum(0) % p
    fy = np.take_along_axis(f, y[..., None], axis=-1)[..., 0]
    G = np.zeros(f.shape)
    for d in range(1, p):
        G[..., d] = fy - np.take_along_axis(f, ((y - d) % p)[..., None], axis=-1)[..., 0]
    return G


def margin_f(f):
    return gaps(f)[..., 1:].min()


def margin(U, W):
    return margin_f(table(U, W))


def cost(U, W, b):
    return float((((U ** 2).sum((1, 2)) + (W ** 2).sum(1)) ** (b / 2)).sum())


def Bp(p, n):
    nu = n + 1
    return 2.0 * (2 * p * nu) ** (-nu / 2)


def kappa(l, d, p):
    return 1 - np.cos(2 * np.pi * l * d / p)


def own_from_table(f):
    """beta_j and beta_{j,T} read off the table's Fourier transform: beta_{j,T} = 2 p^{-nu} fhat(k; -j), k_t = -j on T, j off T."""
    n = f.ndim - 1; p = f.shape[-1]; nu = n + 1
    Fh = np.fft.fftn(f)
    beta, betaT = {}, {}
    for j in F_(p):
        beta[j] = 2 * p ** (-nu) * Fh[tuple([j] * n + [(-j) % p])]
        for r in range(1, n + 1):
            for T in itertools.combinations(range(n), r):
                idx = tuple([((-j) % p) if t in T else j for t in range(n)] + [(-j) % p])
                betaT[(j, T)] = 2 * p ** (-nu) * Fh[idx]
    return beta, betaT


def atoms(U, W):
    m, n, p = U.shape; nu = n + 1
    Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W, axis=-1)
    return {j: 2 * p ** (-nu) * np.conj(Wh[:, j]) * np.prod(Uh[:, :, j], axis=1) for j in F_(p)}


def split(U, W):
    """theta_split: pieces omega_i^{(k)} for all i, k in F (DC dropped, dead pieces kept)."""
    m, n, p = U.shape
    Us, Ws = [], []
    for i in range(m):
        for k in F_(p):
            Us.append(fproj(U[i], [k], p)); Ws.append(fproj(W[i], [k], p))
    return np.array(Us), np.array(Ws)


def piece_energies(U, W):
    """e[i, k] for k = 0 (DC) and k in F: energy of neuron i's frequency-k piece (all factors)."""
    m, n, p = U.shape
    out = np.zeros((m, len(F_(p)) + 1))
    for k in [0] + F_(p):
        out[:, k] = (fproj(U, [k], p) ** 2).sum((1, 2)) + (fproj(W, [k], p) ** 2).sum(1)
    return out


def live_mask(U, W, tol=1e-13):
    """live[i, k-1] = all factors of piece (i,k) non-zero."""
    m, n, p = U.shape
    Uh = np.abs(np.fft.fft(U, axis=-1)); Wh = np.abs(np.fft.fft(W, axis=-1))
    return np.array([[min(Uh[i, :, k].min(), Wh[i, k]) > tol for k in F_(p)] for i in range(m)])


def P_live_by_freq(U, W, b):
    e = piece_energies(U, W); L = live_mask(U, W)
    return {k: float((np.where(L[:, k - 1], e[:, k], 0) ** (b / 2)).sum()) for k in F_(U.shape[2])}


def field(f, l):
    """Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-lc}, array of shape (p,)*n."""
    n = f.ndim - 1; p = f.shape[-1]
    y = np.indices((p,) * n).sum(0) % p
    rho = np.exp(2j * np.pi / p)
    return (2 / p) * rho ** (l * y) * (f @ rho ** (-l * np.arange(p)))


def fcoef(Phi):
    """phi(chi) with Phi = sum_chi phi(chi) rho^{chi.x}."""
    return np.fft.fftn(Phi) / Phi.size


def ghat(f, d):
    G = gaps(f)[..., d]
    return np.fft.fftn(G) / G.size


def diag_proj_table(f):
    """Projection of a table onto modes whose labels (|q_1|,...,|q_n|,|q_0|) are all equal to one k in F.
    This is the table of theta_split (linear in the table)."""
    n = f.ndim - 1; p = f.shape[-1]
    Fh = np.fft.fftn(f)
    lab = lambda q: min(q % p, (-q) % p)
    mask = np.zeros(Fh.shape, bool)
    for idx in itertools.product(range(p), repeat=n + 1):
        ls = {lab(q) for q in idx}
        if len(ls) == 1 and 0 not in ls:
            mask[idx] = True
    return np.real(np.fft.ifftn(np.where(mask, Fh, 0)))


def alldc_table(f):
    n = f.ndim - 1; p = f.shape[-1]
    Fh = np.fft.fftn(f)
    out = np.zeros_like(Fh); out[(0,) * (n + 1)] = Fh[(0,) * (n + 1)]
    return np.real(np.fft.ifftn(out))


def programme(p, s, grid=None, rng=None, iters=4000):
    """Gamma_s(p) by brute random search + local polish over y >= 0, sum y^r = 1 (small p only)."""
    from scipy.optimize import minimize
    F = F_(p); r = 1 / s
    K = np.array([[kappa(l, d, p) for l in F] for d in range(1, p)])
    rng = rng or np.random.default_rng(0)
    best = (-1, None)
    for it in range(iters):
        u = rng.dirichlet(np.ones(len(F)) * rng.choice([0.2, 1, 5]))
        y = u ** s
        v = (K @ y).min()
        if v > best[0]:
            best = (v, y)
    # polish
    def negv(z):
        u = np.abs(z) / np.abs(z).sum(); y = u ** s
        return -(K @ y).min()
    res = minimize(negv, best[1] ** (1 / s), method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
    u = np.abs(res.x) / np.abs(res.x).sum()
    return max(best[0], -res.fun), u ** s


def rand_net(rng, m, n, p, kind='gauss', dc=1.0):
    U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
    if kind == 'gauss':
        return U, W
    if kind == 'sparse':      # each factor on a random subset of frequency planes (some exact zeros -> dead pieces)
        for i in range(m):
            for t in range(n):
                ks = [k for k in [0] + F_(p) if rng.random() < 0.5] or [1]
                U[i, t] = fproj(U[i, t], ks, p)
            ks = [k for k in [0] + F_(p) if rng.random() < 0.5] or [1]
            W[i] = fproj(W[i], ks, p)
        return U, W
    if kind == 'heavydc':
        return U + dc * rng.normal(size=(m, n, 1)), W + dc * rng.normal(size=(m, 1))
    raise ValueError(kind)
