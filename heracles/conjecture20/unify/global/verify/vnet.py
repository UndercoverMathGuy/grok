# Independent referee library (does NOT import ../lib.py).
# Conventions (BRIEF): neuron i has inputs u_it in R^p (t=1..n) and output w_i in R^p;
#   f(x)[c] = sum_i w_i(c) prod_t u_it(x_t);  y(x) = sum_t x_t mod p;  gap g(x,d) = f(x)[y] - f(x)[y-d].
#   cost P_b = sum_i ||omega_i||^b,  ||omega||^2 = sum_t ||u_t||^2 + ||w||^2;  gamma = h / P_b^s, s = nu/b.
#   DFT: uhat(k) = sum_a u(a) rho^{-k a}, rho = exp(2 pi i/p).
#   beta_j = 2 p^{-nu} fhat(j,...,j; -j) where fhat(k_1..k_n; c-freq) uses rho^{-k.x - k_c c}.
import itertools, math
import numpy as np


def B_p(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def table(U, W):
    """U: (m, n, p), W: (m, p) -> f: (p,)*n + (p,)"""
    m, n, p = U.shape
    f = np.zeros((p,) * (n + 1))
    for i in range(m):
        T = W[i]
        for t in reversed(range(n)):
            T = np.multiply.outer(U[i, t], T)
        f += T
    return f


def gaps(f):
    """returns G with shape (p,)*n + (p-1,): G[x, d-1] = f(x)[y] - f(x)[y-d]"""
    n = f.ndim - 1; p = f.shape[0]
    idx = np.indices((p,) * n)
    y = idx.sum(0) % p
    out = np.empty((p,) * n + (p - 1,))
    top = np.take_along_axis(f, y[..., None], axis=-1)[..., 0]
    for d in range(1, p):
        out[..., d - 1] = top - np.take_along_axis(f, ((y - d) % p)[..., None], axis=-1)[..., 0]
    return out


def margin(f):
    return gaps(f).min()


def neuron_sq(U, W):
    return (U ** 2).sum((1, 2)) + (W ** 2).sum(1)


def cost(U, W, b):
    return np.sum(neuron_sq(U, W) ** (b / 2.0))


def gamma(U, W, b):
    n = U.shape[1]; s = (n + 1) / b
    return margin(table(U, W)) / cost(U, W, b) ** s


def fhat(f):
    # numpy fftn uses exp(-2 pi i k x / p): matches uhat(k) = sum u(a) rho^{-k a}
    return np.fft.fftn(f)


def clean(f):
    n = f.ndim - 1; p = f.shape[0]; nu = n + 1; F = (p - 1) // 2; fh = fhat(f)
    return np.array([2 * p ** (-nu) * fh[(j,) * n + ((-j) % p,)] for j in range(1, F + 1)])


def signflip(f, j):
    """dict T(tuple 0/1) -> beta_{j,T} (input index -j on T)"""
    n = f.ndim - 1; p = f.shape[0]; nu = n + 1; fh = fhat(f); out = {}
    for T in itertools.product([0, 1], repeat=n):
        k = tuple(((-j) if T[t] else j) % p for t in range(n)) + ((-j) % p,)
        out[T] = 2 * p ** (-nu) * fh[k]
    return out


def atoms(U, W, j):
    """beta_{ij} = 2 p^{-nu} conj(what_i(j)) prod_t uhat_it(j)"""
    m, n, p = U.shape; nu = n + 1
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    return 2 * p ** (-nu) * np.conj(Wh[:, j]) * np.prod(Uh[:, :, j], axis=1)


def Phi(f, l):
    """feature field Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-l c}"""
    n = f.ndim - 1; p = f.shape[0]
    idx = np.indices((p,) * n); y = idx.sum(0) % p
    c = np.arange(p); rho = np.exp(2j * np.pi / p)
    S = (f * rho ** (-l * c)).sum(-1)
    return (2.0 / p) * rho ** (l * y) * S


def kappa(p, j, d):
    return 1 - math.cos(2 * math.pi * j * d / p)


def freq_basis(p, k):
    a = np.arange(p)
    if k == 0:
        return np.ones((1, p))
    return np.stack([np.cos(2 * np.pi * k * a / p), np.sin(2 * np.pi * k * a / p)])


def if_block(p, n, j, lam=1.0, ph=0.0):
    nu = n + 1; a = np.arange(p); U = []; W = []
    for q in range(nu):
        al = np.pi * q / nu + ph
        U.append([lam * np.cos(2 * np.pi * j * a / p + al) for _ in range(n)])
        W.append(lam * np.cos(2 * np.pi * j * a / p + n * al))
    return np.array(U), np.array(W)
