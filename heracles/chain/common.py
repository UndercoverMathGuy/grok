"""Shared helpers for verifying the CHAIN.md steps (n = 2 operands, nu = 3, product model).

Conventions (theory_modadd.md):
  table z[a, b, c]; answer s = a + b mod p; gap[a, b, D] = z[a, b, s] - z[a, b, s - D] for D = 1..p-1
  (index 0 of the last axis is unused); margin = min gap.
  Zhat = fftn(z), i.e. Zhat[al, be, ga] = sum z w^{-(al a + be b + ga c)}, w = exp(2 pi i / p).
  clean G_k = Zhat[k, k, -k]; junk J_{k,T} = Zhat[eps1 k, eps2 k, -k] with eps_j = -1 iff j in T.
  cost P_2 = sum_i (|e1_i|^2 + |e2_i|^2 + |w_i|^2); gamma = margin / P_2^{3/2}.
"""
import itertools
import numpy as np


def cosvec(p, k, ph, amp=1.0):
    return amp * np.cos(2 * np.pi * k * np.arange(p) / p + ph)


def random_net(p, width, rng, pure=True, kset=None):
    """list of neurons (e1, e2, w). pure: all three slots at one frequency k."""
    h = (p - 1) // 2
    ks_all = list(kset) if kset is not None else list(range(1, h + 1))
    net = []
    for _ in range(width):
        if pure:
            k = rng.choice(ks_all)
            ks = (k, k, k)
        else:
            ks = tuple(rng.choice(ks_all, 3))
        ph = rng.uniform(0, 2 * np.pi, 3)
        amp = rng.uniform(0.2, 1.5, 3)
        net.append(tuple(cosvec(p, ks[j], ph[j], amp[j]) for j in range(3)))
    return net


def table(p, net):
    z = np.zeros((p, p, p))
    for e1, e2, w in net:
        z += np.einsum('a,b,c->abc', e1, e2, w)
    return z


def gaps(z):
    p = z.shape[0]
    A, B = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    S = (A + B) % p
    g = np.zeros((p, p, p))
    for D in range(1, p):
        g[:, :, D] = z[A, B, S] - z[A, B, (S - D) % p]
    return g


def margin(z):
    return gaps(z)[:, :, 1:].min()


def cost2(net):
    return sum(float(e1 @ e1 + e2 @ e2 + w @ w) for e1, e2, w in net)


def gamma(p, net):
    return margin(table(p, net)) / cost2(net) ** 1.5


def spectrum(z):
    """clean G[k] and junk J[T][k] for k = 1..h (index 0 unused)."""
    p = z.shape[0]
    Zh = np.fft.fftn(z)
    h = (p - 1) // 2
    G = np.zeros(h + 1, complex)
    J = {T: np.zeros(h + 1, complex) for T in ('1', '2', '12')}
    for k in range(1, h + 1):
        m = (-k) % p
        G[k] = Zh[k, k, m]
        J['1'][k] = Zh[m, k, m]
        J['2'][k] = Zh[k, m, m]
        J['12'][k] = Zh[m, m, m]
    return G, J


def cover_matrix(p):
    """A[D-1, k-1] = 1 - cos(2 pi k D / p), D, k = 1..h."""
    h = (p - 1) // 2
    D = np.arange(1, h + 1)[:, None]
    k = np.arange(1, h + 1)[None, :]
    return 1 - np.cos(2 * np.pi * k * D / p)


def programme(p, s):
    """Exact Gamma_s(p) by vertex enumeration (astra ARGUMENT.md sec 2).
    Returns (Gamma, y) with y[k-1] the maximiser (sum y^{1/s} = 1, min_D A y = Gamma)."""
    A = cover_matrix(p)
    h = A.shape[0]
    r = 1.0 / s
    best = (np.inf, None)
    for K in range(1, h + 1):
        for S in itertools.combinations(range(h), K):
            for T in itertools.combinations(range(h), K):
                M = A[np.ix_(T, S)]
                if abs(np.linalg.det(M)) < 1e-12:
                    continue
                vS = np.linalg.solve(M, np.ones(K))
                if (vS < -1e-12).any():
                    continue
                v = np.zeros(h)
                v[list(S)] = np.clip(vS, 0, None)
                if (A @ v < 1 - 1e-9).any():
                    continue
                c = (v[v > 0] ** r).sum()
                if c < best[0]:
                    best = (c, v)
    C, v = best
    Gam = C ** (-1 / r)
    y = v * C ** (-1 / r)
    return Gam, y


def block_net(p, y):
    """Share-weighted 3-neuron common-phase block (theory_modadd Prop 7.3, n = 2).
    y[k-1] = clean amplitude of frequency k in the target table sum_k y_k cos(2 pi k (a+b-c)/p)."""
    net = []
    for k0, yk in enumerate(y):
        if yk <= 0:
            continue
        k = k0 + 1
        rho = (4 * yk / 3) ** (1 / 3)
        for l in range(3):
            phi = np.pi * l / 3
            net.append((cosvec(p, k, phi, rho), cosvec(p, k, phi, rho), cosvec(p, k, 2 * phi, rho)))
    return net


def gamma_sym(p, Gam):
    B = 2 * (6 * p) ** -1.5
    return 3 ** -0.5 * B * Gam
