"""arch_core2.py -- shared numerics for the (E+) programme (tenth pass).

Population: N atoms, complex amplitudes c_l, unimodular slot phases v_lj (j=1..n).
  h(u)   = sum_l c_l prod_j (1 + u_j v_lj)          generating polynomial on the torus T^n
  m      = min_{T^n} Re h                            signed aimed output
  m_l    = (1, v_l1, v_l1 v_l2, ..., v_l1..v_ln)     chain vector in C^D, D = n+1
  M_+    = sum_l |c_l| m_l m_l^*                     absolute chain matrix (Hermitian PSD)
Conjecture (E+): det(I + t M_+) >= (1 + t m)^D for t >= 0  <=>  E_k(M_+) >= C(D,k) m^k.
Since M_+ is PSD, Maclaurin gives E_k >= C(D,k) det^{k/D}, so (E+) <=> det M_+ >= m^D.
"""
import numpy as np
from scipy.optimize import minimize
from math import comb

def chain(v):
    """v: (N,n) unimodular -> (N,D) chain vectors."""
    N, n = v.shape
    return np.concatenate([np.ones((N, 1), dtype=complex), np.cumprod(v, axis=1)], axis=1)

def Mplus(x, v):
    ch = chain(v)
    return (ch.T * x) @ ch.conj()

def Mcomplex(c, v):
    ch = chain(v)
    return (ch.T * c) @ ch.conj()

def hparts(c, v, U):
    """U: (...,n-1) first n-1 torus coords. Returns a,b with h = a + b u_n."""
    n = v.shape[1]
    g = np.ones(U.shape[:-1] + (len(c),), dtype=complex)
    for j in range(n - 1):
        g = g * (1 + U[..., j, None] * v[:, j])
    return (g * c).sum(-1), (g * c * v[:, n - 1]).sum(-1)

def aimed(c, v, G=None, polish=True):
    """m = min_{T^n} Re h, exact in the last slot (Re a - |b|), grid + Nelder-Mead in the others.
    Returned value is an evaluation at a real torus point, hence an UPPER bound on the true min."""
    n = v.shape[1]
    if n == 1:
        a, b = hparts(c, v, np.zeros((1, 0)))
        return float((a.real - np.abs(b))[0])
    if G is None:
        G = {2: 720, 3: 64, 4: 20}[n]
    th = np.exp(2j * np.pi * np.arange(G) / G)
    grids = np.stack(np.meshgrid(*([th] * (n - 1)), indexing='ij'), -1).reshape(-1, n - 1)
    a, b = hparts(c, v, grids)
    f = a.real - np.abs(b)
    i = int(f.argmin())
    if not polish:
        return float(f[i])
    t0 = np.angle(grids[i])
    def ff(t):
        a, b = hparts(c, v, np.exp(1j * t)[None, :])
        return float((a.real - np.abs(b))[0])
    rs = minimize(ff, t0, method='Nelder-Mead', options=dict(xatol=1e-11, fatol=1e-15, maxiter=600))
    return float(min(f[i], rs.fun))

def Ek_ratios(x, v, m):
    """E_k(M_+)/(C(D,k) m^k), k=1..D, plus det ratio and eigenvalues."""
    D = v.shape[1] + 1
    ev = np.clip(np.linalg.eigvalsh(Mplus(x, v)), 0, None)
    cf = np.poly(-ev)  # cf[k] = E_k
    return np.array([cf[k] / (comb(D, k) * m ** k) for k in range(1, D + 1)]), ev

def det_ratio(x, v, m):
    D = v.shape[1] + 1
    ev = np.clip(np.linalg.eigvalsh(Mplus(x, v)), 0, None)
    return float(np.prod(ev) / m ** D)

def unpack(z, N, n):
    c = z[:N] + 1j * z[N:2 * N]
    v = np.exp(1j * z[2 * N:].reshape(N, n))
    return c, v

def pack(c, v):
    return np.concatenate([c.real, c.imag, np.angle(v).ravel()])

def starts(kind, N, n, rng, ns=1.0):
    """Adversarial start families."""
    D = n + 1
    l = np.arange(N)
    w = 2 * np.pi / D
    if kind == 'block':      # perturbed common-phase block (first D atoms), extras small
        cc = np.where(l < D, 1.0 / D, 0.0) + 0.1 * ns * rng.normal(size=N)
        ph = np.tile((w * l)[:, None], (1, n)) + 0.5 * ns * rng.normal(size=(N, n))
        return np.concatenate([cc, 0.1 * ns * rng.normal(size=N), ph.ravel()])
    if kind == 'random':
        return np.concatenate([rng.normal(size=2 * N), rng.uniform(0, 2 * np.pi, N * n)])
    if kind == 'aligned':    # positive amplitudes, random phases
        return np.concatenate([np.abs(rng.normal(size=N)), 0.05 * rng.normal(size=N), rng.uniform(0, 2 * np.pi, N * n)])
    if kind == 'twist':      # block with one slot conjugated (twisted block) + noise
        cc = np.where(l < D, 1.0 / D, 0.0) + 0.1 * ns * rng.normal(size=N)
        ph = np.tile((w * l)[:, None], (1, n)); ph[:, 0] *= -1
        ph = ph + 0.3 * ns * rng.normal(size=(N, n))
        return np.concatenate([cc, 0.1 * ns * rng.normal(size=N), ph.ravel()])
    if kind == 'collapse':   # two atoms nearly equal phases (near-collapsed chain frame)
        cc = np.where(l < D, 1.0 / D, 0.0) + 0.1 * ns * rng.normal(size=N)
        ph = np.tile((w * l)[:, None], (1, n)) + 0.3 * ns * rng.normal(size=(N, n))
        ph[1] = ph[0] + 0.05 * rng.normal(size=n)
        return np.concatenate([cc, 0.1 * ns * rng.normal(size=N), ph.ravel()])
    if kind == 'pop':        # aligned population on a bigger regular polygon per slot
        K = N
        ph = np.tile((2 * np.pi * l / K)[:, None], (1, n)) * rng.integers(1, K, size=(1, n)) + 0.2 * ns * rng.normal(size=(N, n))
        return np.concatenate([np.ones(N) / N + 0.05 * rng.normal(size=N), 0.05 * rng.normal(size=N), ph.ravel()])
    raise ValueError(kind)
