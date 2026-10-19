"""n3lib: shared helpers for the n=3 (nu=4) single-feature payment and detection work.

Atom system: beta_i in C (i=1..N), xi_i in T^3.  T ranges over the 7 non-empty subsets of [3].
  A = sum beta_i,  beta_T = sum_i beta_i prod_{t in T} xi_it,  Q = sum |beta_i|^r,  r = 1/s,
  bonus = |A| - 4^{1-s} Q^s   (IF price theta Q^s, theta = 4^{1-s}).
"""
import itertools
import numpy as np

n = 3
NU = 4
TS = [T for k in range(1, n + 1) for T in itertools.combinations(range(n), k)]
TNAMES = [''.join(str(t + 1) for t in T) for T in TS]
MASK = np.array([[1.0 if t in T else 0.0 for t in range(n)] for T in TS])  # 7 x 3


def coeffs(beta, ang):
    """beta: (N,) complex; ang: (N,3) angles of xi. Returns A, bT (7,)"""
    ph = np.exp(1j * (ang @ MASK.T))  # N x 7
    return beta.sum(), (beta[:, None] * ph).sum(axis=0)


def bonus(beta, ang, s):
    A, bT = coeffs(beta, ang)
    Q = np.sum(np.abs(beta) ** (1.0 / s))
    return abs(A) - 4.0 ** (1 - s) * Q ** s, A, bT, Q


def unpack(v, N):
    rho = np.abs(v[:N])
    ph = v[N:2 * N]
    ang = v[2 * N:2 * N + 3 * N].reshape(N, 3)
    return rho * np.exp(1j * ph), ang


def unpack_sym(v, N):
    """slot-symmetric: xi_it = exp(i*phi_i) for all t."""
    rho = np.abs(v[:N])
    ph = v[N:2 * N]
    phi = v[2 * N:3 * N]
    ang = np.repeat(phi[:, None], 3, axis=1)
    return rho * np.exp(1j * ph), ang
