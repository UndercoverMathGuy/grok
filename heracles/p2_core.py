"""P2 shared machinery: atoms -> sign-pattern amplitudes, junk energy, costs.

Setting (see GOPT_STATE.md).  One frequency.  N atoms.  Atom l has a complex
"clean amplitude" c_l and n unit-modulus slot phases v_lj = exp(i th_lj).
For a subset T of the n operand slots ("sign pattern"),

    C_T = sum_l c_l prod_{j in T} v_lj ,

C_empty is the CLEAN amplitude, the other 2^n - 1 are JUNK.  Normalisation
C_empty = 1 throughout.  Junk energy eta^2 := sum_{T != empty} |C_T|^2.

Cost of a population at exponent r = q/nu in (0,1]:  sum_l |c_l|^r.
"""
import numpy as np
from itertools import permutations


def masks(n):
    """All 2^n subsets of {0..n-1} as boolean rows; row 0 is the empty set."""
    m = np.zeros((1 << n, n), dtype=bool)
    for t in range(1 << n):
        for j in range(n):
            m[t, j] = (t >> j) & 1
    return m


def amps(c, V, M):
    """C_T for every T.  c: (N,) complex, V: (N,n) complex unit, M: (2^n,n) bool."""
    # prod over j in T of v_lj  ->  (2^n, N)
    P = np.where(M[:, None, :], V[None, :, :], 1.0).prod(axis=2)
    return P @ c


def pack(c, th):
    return np.concatenate([c.real, c.imag, th.ravel()])


def unpack(z, N, n):
    c = z[:N] + 1j * z[N:2 * N]
    th = z[2 * N:].reshape(N, n)
    return c, th


def junk_and_clean(z, N, n, M):
    c, th = unpack(z, N, n)
    V = np.exp(1j * th)
    C = amps(c, V, M)
    return C[0], C[1:]


def cost_r(c, r):
    return np.sum(np.abs(c) ** r)


def cost_r_grad(z, N, n, r):
    """d/dz of sum_l |c_l|^r ; theta part is zero."""
    c = z[:N] + 1j * z[N:2 * N]
    a = np.abs(c)
    a = np.maximum(a, 1e-300)
    w = r * a ** (r - 2)
    g = np.zeros_like(z)
    g[:N] = w * c.real
    g[N:2 * N] = w * c.imag
    return g


def _dC(z, N, n, M):
    """Jacobian pieces: C_T as a function of z.  Returns P (2^n,N) and
    dP/dth (2^n,N,n) so that dC_T/dth_lj = c_l * dP[T,l,j]."""
    c, th = unpack(z, N, n)
    V = np.exp(1j * th)
    P = np.where(M[:, None, :], V[None, :, :], 1.0).prod(axis=2)
    # d/dth_lj prod = i * v_lj * (prod without j) if j in T else 0
    dP = np.zeros((M.shape[0], N, n), dtype=complex)
    for j in range(n):
        sel = M[:, j]
        dP[sel, :, j] = 1j * P[sel, :]
    return c, P, dP


def eq_clean(z, N, n, M):
    """[Re C_empty - 1, Im C_empty] and its Jacobian."""
    c, P, dP = _dC(z, N, n, M)
    C0 = P[0] @ c
    J = np.zeros((2, z.size))
    J[0, :N] = P[0].real
    J[0, N:2 * N] = -P[0].imag
    J[1, :N] = P[0].imag
    J[1, N:2 * N] = P[0].real
    return np.array([C0.real - 1.0, C0.imag]), J


def junk_energy(z, N, n, M):
    """sum_{T != empty}|C_T|^2 and its gradient."""
    c, P, dP = _dC(z, N, n, M)
    C = P @ c
    g = np.zeros(z.size)
    for t in range(1, M.shape[0]):
        Ct = C[t]
        # d|Ct|^2/dRe c_l = 2 Re(conj(Ct) P[t,l]) ; /dIm c_l = 2 Re(conj(Ct) i P[t,l])
        g[:N] += 2 * (np.conj(Ct) * P[t]).real
        g[N:2 * N] += 2 * (np.conj(Ct) * 1j * P[t]).real
        gth = 2 * (np.conj(Ct) * (c[:, None] * dP[t])).real
        g[2 * N:] += gth.ravel()
    return float(np.sum(np.abs(C[1:]) ** 2)), g


def block_start(n, N, rng=None, noise=0.0):
    """The (n+1)-atom common-phase block, padded with `N-(n+1)` tiny atoms."""
    nb = n + 1
    w = np.exp(-2j * np.pi / nb)
    c = np.zeros(N, dtype=complex)
    th = np.zeros((N, n))
    c[:nb] = 1.0 / nb
    for l in range(nb):
        th[l, :] = -2 * np.pi * l / nb
    if N > nb and rng is not None:
        c[nb:] = 1e-6 * (rng.standard_normal(N - nb) + 1j * rng.standard_normal(N - nb))
        th[nb:, :] = rng.uniform(0, 2 * np.pi, (N - nb, n))
    if noise and rng is not None:
        c = c + noise * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / nb
        th = th + noise * rng.standard_normal((N, n))
    return pack(c, th)


def rand_start(n, N, rng):
    c = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / N
    th = rng.uniform(0, 2 * np.pi, (N, n))
    return pack(c, th)


def kill_points(V):
    """The (n+1)! kill points of an (n+1)-atom population.

    For each atom l0 and each bijection sigma from the n slots onto the n other
    atoms, u_j = -conj(v_{sigma(j),j}) makes every atom except l0 vanish.
    Returns an array (n+1)! x n of complex u, and the matching l0 index.
    """
    N, n = V.shape
    assert N == n + 1
    U, L0 = [], []
    for l0 in range(N):
        others = [l for l in range(N) if l != l0]
        for sg in permutations(others):
            U.append([-np.conj(V[sg[j], j]) for j in range(n)])
            L0.append(l0)
    return np.array(U), np.array(L0)


def hval(c, V, U):
    """h(u) = sum_l c_l prod_j (1 + u_j v_lj) evaluated at each row of U."""
    # (npts, N)
    F = np.prod(1.0 + U[:, None, :] * V[None, :, :], axis=2)
    return F @ c
