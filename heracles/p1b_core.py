"""P1B core: the (KL) atom system as a complex multi-affine cubature rule on the torus.

Symbols (all defined in P1B.md too):
  n      number of operand slots.
  N      number of atoms.
  v      (N, n) array of unit-modulus complex numbers; v[l, j] is atom l's phase in slot j.
  T      a subset of {0..n-1} ("sign pattern"); encoded as a bitmask 0 .. 2^n - 1.
  M      the 2^n x N matrix  M[T, l] = prod_{j in T} v[l, j].
  (KL)   M c = A e_0, i.e. sum_l c_l prod_{j in T} v_lj = A [T empty].
  z      the normalised atom vector z = c / A; then sum_l z_l = 1 and every other
         multi-affine moment of z vanishes: z is a COMPLEX CUBATURE RULE that
         integrates every multi-affine polynomial on the torus exactly (against Haar).
  r      = q / nu in (0, 1]; the cost exponent.
  cost   sum_l |z_l|^r.  P1 claims cost >= (n+1)^(1-r).

Nothing here is an approximation: for N = 2^n the rule z is unique (generic v) and is
obtained by one linear solve.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def moment_matrix(v):
    """M[T, l] = prod_{j in T} v[l, j], rows in bitmask order T = 0 .. 2^n - 1."""
    N, n = v.shape
    M = np.ones((1 << n, N), dtype=complex)
    for T in range(1 << n):
        for j in range(n):
            if T >> j & 1:
                M[T] *= v[:, j]
    return M


def rule_from_config(v):
    """Unique cubature rule z for N = 2^n (square case).  Returns z with sum z = 1."""
    M = moment_matrix(v)
    rhs = np.zeros(M.shape[0], dtype=complex)
    rhs[0] = 1.0
    return np.linalg.solve(M, rhs)


def rule_cofactor(v):
    """Same as rule_from_config but built from cofactors -- stable when det M -> 0 only
    if the cofactors vanish too.  Used as an independent check."""
    M = moment_matrix(v)
    N = M.shape[1]
    d = np.empty(N, dtype=complex)
    for l in range(N):
        sub = np.delete(M[1:], l, axis=1)
        d[l] = (-1) ** l * np.linalg.det(sub)
    return d / d.sum()


def cost(z, r):
    return float(np.sum(np.abs(z) ** r))


def target(n, r):
    return (n + 1) ** (1.0 - r)


def ratio(v, r):
    """cost / target for the square case."""
    z = rule_from_config(v)
    return cost(z, r) / target(v.shape[1], r)


def angles_to_v(theta, N, n, gauge=True):
    """theta: flat array of free angles.  With gauge=True atom 0 is pinned at (1,..,1)
    (the slot-rotation gauge v[:, j] -> lam_j v[:, j] leaves every |z_l| unchanged)."""
    v = np.ones((N, n), dtype=complex)
    k = 0
    start = 1 if gauge else 0
    for l in range(start, N):
        for j in range(n):
            v[l, j] = np.exp(1j * theta[k])
            k += 1
    return v


def n_free_angles(N, n, gauge=True):
    return (N - 1) * n if gauge else N * n


# ----------------------------------------------------------------- known configs

def block(n):
    """The (n+1)-atom common-phase block: v[l, j] = w^{-l}, w = exp(2 pi i/(n+1)).
    Its rule is z_l = 1/(n+1): the extremal cubature rule of P1."""
    w = np.exp(2j * np.pi / (n + 1))
    l = np.arange(n + 1)
    return np.tile((w ** (-l))[:, None], (1, n)).astype(complex)


def hadamard_config(n):
    """The 2^n-atom sign config v[l, j] = +-1 (all sign vectors).  Its rule is the
    uniform one z_l = 2^-n: a second, non-extremal, exact cubature rule."""
    N = 1 << n
    v = np.empty((N, n), dtype=complex)
    for l in range(N):
        for j in range(n):
            v[l, j] = -1.0 if (l >> j & 1) else 1.0
    return v


def rule_general(v, tol=1e-10):
    """Least-squares/min-norm rule for any N (may be infeasible for N < 2^n).
    Returns (z, residual)."""
    M = moment_matrix(v)
    rhs = np.zeros(M.shape[0], dtype=complex)
    rhs[0] = 1.0
    z, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    res = float(np.linalg.norm(M @ z - rhs))
    return z, res


def nullspace(v, tol=1e-9):
    """Basis of {k : M k = 0} -- the directions along which a rule can be moved."""
    M = moment_matrix(v)
    _, s, Vh = np.linalg.svd(M)
    N = M.shape[1]
    rank = int(np.sum(s > tol * max(s[0], 1.0)))
    return Vh[rank:].conj().T  # (N, N - rank)
