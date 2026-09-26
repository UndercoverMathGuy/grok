"""eleg2_core.py -- shared algebra for the ELEGANT2 pass (junk, elegantly).

Objects (all symbols spelled out; see ELEGANT2.md SS0):

  n         number of operand slots;  D = nu = n+1 is both the homogeneity degree
            and the dimension of the chain space.
  N         number of atoms (neurons touching one frequency).
  v[l,j]    unit-modulus phase of atom l in slot j.
  c[l]      complex amplitude of atom l.
  C_T       = sum_l c_l prod_{j in T} v_lj,  the sign-pattern amplitude.
            T = empty  -> the CLEAN amplitude A;  T != empty -> junk.
  h(u)      = sum_l c_l prod_j (1 + u_j v_lj) = sum_T C_T u^T,
            the feature's generating polynomial ("what this feature outputs
            when the operands are probed at the torus point u").
  m         = min over u in T^n of Re h(u)   (the AIMED OUTPUT of the feature:
            the least the feature can be made to deliver by aiming a probe at it).
  m_A       = same but over the poly-torus |u_j| = A   ("aiming to depth A").
  M         = sum_l c_l m_l m_l^*, m_l = (1, v_l1, v_l1 v_l2, ...) the chain vector.

Everything float64/complex128 unless stated.
"""
import numpy as np
from itertools import combinations


# ---------------------------------------------------------------- populations

def rand_phases(rng, N, n):
    return np.exp(2j * np.pi * rng.random((N, n)))


def rand_pop(rng, N, n, kind="generic"):
    """Return (c, v). kind:
       generic   - random atoms, random phases (junk generic)
       junkfree  - random phases, c solved to kill all 2^n-1 junk patterns
       block     - the (n+1)-neuron common-phase block (exactly junk free)
       Mm        - the m-atom root-of-unity family of P3C
       nearjf    - junk-free plus a small random perturbation of c
    """
    if kind == "block":
        D = n + 1
        w = np.exp(-2j * np.pi * np.arange(D) / D)
        v = np.repeat(w[:, None], n, axis=1)
        c = np.ones(D, dtype=complex) / D
        return c, v
    if kind == "Mm":
        mm = N
        w = np.exp(-2j * np.pi * np.arange(mm) / mm)
        v = np.repeat(w[:, None], n, axis=1)
        c = np.ones(mm, dtype=complex) / mm
        return c, v
    v = rand_phases(rng, N, n)
    if kind == "generic":
        c = rng.normal(size=N) + 1j * rng.normal(size=N)
        return c, v
    # junk-free: solve  sum_l c_l prod_{j in T} v_lj = [T empty]
    Tmat = pattern_matrix(v)                     # (2^n, N)
    rhs = np.zeros(Tmat.shape[0], dtype=complex)
    rhs[0] = 1.0
    c, *_ = np.linalg.lstsq(Tmat, rhs, rcond=None)
    # add a null-space jitter so the sampler is not biased to min-norm solutions
    if Tmat.shape[1] > Tmat.shape[0]:
        u_, s_, vh_ = np.linalg.svd(Tmat)
        ns = vh_[np.sum(s_ > 1e-10):].conj().T
        if ns.shape[1] > 0:
            z = rng.normal(size=ns.shape[1]) + 1j * rng.normal(size=ns.shape[1])
            c = c + ns @ z
    if kind == "nearjf":
        c = c + (rng.normal(size=N) + 1j * rng.normal(size=N)) * 0.3 * np.abs(c).max()
    return c, v


def subsets(n):
    """All subsets of {0..n-1} as tuples, ordered by (size, lexicographic)."""
    out = []
    for k in range(n + 1):
        out.extend(combinations(range(n), k))
    return out


def pattern_matrix(v):
    """(2^n, N) matrix P with P[T, l] = prod_{j in T} v[l,j]."""
    N, n = v.shape
    S = subsets(n)
    P = np.ones((len(S), N), dtype=complex)
    for i, T in enumerate(S):
        for j in T:
            P[i] *= v[:, j]
    return P


def CT(c, v):
    """Vector of sign-pattern amplitudes C_T, indexed as in subsets(n)."""
    return pattern_matrix(v) @ c


# ---------------------------------------------------------------- h and min Re h

def h_at(c, v, u):
    """h(u) = sum_l c_l prod_j (1 + u_j v_lj)."""
    return np.sum(c * np.prod(1.0 + u[None, :] * v, axis=1))


def min_re_h(c, v, A=1.0, starts=40, rng=None, iters=60, tol=1e-14):
    """min over |u_j| = A of Re h(u), by EXACT cyclic coordinate descent.

    For fixed u_{-j}, h = a + b u_j is affine in u_j, so the exact minimiser of
    Re h over |u_j| = A is u_j = -A conj(b)/|b|, value Re a - A|b|.  Each sweep
    is therefore monotone and the descent converges to a coordinate-wise
    minimum; multi-start covers the global one.
    """
    N, n = v.shape
    if rng is None:
        rng = np.random.default_rng(0)
    best = np.inf
    best_u = None
    for s in range(starts):
        if s == 0:
            u = A * np.ones(n, dtype=complex)
        elif s <= n:                              # antipodes of atom 0..: good seeds
            l0 = (s - 1) % N
            u = -A * np.conj(v[l0])
        else:
            u = A * np.exp(2j * np.pi * rng.random(n))
        prev = np.inf
        for _ in range(iters):
            for j in range(n):
                fac = 1.0 + u[None, :] * v
                fac[:, j] = 1.0
                pr = np.prod(fac, axis=1)
                a = np.sum(c * pr)
                b = np.sum(c * v[:, j] * pr)
                if abs(b) < 1e-300:
                    val = a.real
                else:
                    u[j] = -A * np.conj(b) / abs(b)
                    val = a.real - A * abs(b)
            if prev - val < tol:
                break
            prev = val
        if val < best:
            best, best_u = val, u.copy()
    return best, best_u


def min_re_h_grid(c, v, P, k=1):
    """min over the GRID u_j in mu_p^n (the rows of the actual task) of Re h.

    u_j(a) = omega^{-2 k a_j};  as a ranges over Z_p^n and 2k is invertible the
    grid is all of (mu_p)^n.  Brute force -- only for small p^n.
    """
    N, n = v.shape
    w = np.exp(-2j * np.pi * np.arange(P) / P)
    best = np.inf
    idx = np.zeros(n, dtype=int)
    for t in range(P ** n):
        tt = t
        for j in range(n):
            idx[j] = tt % P
            tt //= P
        u = w[idx]
        val = h_at(c, v, u).real
        if val < best:
            best = val
    return best


# ---------------------------------------------------------------- chain matrix

def chain_vectors(v):
    """m_l in C^{n+1}: (m_l)_k = prod_{j<=k} v_lj (empty product 1)."""
    N, n = v.shape
    m = np.ones((N, n + 1), dtype=complex)
    for k in range(1, n + 1):
        m[:, k] = m[:, k - 1] * v[:, k - 1]
    return m


def chain_matrix(c, v):
    """M = sum_l c_l m_l m_l^*."""
    m = chain_vectors(v)
    return np.einsum("l,lk,lj->kj", c, m, m.conj())


def elem_sym(x, kmax=None):
    """e_0..e_kmax of the real non-negative vector x, by the product expansion."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if kmax is None:
        kmax = N
    e = np.zeros(kmax + 1)
    e[0] = 1.0
    for xi in x:
        for k in range(min(kmax, N), 0, -1):
            e[k] += e[k - 1] * xi
    return e


def Ek_matrix(M, k):
    """E_k(M) = coefficient of z^k in det(I + zM) = sum of principal k-minors."""
    D = M.shape[0]
    if k == 0:
        return 1.0 + 0j
    tot = 0.0 + 0j
    for L in combinations(range(D), k):
        tot += np.linalg.det(M[np.ix_(L, L)])
    return tot
