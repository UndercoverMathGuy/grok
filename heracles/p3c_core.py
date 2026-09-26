"""P3C shared machinery.  Deliberately an INDEPENDENT implementation of the
one-frequency atom algebra (it does not import p2_core / p3a_*), so that the
re-check of P2's chord numbers is a genuine second opinion.

Setting (GOPT_STATE.md, one frequency, `n` operand slots):

  atom l  =  complex clean amplitude c_l  and  n unit phases v_lj = exp(i th_lj)
  h(u)    =  sum_l c_l prod_{j=1..n} (1 + u_j v_lj)          (multi-affine in u)
  C_T     =  coefficient of prod_{j in T} u_j  =  sum_l c_l prod_{j in T} v_lj

  C_empty  ("clean")  is normalised to 1 throughout;
  eta^2 := sum_{T != empty} |C_T|^2          ("junk energy")
  cost   := sum_l |c_l|^r ,   r = q/nu in (0,1].

Plain reading: one neuron ("atom") at one frequency emits the wanted cosine of
the SUM of the operands (that is C_empty) plus 2^n - 1 unwanted cosines, one for
each way of flipping the sign of a subset T of the operands (those are C_T).
A population of atoms can cancel the junk, but only by splitting the clean
amplitude over several neurons, which costs, because r < 1.

Here C_T is built by an explicit multi-affine convolution over the slots, i.e.
by literally expanding the product one slot at a time -- a different code path
from p2_core's mask/product matrix.
"""
import numpy as np


def coeffs(c, V):
    """C_T for all 2^n subsets T, indexed by the bitmask sum_{j in T} 2^j.

    c : (N,) complex atom amplitudes.  V : (N,n) complex unit slot phases.
    Built slot by slot: after slot j the array holds, for each atom, the
    coefficients of the partial product prod_{j' <= j} (1 + u_{j'} v_{l j'}).
    """
    c = np.asarray(c, dtype=complex)
    V = np.asarray(V, dtype=complex)
    N, n = V.shape
    A = np.zeros((N, 1), dtype=complex)
    A[:, 0] = c
    for j in range(n):
        B = np.zeros((N, A.shape[1] * 2), dtype=complex)
        B[:, :A.shape[1]] = A                       # u_j absent
        B[:, A.shape[1]:] = A * V[:, j][:, None]    # u_j present  -> bit j set
        A = B
    return A.sum(axis=0)


def report(c, V, r):
    """Exact fp64 audit of one configuration: (clean, eta, cost)."""
    C = coeffs(c, V)
    clean = C[0]
    junk = C[1:]
    eta = float(np.sqrt(np.sum(np.abs(junk) ** 2)))
    cost = float(np.sum(np.abs(np.asarray(c)) ** r))
    return clean, eta, cost


# ---------------------------------------------------------------- packing
def unpack(z, N, n):
    c = z[:N] + 1j * z[N:2 * N]
    V = np.exp(1j * z[2 * N:].reshape(N, n))
    return c, V


def pack(c, th):
    c = np.asarray(c, dtype=complex)
    return np.concatenate([c.real, c.imag, np.asarray(th, dtype=float).ravel()])


def block(n, r=None):
    """The (n+1)-atom common-phase block: c_l = 1/(n+1), v_lj = w^{-l},
    w = exp(2 pi i/(n+1)).  Junk-free, cost (n+1)^{1-r}."""
    nb = n + 1
    c = np.full(nb, 1.0 / nb, dtype=complex)
    th = np.zeros((nb, n))
    for l in range(nb):
        th[l, :] = -2 * np.pi * l / nb
    return c, th


def kappa0(n, r):
    """A(0) = Cl2_r(0)^{1/r} = (n+1)^{(1-r)/r}: clean amplitude per unit cost
    at zero junk (the block)."""
    return (n + 1.0) ** ((1.0 - r) / r)


def rho_need(n, r):
    """(1 - 1/kappa0)/sqrt(2^n - 1): the chord slope P3A's criterion allows."""
    return (1.0 - 1.0 / kappa0(n, r)) / np.sqrt(2 ** n - 1.0)


def rho_of(eta, cost, n, r):
    """rho(eta) = [1 - A(eta)/kappa0]/eta with A = cost^{1/r}."""
    return (1.0 - (cost ** (1.0 / r)) / kappa0(n, r)) / eta
