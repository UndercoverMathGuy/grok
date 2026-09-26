"""Constructed networks with exactly known answers (positive controls, NORMS E17).

1. nu_block(p, n, y): the nu-block product network of cyclic.tex prop:block / appendix app:block.
   For every j with y_j > 0 and lambda_j = (2^n y_j / nu)^{1/nu}, alpha_l = pi l / nu (l = 0..n):
       u_t^{(l)}(a) = lambda_j cos(2 pi j a / p + alpha_l)   (t = 1..n)
       w^{(l)}(c)   = lambda_j cos(2 pi j c / p + n alpha_l)
   Total table: sum_j y_j cos(2 pi j (a_1 + .. + a_n - c) / p), so beta_j = y_j exactly, the network is
   interference-free, and with y = y* (budget 1) its normalised L_{2,q} margin is
   gamma_IF = nu^{-(nu-q)/q} B_p Gamma_s(p).

2. morwani_dense(p, y): a power network at n = 2 (the quadratic network of Morwani et al.).
   For every j with y_j > 0, 9 neurons with phases phi_1, phi_2 in {0, 2pi/3, 4pi/3}, phi_3 = phi_1 + phi_2:
       u(a) = lam cos(2 pi j a / p + phi_1), v(b) = lam cos(2 pi j b / p + phi_2),
       w(c) = lam cos(2 pi j c / p + phi_3),  lam = (2 y_j / 9)^{1/3}.
   Each neuron (u + v)^2 w carries the clean term (lam^3 / 2) cos(2 pi j (a + b - c)/p); every other
   trigonometric term has a phase in {phi_1+phi_2, 3phi_1+phi_2, phi_1-phi_2, phi_1+3phi_2, 2phi_1+2phi_2,
   2phi_1, 2phi_2}, whose sum over the 3 x 3 phase grid vanishes. The table is therefore exactly
   sum_j y_j cos(2 pi j (a + b - c)/p). With y flat (y_j = 2/(p-1)) at q = 3 its normalised margin is
   2 B_p p/(p-1) = n! B_p Gamma_1(p), the power-network ceiling (approach.tex cor:dense).
"""
from __future__ import annotations

import numpy as np


def _cos(p: int, j: int, phase: float, amp: float) -> np.ndarray:
    return amp * np.cos(2 * np.pi * ((j * np.arange(p)) % p) / p + phase)


def nu_block(p: int, n: int, y, width: int | None = None):
    """nu-block product network for amplitudes y (length h, y_j >= 0). Returns float64 (U (1,n,m,p), W (1,m,p)).

    width: total number of neurons (>= nu * |supp y|); extra neurons are exactly zero.
    """
    y = np.asarray(y, dtype=float)
    h = (p - 1) // 2
    assert y.shape == (h,), (y.shape, h)
    nu = n + 1
    Us, Ws = [], []
    for j in range(1, h + 1):
        yj = y[j - 1]
        if yj <= 0:
            continue
        lam = (2.0 ** n * yj / nu) ** (1.0 / nu)
        for l in range(nu):
            al = np.pi * l / nu
            Us.append(np.stack([_cos(p, j, al, lam) for _ in range(n)]))
            Ws.append(_cos(p, j, n * al, lam))
    m0 = len(Ws)
    m = m0 if width is None else width
    assert m >= m0, (m, m0)
    U = np.zeros((1, n, m, p))
    W = np.zeros((1, m, p))
    if m0:
        U[0, :, :m0, :] = np.stack(Us, axis=1)
        W[0, :m0, :] = np.stack(Ws)
    return U, W


def morwani_dense(p: int, y=None, width: int | None = None):
    """Morwani-type dense power network at n = 2 (see module docstring). y defaults to flat 2/(p-1)."""
    h = (p - 1) // 2
    y = np.full(h, 2.0 / (p - 1)) if y is None else np.asarray(y, dtype=float)
    phases = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    Us, Ws = [], []
    for j in range(1, h + 1):
        yj = y[j - 1]
        if yj <= 0:
            continue
        lam = (2.0 * yj / 9.0) ** (1.0 / 3.0)
        for f1 in phases:
            for f2 in phases:
                Us.append(np.stack([_cos(p, j, f1, lam), _cos(p, j, f2, lam)]))
                Ws.append(_cos(p, j, f1 + f2, lam))
    m0 = len(Ws)
    m = m0 if width is None else width
    assert m >= m0
    U = np.zeros((1, 2, m, p))
    W = np.zeros((1, m, p))
    U[0, :, :m0, :] = np.stack(Us, axis=1)
    W[0, :m0, :] = np.stack(Ws)
    return U, W


def target_table(p: int, n: int, y) -> np.ndarray:
    """sum_j y_j cos(2 pi j (a_1 + .. + a_n - c) / p) on the full grid, shape (p,)*(n+1)."""
    y = np.asarray(y, dtype=float)
    idx = np.indices((p,) * (n + 1))
    d = (idx[:n].sum(0) - idx[n]) % p
    T = np.zeros((p,) * (n + 1))
    for j in range(1, len(y) + 1):
        if y[j - 1] != 0:
            T += y[j - 1] * np.cos(2 * np.pi * ((j * d) % p) / p)
    return T
