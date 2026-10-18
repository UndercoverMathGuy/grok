"""Single-feature free relaxation R_free({1}) evaluated at simple atom systems.
Only the field Phi_1 exists; its own coefficients (characters 2*1_T) are those of the atoms, all its other Fourier coefficients
are free (partner, DC-type, arbitrary).  Margin = exact LP.  Cost = atomic Q = sum |beta_i|^r (exact own-mode price).
Ratio = h / (nu^{1-s} kappa_min Q^s)  (IF single-feature normalisation).  Ratio > 1 means R_free({1}) beats the IF price.
"""
import sys
import numpy as np
from mrel import Model, subsets

rng = np.random.default_rng(1)


def own_from_atoms(beta, xi, n):
    return {T: np.conj(sum(b * np.prod(x[list(T)]) for b, x in zip(beta, xi))) for T in subsets(n)}


def ratio(M, beta, xi, n, s):
    nu = n + 1; r = 1 / s; p = M.p
    a = {1: own_from_atoms(beta, xi, n)}
    h, _ = M.margin(a)
    Q = np.sum(np.abs(beta) ** r)
    kmin = 1 - np.cos(2 * np.pi / p)
    return h / (nu ** (1 - s) * kmin * Q ** s)


for (p, n) in [(5, 2), (7, 2), (5, 3), (7, 3), (5, 4), (3, 3), (3, 4)]:
    for b in [2.0, 2.5]:
        nu = n + 1; s = nu / b
        if b > nu:
            continue
        M = Model(p, n, [1], mode='free', s=s)
        Mc = Model(p, n, [1], mode='C', s=s)
        res = []
        # single atom, xi = 1
        res.append(('1 atom', ratio(M, np.array([1.0]), np.ones((1, n)), n, s), ratio(Mc, np.array([1.0]), np.ones((1, n)), n, s)))
        # best of random single atoms
        best = max(ratio(M, np.array([1.0]), np.exp(1j * rng.uniform(0, 2 * np.pi, (1, n))), n, s) for _ in range(6))
        res.append(('1 atom rnd', best, None))
        # IF block
        om = np.exp(2j * np.pi / nu)
        xi_b = np.array([[om ** k] * n for k in range(nu)])
        res.append(('IF block', ratio(M, np.full(nu, 1 / nu), xi_b, n, s), None))
        # two aligned atoms (P12 type at n=2): xi_2 = -xi_1 on slot 1 ...
        if n >= 2:
            xi2 = np.ones((2, n), dtype=complex); xi2[1, :] = -1
            res.append(('2 atoms +-', ratio(M, np.array([0.5, 0.5]), xi2, n, s), None))
        print(f"p={p} n={n} b={b}: " + "; ".join(f"{nm}: {v:.4f}" + (f" (classC {c:.4f})" if c is not None else "") for nm, v, c in res), flush=True)
