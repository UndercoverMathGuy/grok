"""Sanity check of Theorem 2 (linear matrix relaxations see complex rank, not real rank).

Per-frequency tables are real tensors X in (R^2)^{(x) nu} (functions prod_t (a_t cos th_t + b_t sin th_t)).  For a real-linear
M : (R^2)^{(x)nu} -> C^{d x d'} let C_M = sup over real rank-one T of ||M(T)||_{S_r}^r / ||T||_F^r (budget constant).
Claim: for a generic interference-free block f (nu atoms) one has ||M(f)||_{S_r}^r <= (2/nu)^{1-r} C_M N_r(f)^r,
where N_r(f)^r = sum over the block atoms of ||T_l||_F^r (Lemma 7 makes the block optimal).
We test (i) flattenings (rank-one preserving: the best case, ratio (2/nu)^{1-r} * const), (ii) random linear maps,
(iii) random maps composed with flattenings, (iv) the chain lower-triangular map; the ratio printed is
 ||M(f)||^r / (C_M N_r(f)^r)  and must be <= (2/nu)^{1-r} (< 1 for nu >= 3).
"""
import itertools
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(2)


def rank1(angles, norms=None):
    nu = len(angles)
    T = np.array([1.0])
    for k in range(nu):
        v = np.array([np.cos(angles[k]), np.sin(angles[k])]) * (1.0 if norms is None else norms[k])
        T = np.multiply.outer(T, v) if T.ndim > 0 and T.size > 1 else v.copy() if T.size == 1 and k == 0 else np.multiply.outer(T, v)
    return T


def block(nu, alpha0):
    """paper's IF block (Prop 9) with a global phase offset: atoms u_t = cos(th + a_l + alpha0_t), output cos(th + n a_l + alpha0_0)"""
    atoms = []
    for l in range(nu):
        a = np.pi * l / nu
        angs = [-(nu - 1) * a + alpha0[0]] + [a + alpha0[t] for t in range(1, nu)]
        atoms.append(rank1(angs))
    return atoms


def sr(Mx, r):
    return np.sum(np.linalg.svd(Mx, compute_uv=False) ** r)


def budget_const(Mmap, nu, r, tries=30):
    best = 0
    for _ in range(tries):
        f = lambda ang: -sr(Mmap(rank1(ang)), r)
        res = minimize(f, rng.uniform(0, 2 * np.pi, nu), method='Nelder-Mead', options={'maxiter': 3000})
        best = max(best, -res.fun)
    return best


for nu in [3, 4]:
    for b in [2.0, 2.5]:
        if b > nu:
            continue
        r = b / nu
        bound = (2 / nu) ** (1 - r)
        maps = {}
        # flattening slot 0 vs rest
        maps['flatten'] = lambda X: X.reshape(2, -1).astype(complex)
        # flattening slots {0,1} vs rest
        if nu >= 4:
            maps['flatten2'] = lambda X: X.reshape(4, -1).astype(complex)
        R1 = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2)); R2 = rng.normal(size=(2 ** (nu - 1), 3)) + 1j * rng.normal(size=(2 ** (nu - 1), 3))
        maps['A flat B'] = lambda X, R1=R1, R2=R2: R1 @ X.reshape(2, -1) @ R2
        G = rng.normal(size=(2 ** nu, 4, 4)) + 1j * rng.normal(size=(2 ** nu, 4, 4))
        maps['random 4x4'] = lambda X, G=G: np.tensordot(X.ravel(), G, axes=1)
        for name, Mmap in maps.items():
            C = budget_const(Mmap, nu, r)
            worst = 0
            for trial in range(20):
                alpha0 = rng.uniform(0, 2 * np.pi, nu)
                atoms = block(nu, alpha0)
                f = sum(atoms)
                Nr = sum(np.linalg.norm(T) ** r for T in atoms)
                ratio = sr(Mmap(f), r) / (C * Nr)
                worst = max(worst, ratio)
            print(f"nu={nu} b={b} r={r:.3f} map={name:11s}: C={C:.4f}  max ratio over 20 generic blocks = {worst:.4f}   bound (2/nu)^(1-r) = {bound:.4f}  {'OK' if worst <= bound + 1e-6 else 'VIOLATED'}")
