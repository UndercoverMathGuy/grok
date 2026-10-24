"""Math referee: R14 (7,2) two-feature simple bound -- independent float evaluation of
   sup_{tau, x1, x2}  min_mu [ sum_j K_j(mu) A_j - (7/2) min_d mu_d max_j A_j x_j ] / [ (7/2) 3^{-s} (sum_j (F_j A_j)^r)^s ]
with the exact LP over symmetric mu (any mu is admissible, the prover takes the best), F_j from R13.
Expect: < 0.999 for s <= 1.175, < 0.9995 for s <= 1.182, and > 1 near s = 1.2 (reported phase-blind cap)."""
import numpy as np
from scipy.optimize import linprog, minimize

kap = lambda j, d: 1 - np.cos(2 * np.pi * j * d / 7)


def lpmin(A1, A2, M):
    # variables mu1, mu2, mu3, t ; minimise sum_d 2 mu_d (k1d A1 + k2d A2) - 3.5 t M ; t <= mu_d ; 2 sum mu = 1
    c = np.array([2 * (kap(1, d) * A1 + kap(2, d) * A2) for d in (1, 2, 3)] + [-3.5 * M])
    Aub = np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]], float)
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(3), A_eq=np.array([[2, 2, 2, 0.0]]), b_eq=[1], bounds=[(0, None)] * 4, method='highs')
    return res.fun


def ratio(tau, x1, x2, s):
    r = 1 / s; c1 = 1 - 3 ** (1 - s)
    A1, A2 = 1 - tau, tau
    M = max(A1 * x1, A2 * x2)
    F1 = max(1 - c1 * x1, 3 ** (1 - s) * max(1, x1)); F2 = max(1 - c1 * x2, 3 ** (1 - s) * max(1, x2))
    den = 3.5 * 3 ** (-s) * ((F1 * A1) ** r + (F2 * A2) ** r) ** s
    return lpmin(A1, A2, M) / den


def sup_at(s, rng):
    best = -np.inf; arg = None
    taus = np.linspace(0.0, 1.0, 81); xs = np.concatenate([np.linspace(0, 2.5, 51), [4, 8]])
    for tau in taus:
        for x1 in xs[::2]:
            for x2 in xs[::2]:
                v = ratio(tau, x1, x2, s)
                if v > best:
                    best, arg = v, (tau, x1, x2)
    # local refinement from the best few points
    def neg(v):
        tau, x1, x2 = v
        if not (0 <= tau <= 1 and x1 >= 0 and x2 >= 0):
            return 10
        return -ratio(tau, x1, x2, s)
    starts = [arg] + [(rng.random(), rng.uniform(0, 2), rng.uniform(0, 2)) for _ in range(8)]
    for st in starts:
        r = minimize(neg, st, method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-10, maxiter=2000))
        if -r.fun > best:
            best, arg = -r.fun, tuple(r.x)
    return best, arg


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    for s in (1.0, 1.1, 1.15, 1.175, 1.182, 1.19, 1.2):
        b, a = sup_at(s, rng)
        print(f"s={s:.3f} (b={3/s:.4f}): sup simple bound = {b:.5f} at tau,x1,x2 = {np.round(a,4)}")
