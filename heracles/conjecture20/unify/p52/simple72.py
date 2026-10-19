"""(7,2), two features S={1,2}, beta_3 = 0: a simple phase-blind bound via the FREE relaxation with rigid zero-leakage
single-character tests.  For a symmetric probability mu on Z_7^* (mu_d = mu_{-d}, d=1,2,3) and test weights w_j >= 0,
sum_j w_j <= 1:
   h <= sum_j K_j(mu) |A_j| - (7/2) min_d mu_d * sum_j w_j M_j          (M_j = max_T |beta_{j,T}|)
Cost: 3^{1-s} B_7 P^s >= ( sum_j (F_j |A_j|)^r )^s,  F_j = max(1 - c1(s) M_j/|A_j|, 3^{1-s} max(1, M_j/|A_j|)) (Note payments).
We compute  sup_{A,M} min_{mu,w} ratio  numerically.
"""
import sys
import numpy as np
from scipy.optimize import minimize, linprog

p = 7
s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2
r = 1 / s
c1 = 1 - 3 ** (1 - s)
kap = np.array([[1 - np.cos(2 * np.pi * j * d / p) for d in (1, 2, 3)] for j in (1, 2, 3)])   # kap[j-1, d-1]


def Gamma7(s):
    best = 0
    for t in range(40):
        y0 = np.random.default_rng(t).uniform(0.1, 1, 3)
        f = lambda y: -np.min(kap.T @ (np.abs(y) / np.sum(np.abs(y) ** r) ** s))
        res = minimize(f, y0, method='Nelder-Mead', options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 6000})
        best = max(best, -res.fun)
    return best


G = Gamma7(s)


def best_bound(A1, A2, M1, M2):
    """min over mu (symmetric, 3 params) and w of the bound numerator  (LP in (mu, w-products) after fixing which
    feature gets detection: detection is linear in mu_min, so use t = min mu as a variable)"""
    # variables: mu1, mu2, mu3, t ; minimise sum_j K_j |A_j| - 3.5 t max(M1, M2)  s.t. t <= mu_d, 2(mu1+mu2+mu3) = 1
    M = max(M1, M2)
    c = np.array([2 * (kap[0, 0] * A1 + kap[1, 0] * A2), 2 * (kap[0, 1] * A1 + kap[1, 1] * A2), 2 * (kap[0, 2] * A1 + kap[1, 2] * A2), -3.5 * M])
    Aub = np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]])
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(3), A_eq=np.array([[2, 2, 2, 0]]), b_eq=[1], bounds=[(0, None)] * 4, method='highs')
    return res.fun


def F(A, M):
    return max(1 - c1 * M / A, 3 ** (1 - s) * max(1, M / A))


def ratio(v):
    A1 = 1.0; A2 = abs(v[0]); M1 = abs(v[1]); M2 = abs(v[2])
    num = best_bound(A1, A2, M1, M2)
    den = G * ((F(A1, M1) * A1) ** r + (F(A2, M2) * A2) ** r) ** s
    return num / den


rng = np.random.default_rng(0)
best = (-1, None)
for k in range(60):
    v0 = rng.uniform(0, 1.8, 3)
    res = minimize(lambda v: -ratio(v), v0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 4000})
    if -res.fun > best[0]:
        best = (-res.fun, res.x)
print(f"s={s} Gamma7={G:.5f}: sup ratio (simple phase-blind free bound) = {best[0]:.5f} at A2/A1, M1, M2 = {np.round(np.abs(best[1]),4)}")
