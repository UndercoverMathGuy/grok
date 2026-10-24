"""Check: a single product neuron at n = 2 has margin h <= 0 (any odd p >= 3, any real vectors).
Proof idea: h > 0 forces y(x) to be the strict argmax of w where u1 u2 > 0 and the strict argmin where u1 u2 < 0, and
u1 u2 != 0 everywhere; then all p^2 inputs lie on two lines x1 + x2 = c, impossible for p >= 3.
Numerics: random neurons and local maximisation of h / ||omega||^3."""
import numpy as np
from scipy.optimize import minimize
from n2lib import Grid, gaps

rng = np.random.default_rng(9)
for p in (3, 5, 7, 11):
    G = Grid(p)
    best = -np.inf
    for t in range(200):
        v = rng.normal(size=3 * p)
        U = v[:2 * p].reshape(1, 2, p); W = v[2 * p:].reshape(1, p)
        best = max(best, gaps(G, U, W).min() / np.linalg.norm(v) ** 3)
    for t in range(20):
        def f(v):
            U = v[:2 * p].reshape(1, 2, p); W = v[2 * p:].reshape(1, p)
            return -gaps(G, U, W).min() / np.linalg.norm(v) ** 3
        res = minimize(f, rng.normal(size=3 * p), method='Nelder-Mead', options={'maxiter': 4000})
        best = max(best, -res.fun)
    print(f"p={p}: max over random + optimised single neurons of h/||omega||^3 = {best:.3e}  (claim: <= 0)")
