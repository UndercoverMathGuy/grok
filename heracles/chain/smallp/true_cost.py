# NUMERICAL: true minimal population cost at given junk moduli j (G = 1) vs the PSD-matrix lower bound max_e LHS(e, j).
import numpy as np
from scipy.optimize import minimize
from pi_feasible import LHS
rng = np.random.default_rng(1)
def true_min(j, starts=60):
    best = 9
    for N in (1, 2, 3, 4):
        for _ in range(starts):
            def obj(v):
                x = np.exp(v[:N]); c = x * np.exp(1j * v[N:2 * N]); v1 = np.exp(1j * v[2 * N:3 * N]); v2 = np.exp(1j * v[3 * N:])
                G = c.sum(); J = np.array([abs((c * v1).sum()), abs((c * v2).sum()), abs((c * v1 * v2).sum())])
                pen = 200 * (abs(G - 1) ** 2 + ((J - np.array(j)) ** 2).sum())
                return (x ** (2 / 3)).sum() + pen
            r = minimize(obj, np.r_[rng.normal(0, .5, N), rng.uniform(0, 6.3, 3 * N)], method='Nelder-Mead',
                         options=dict(maxiter=20000, xatol=1e-10, fatol=1e-12))
            best = min(best, r.fun)
    return best
for j in [(0.83, 0.83, 0.97), (0.0, 0.97, 0.0), (0.66, 0.66, 0.99), (1.0, 1.0, 1.0), (0.4, 0.4, 0.97)]:
    e = np.linspace(max(1, max(j)), np.sqrt(3), 400); mb = LHS(e, *j).min()
    print(f"j={j}: true min cost ~ {true_min(j):.4f}   matrix bound min_e LHS = {mb:.4f}   (3^(1/3) = 1.4422)")
