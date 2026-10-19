# Referee check of the DC slice radius r_p used in largen R3: r_p = sup{r : (v, v^2) in conv{(w^a, w^{2a})} for all |v|<=r}.
# p=5: the 5 points form a simplex in R^4, barycentric weights t_a = (1/5)(1 + 2Re(v w^{-a}) + 2Re(v^2 w^{-2a})),
#      min over directions is 3/4 - 2r^2  =>  r_5 = sqrt(3/8) exactly.
# p=7, 11, 13: LP feasibility, using the symmetry (v,v^2) -> (w v, w^2 v^2) and conjugation (angles in [0, pi/p]),
#      a fine angle grid, and bisection; reported as the min over the grid (an upper estimate of r_p up to grid error).
import math, numpy as np
from scipy.optimize import linprog
print(f"p=5 closed form r_5 = sqrt(3/8) = {math.sqrt(3/8):.6f}")
def feasible(v, p):
    a = np.arange(p); z = np.exp(2j * np.pi * a / p)
    Aeq = np.vstack([np.ones(p), z.real, z.imag, (z ** 2).real, (z ** 2).imag])
    beq = np.array([1, v.real, v.imag, (v * v).real, (v * v).imag])
    return linprog(np.zeros(p), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs').status == 0
for p in [3, 5, 7, 11]:
    worst = 1.0; worst_phi = None
    for phi in np.linspace(0, math.pi / p, 401):
        lo, hi = 0.0, 1.0
        for it in range(34):
            mid = (lo + hi) / 2
            if feasible(mid * np.exp(1j * phi), p): lo = mid
            else: hi = mid
        if lo < worst: worst, worst_phi = lo, phi
    print(f"p={p}: radial sup minimised over 401 angles in [0,pi/p]: r_p ~ {worst:.6f} at phi={worst_phi:.4f}", flush=True)
