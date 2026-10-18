# r_p = sup{ r : (v, v^2) in conv{(rho^a, rho^{2a}) : a in Z_p} for all |v| <= r }  (slice radius for single-frequency nets with DC)
import math, numpy as np
from scipy.optimize import linprog
def feasible(v, p):
    a = np.arange(p); z = np.exp(2j*np.pi*a/p)
    Aeq = np.vstack([np.ones(p), z.real, z.imag, (z**2).real, (z**2).imag])
    beq = np.array([1, v.real, v.imag, (v*v).real, (v*v).imag])
    r = linprog(np.zeros(p), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    return r.status == 0
for p in [3, 5, 7, 11, 13]:
    lo, hi = 0.0, 1.0
    for it in range(40):
        mid = (lo + hi) / 2
        ok = all(feasible(mid*np.exp(1j*phi), p) for phi in np.linspace(0, 2*np.pi, 721))
        lo, hi = (mid, hi) if ok else (lo, mid)
    print(f"p={p}: r_p ~ {lo:.4f}   (class-C radius cos(pi/p) = {math.cos(math.pi/p):.4f})")
