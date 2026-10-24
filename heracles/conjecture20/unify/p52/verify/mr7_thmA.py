"""Math referee: spot-check the imported payments used in R7/R8/R9 on random and adversarial atom systems:
Part I Thm A: |A| - sqrt(|B|^2+|C|^2+|D|^2)/(2 sqrt3) <= 3^{1-s} Q^s  for every s in [1, 3/2]  (s-independent left side),
and the s-monotonicity of 3^{1-s} Q^s itself (NOT monotone in general: shows why R8 needs bound-wise monotonicity)."""
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(3)
def own(v, k):
    vr = np.exp(v[:k]); pp = np.exp(1j*v[k:2*k]); xi = np.exp(1j*v[2*k:3*k]); eta = np.exp(1j*v[3*k:4*k])
    b = vr*pp
    return vr, b.sum(), (b*xi).sum(), (b*eta).sum(), (b*xi*eta).sum()
def gap(v, k, s):
    vr, A, B, C, D = own(v, k)
    return (abs(A) - np.sqrt(abs(B)**2+abs(C)**2+abs(D)**2)/(2*np.sqrt(3)) - 3**(1-s)*np.sum(vr**(1/s))**s)/vr.sum()
worst = -np.inf
for t in range(20000):
    k = rng.integers(1, 7); s = rng.uniform(1, 1.5)
    worst = max(worst, gap(rng.normal(size=4*k)*rng.uniform(0.1, 3), k, s))
adv = -np.inf
for st in range(30):
    k = 1 + st % 5; s = [1.0, 1.2, 1.35, 1.5][st % 4]
    r = minimize(lambda v: -gap(v, k, s), rng.normal(size=4*k), method='Nelder-Mead', options=dict(maxiter=4000))
    adv = max(adv, -r.fun)
print(f"Thm A: random max (lhs - 3^(1-s)Q^s)/L = {worst:.3e}; adversarial (Nelder-Mead, 1-5 atoms) max = {adv:.3e}  (<= 0)")
# P_1(s) = 3^{1-s} ||beta||_{1/s} is not monotone in s: many equal atoms
for N in (1, 3, 10, 100):
    P1 = [3**(1-s)*(N*(1/N)**(1/s))**s for s in (1.0, 1.25, 1.5)]
    print(f"   N={N} equal aligned atoms: P_1(s=1,1.25,1.5) = {np.round(P1,4)}")
