"""Inspect the free-relaxation optimum for a single atom at (5,2): which Fourier modes of Phi_1, Phi_2 does it use?"""
import sys
import numpy as np
from p52lib import *

b = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
s = 3 / b; r = 1 / s
beta = np.array([1.0 + 0j]); xi = np.ones((1, 2), dtype=complex)
A, bT = own_from_atoms(beta, xi)
hf, res = h_free(A, bT, return_dual=True)
N = 25
g = res.x[:4 * N].reshape(4, N)          # g[di, x]
print(f"single atom, beta=1, xi=1: h_free = {hf:.6f} ; trivial kappa_a = {KA:.6f}; IF-price*Gamma = {3**(1-s)*Gamma(s):.6f}")
# recover Phi_j(x) = -(2/5) sum_d g(x,d) rho^{j d}
Phi = {}
for j in [1, 2]:
    Phi[j] = -(2 / 5) * sum(g[di] * rho ** (j * d) for di, d in enumerate(DELS))
    Phi[j] = Phi[j].reshape(5, 5)
for j in [1, 2]:
    print(f"Phi_{j} Fourier coefficients (|coef| > 1e-6):")
    for c1 in range(5):
        for c2 in range(5):
            v = fcoef(Phi[j], (c1, c2))
            if abs(v) > 1e-6:
                k = ((c1 - j) % 5, (c2 - j) % 5)
                print(f"   chi=({c1},{c2})  input freqs k={k}  coef={v:.4f}  |.|={abs(v):.4f}")
# dual (marginals)
print("eq marginals", np.round(res.eqlin.marginals, 4))
lam = -res.ineqlin.marginals
q = lam.reshape(4, 5, 5)
print("dual mass per delta:", np.round(q.sum(axis=(1, 2)), 4))
for di in range(4):
    print(f" delta={DELS[di]}:\n", np.round(q[di] * 25, 3))
