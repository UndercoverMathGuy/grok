# Referee: R5 label-row principle on the p=3 counterexamples (any n, DC present).  At p=3, F={1}, Gamma_s(3)=kappa_1=3/2,
# so R5 says h <= (3/2) Re Phi_1(x) for EVERY x, with Phi_1(x) = (2/3) rho^{y(x)} sum_c f(x)[c] rho^{-c}.  Exact via counts.
import sys, json, glob, cmath
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from fractions import Fraction as Fr
import exact_certificate as ec
rho = cmath.exp(2j*cmath.pi/3)
for n, net in ec.NETWORKS.items():
    U = [[float(Fr(a)) for a in row] for row in net["u"]]; W = [[float(Fr(a)) for a in row] for row in net["w"]]
    m = len(U); h = None; minrow = None; worst_identity = 0
    for N0 in range(n+1):
        for N1 in range(n+1-N0):
            N2 = n-N0-N1; y = (N1+2*N2) % 3
            z = [U[i][0]**N0*U[i][1]**N1*U[i][2]**N2 for i in range(m)]
            f = [sum(W[i][c]*z[i] for i in range(m)) for c in range(3)]
            Phi = (2/3)*rho**y*sum(f[c]*rho**(-c) for c in range(3))
            for d in (1, 2):
                g = f[y]-f[(y-d) % 3]
                worst_identity = max(worst_identity, abs(g-(Phi*(1-rho**(-d))).real))
                h = g if h is None else min(h, g)
            row = 1.5*Phi.real
            minrow = row if minrow is None else min(minrow, row)
    print(f"p=3 n={n}: h={h:.6e}  min_x (3/2)Re Phi_1(x) = {minrow:.6e}  (R5 needs h <= this): {h <= minrow+1e-12};"
          f"  field identity err {worst_identity:.1e}")
