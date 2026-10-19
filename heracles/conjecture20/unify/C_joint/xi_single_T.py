# Averaged field-row constant for aligned single-T features (torus model):
#   Xi_n(s) = sup_{d in [0,1]} E_U (1 + d cos U)^r / F_n(d)^r ,  F_n(d) = nu^{1-s} Phi_n(d)^s (sharp aligned single-T payment).
# Xi_n(s) <= 1 means: every aligned single-T network has gamma <= gamma_IF (up to the p-point aliasing of E_U).
import numpy as np
from scipy.special import gamma as G
U = np.linspace(0, 2*np.pi, 20001)[:-1]
def Phi(n, r, d):
    nu = n+1
    return 0.5*((1+d)**r*(nu+(n-1)*d)**(1-r) + (1-d)**r*(nu-(n-1)*d)**(1-r))
ds = np.linspace(0, 1, 401)
print("n, s(b), Xi_n(s), argmax d, [pure n-atom value (nu/n)^{1-r} c0(r)]")
for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]:
    nu = n+1
    for s in sorted(set([1.1, 1.25, 1.5, min(2.0, nu/2), min(3.0, nu/2), nu/2])):
        if s > nu/2: continue
        r = 1/s
        E = np.array([np.mean(np.maximum(1+d*np.cos(U), 0)**r) for d in ds])
        F = nu**(1-s)*Phi(n, r, ds)**s
        X = E/F**r
        c0 = 2**r*G(r+0.5)/(np.sqrt(np.pi)*G(r+1)); pure = (nu/n)**(1-r)*c0
        print(f"  n={n:2d} s={s:5.2f} (b={nu/s:5.2f}): Xi={X.max():.5f} at d={ds[X.argmax()]:.3f}   pure={pure:.5f}")
