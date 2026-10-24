# Averaged field-row constant for GENERAL single features (torus model):
#   Xi = nu^{1-r} E_U[(Re Phi(U))_+^r] / Q,  Phi(U) = sum_i c_i prod_t (1 + xi_it e^{i U_t}),  Q = sum |c_i|^r.
# (i) single atom; (ii) sup over m-atom systems by local search.  Xi <= 1 for a feature means it is "tame".
import numpy as np, sys
from scipy.optimize import minimize
rng = np.random.default_rng(11)
def grid(n, N):
    g = (np.arange(N)+0.5)*2*np.pi/N
    return np.stack(np.meshgrid(*([g]*n), indexing='ij'), -1).reshape(-1, n)
def Xi(c, xi, U, r, nu):
    Z = np.exp(1j*U)                                       # (M, n)
    Phi = (c[None, :]*np.prod(1 + xi[None, :, :]*Z[:, None, :], axis=2)).sum(1)
    return nu**(1-r)*np.mean(np.maximum(Phi.real, 0)**r)/np.sum(np.abs(c)**r)
for n, N in [(2, 160), (3, 48), (4, 20)]:
    nu = n+1; U = grid(n, N)
    for s in sorted(set([1.25, 1.5, nu/2])):
        r = 1/s
        x1 = Xi(np.array([1.0+0j]), np.ones((1, n), complex), U, r, nu)
        best = (0, None)
        for m in [1, 2, 3, nu]:
            def f(x):
                c = np.exp(x[:m])*np.exp(1j*x[m:2*m]); xi = np.exp(1j*x[2*m:].reshape(m, n))
                return -Xi(c, xi, U, r, nu)
            for k in range(6 if n < 4 else 3):
                x0 = np.concatenate([rng.normal(0, .5, m), rng.uniform(-np.pi, np.pi, m), rng.uniform(-np.pi, np.pi, m*n)])
                res = minimize(f, x0, method='Nelder-Mead', options={'maxiter': 3000 if n < 4 else 1200, 'xatol': 1e-6, 'fatol': 1e-9})
                if -res.fun > best[0]: best = (-res.fun, m)
        print(f"n={n} s={s:.2f}: Xi(single atom)={x1:.4f}  sup Xi found={best[0]:.4f} (m={best[1]})", flush=True)
