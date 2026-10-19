# Referee check of R6: adversarial search for atoms violating |A| - th Q^s <= ((1-th)/sqrt3) R.
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(5)
def parts(z, N):
    rho = np.exp(z[:N]); ph = z[N:2*N]; a = z[2*N:3*N]; b = z[3*N:4*N]
    beta = rho*np.exp(1j*ph); xi = np.exp(1j*a); eta = np.exp(1j*b)
    return beta, xi, eta
def viol(z, N, s):
    beta, xi, eta = parts(z, N); th = 3**(1 - s); c = (1 - th)/np.sqrt(3)
    A = beta.sum(); B = (beta*xi).sum(); C = (beta*eta).sum(); D = (beta*xi*eta).sum()
    L = np.abs(beta).sum(); Q = (np.abs(beta)**(1/s)).sum()
    R = np.sqrt(abs(B)**2 + abs(C)**2 + abs(D)**2)
    return (abs(A) - th*Q**s - c*R)/L
for s in [1.5, 1.35, 1.2, 1.08]:
    best = -np.inf; bestz = None
    for t in range(400):
        N = int(rng.integers(1, 7)); kind = t % 4
        z = np.concatenate([rng.normal(size=N)*0.7, rng.uniform(0, 2*np.pi, 3*N)])
        if kind == 1:   # near IF block
            N = 3; w = np.exp(2j*np.pi*np.arange(3)/3)
            z = np.concatenate([0.05*rng.normal(size=3), 0.05*rng.normal(size=3), np.angle(w) + 0.1*rng.normal(size=3), np.angle(w**2) + 0.1*rng.normal(size=3)])
        if kind == 2:   # near single atom + small atoms
            z[:N] = np.r_[0, -3 + rng.normal(size=N - 1)]
        if kind == 3:   # two aligned atoms
            N = 2; z = np.concatenate([0.1*rng.normal(size=2), 0.1*rng.normal(size=2), rng.uniform(0, 2*np.pi, 4)])
        res = minimize(lambda w: -viol(w, N, s), z, method='Nelder-Mead', options=dict(maxiter=4000*N, xatol=1e-10, fatol=1e-13))
        res = minimize(lambda w: -viol(w, N, s), res.x, method='BFGS')
        v = -res.fun
        if v > best: best, bestz = v, (N, res.x)
    N, z = bestz; beta, xi, eta = parts(z, N)
    print(f"s={s}: max over 400 restarts of (|A| - th Q^s - c R)/L = {best:.3e}  (<= 0 required; 0 at a single atom); argmax N={N}, |beta|/L={np.round(np.abs(beta)/np.abs(beta).sum(),4)}")
