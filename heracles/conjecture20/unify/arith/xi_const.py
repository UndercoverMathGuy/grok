# Pointwise-payment constants.
# Relations case (B=C=0): field Phi(zeta) = conj(A) + conj(D) zeta, zeta uniform on mu_p (or T).
#   IF price a satisfies a >= max(|A|,|D|) - c12 |D|  (P12 + two-atom cap, using tau_rel (1-c12) = 1).
#   Xi_rel(p) = sup_{t=|D|/|A|, theta} avg_{zeta in mu_p} (1 + t cos(theta+arg zeta))_+^r / (max(1,t) - c12 t)^r.
# Class C, n=2: Phi(z1,z2) = sum_i conj(b_i)(1+conj(xi_i) z1)(1+conj(eta_i) z2); price 3^{r-1} sum|b_i|^r.
#   Xi_C(p)  = sup over atom systems of avg_{z in mu_p^2} (Re Phi)_+^r / (3^{r-1} sum |b_i|^r).
import numpy as np, sys
from scipy.optimize import minimize
from math import gamma, sqrt, pi

def c0(r):  # E (1+cos U)^r
    return 2**r*gamma(r+0.5)/(sqrt(pi)*gamma(r+1))

def xi_rel(s, p=None, nt=400):
    r = 1/s; c = 1-(2/3)**(s-1)
    best = (0, None)
    for t in np.linspace(0, 3, nt+1)[1:]:
        if p is None:
            U = np.linspace(0, 2*np.pi, 20001)[:-1]; ths = [0.0]
        else:
            U = 2*np.pi*np.arange(p)/p; ths = np.linspace(0, 2*np.pi/p, 41)
        for th in ths:
            num = np.mean(np.maximum(1 + t*np.cos(U+th), 0)**r)
            val = num/(max(1, t) - c*t)**r
            if val > best[0]: best = (val, (t, th))
    return best

def xi_C(s, grid=48, natoms=(1, 2, 3, 4), restarts=40, seed=0):
    r = 1/s; rng = np.random.default_rng(seed)
    g = 2*np.pi*np.arange(grid)/grid; Z1, Z2 = np.meshgrid(np.exp(1j*g), np.exp(1j*g)); Z1 = Z1.ravel(); Z2 = Z2.ravel()
    def neg(x, N):
        b = x[:N]*np.exp(1j*x[N:2*N]); xi = np.exp(1j*x[2*N:3*N]); eta = np.exp(1j*x[3*N:4*N])
        Phi = (np.conj(b)[:, None]*(1+np.conj(xi)[:, None]*Z1[None])*(1+np.conj(eta)[:, None]*Z2[None])).sum(0)
        num = np.mean(np.maximum(Phi.real, 0)**r)
        den = 3**(r-1)*np.sum(np.abs(x[:N])**r)
        return -num/den
    best = (0, None)
    for N in natoms:
        for k in range(restarts):
            x0 = np.concatenate([rng.uniform(0.2, 1, N), rng.uniform(0, 2*np.pi, 3*N)])
            res = minimize(neg, x0, args=(N,), method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-8, fatol=1e-10))
            if -res.fun > best[0]: best = (-res.fun, (N, res.x))
    return best

if __name__ == '__main__':
    for s in [1.1, 1.25, 1.5]:
        r = 1/s; c = 1-(2/3)**(s-1); tau = (1.5)**(s-1)
        v, arg = xi_rel(s)
        print(f"s={s}: c0(r)={c0(r):.5f}  c0*tau^r={c0(r)*tau**r:.5f}  Xi_rel(T)={v:.5f} at t={arg[0]:.3f}  -> gain bound Xi^s={v**s:.4f}", flush=True)
        for p in [5, 7, 11, 13, 31, 101]:
            vp, ap = xi_rel(s, p)
            print(f"      p={p}: Xi_rel(p)={vp:.5f} (t={ap[0]:.3f})  Xi^s={vp**s:.4f}", flush=True)
    if len(sys.argv) > 1:
        for s in [1.5, 1.25]:
            v, (N, x) = xi_C(s)
            print(f"class C n=2, s={s}: Xi_C(T, grid 48) >= {v:.5f} with {N} atoms; gain bound Xi^s = {v**s:.4f}", flush=True)
