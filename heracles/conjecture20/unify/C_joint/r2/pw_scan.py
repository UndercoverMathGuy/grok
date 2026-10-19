# Scan: PW_C ratio  min_w sum_l (beta_l(1+cos(phi_l+2 pi l w/p)))^r / ((n/nu)^{1-r} sum beta_l^r)
# for (a) Legendre phases, (b) best of random phases (equal beta), (c) optimised phases (maximise the min).
import numpy as np, sys
from scipy.optimize import minimize
def legendre_phases(p):
    x = np.arange(p); qr = set((x[1:]**2) % p)
    chi = np.array([0]+[1 if a in qr else -1 for a in x[1:]]); l = np.arange(1, (p-1)//2+1)
    ch = (chi[None, :]*np.exp(-2j*np.pi*np.outer(l, x)/p)).sum(1)/p
    return np.angle(ch)
def ratio(phi, p, n, s):
    r = 1/s; nu = n+1; F = len(phi); l = np.arange(1, F+1); w = np.arange(p)
    lhs = ((1 + np.cos(phi[None, :] + 2*np.pi*np.outer(w, l)/p))**r).sum(1)
    return lhs.min()/((n/nu)**(1-r)*F)
def opt_phases(p, n, s, seed, tau=0.05):
    r = 1/s; nu = n+1; F = (p-1)//2; l = np.arange(1, F+1); w = np.arange(p); M = 2*np.pi*np.outer(w, l)/p
    den = (n/nu)**(1-r)*F
    def f(phi):
        c = 1 + np.cos(phi[None, :] + M); c = np.maximum(c, 1e-12)
        val = (c**r).sum(1)/den; mn = val.min(); e = np.exp(-(val-mn)/tau); q = e/e.sum()
        sm = mn - tau*np.log(e.sum())
        g = (q[:, None]*(r*c**(r-1)*(-np.sin(phi[None, :] + M)))).sum(0)/den
        return -sm, -g
    rng = np.random.default_rng(seed); phi = rng.uniform(0, 2*np.pi, F)
    for t in [0.05, 0.02, 0.01, 0.005]:
        phi = minimize(f, phi, jac=True, method='L-BFGS-B', args=(), options={'maxiter': 500}).x if False else \
              minimize(lambda x: f(x) if True else 0, phi, jac=True, method='L-BFGS-B', options={'maxiter': 500}).x
    return ratio(phi, p, n, s)
if __name__ == '__main__':
    n, s = int(sys.argv[1]), float(sys.argv[2])
    for p in [int(a) for a in sys.argv[3:]]:
        leg = ratio(legendre_phases(p), p, n, s)
        rng = np.random.default_rng(p); rnd = max(ratio(rng.uniform(0, 2*np.pi, (p-1)//2), p, n, s) for _ in range(20))
        op = max(opt_phases(p, n, s, sd) for sd in range(2))
        print(f"n={n} s={s} p={p} (mod4={p%4}): Legendre {leg:.4f}  best-of-20 random {rnd:.4f}  optimised {op:.4f}", flush=True)
