# Pure form of the plane Bhatia-Davis conjecture (PB2):
#   f, g, h : Z_p -> R mean zero,  Z_c(u,v) = f(u) + g(v) + h(u+v-c).
#   claim:  ( -min Z_0 ) * ( max Z_d ) >= Var f + Var g + Var h   for every d in Z_p.
# Adversarial minimisation of the ratio over function values (smoothed min/max, L-BFGS, restarts).
import numpy as np, sys
from scipy.optimize import minimize
p = int(sys.argv[1]); reps = int(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
U, V = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); U = U.ravel(); V = V.ravel()
def split(z):
    f = z[:p] - z[:p].mean(); g = z[p:2*p] - z[p:2*p].mean(); h = z[2*p:] - z[2*p:].mean(); return f, g, h
def ratio(z, d, tau):
    f, g, h = split(z)
    Z0 = f[U] + g[V] + h[(U+V) % p]; Zd = f[U] + g[V] + h[(U+V-d) % p]
    var = (f**2).mean() + (g**2).mean() + (h**2).mean()
    if tau is None: return (-Z0.min())*Zd.max()/var
    a = -( -np.log(np.mean(np.exp(-tau*(Z0 - Z0.min()))))/tau + Z0.min())   # smooth -min (lower bound of -min)
    b = np.log(np.mean(np.exp(tau*(Zd - Zd.max()))))/tau + Zd.max()            # smooth max (lower bound of max)
    return a*b/var
best = (np.inf, None, None)
for r in range(reps):
    d = int(rng.integers(1, p))
    z = rng.normal(size=3*p)
    if r % 4 == 1: z[:2*p] *= 0.05          # near line case
    if r % 4 == 2: z *= np.repeat(rng.random(3) < 0.67, p) + 0.02
    if r % 4 == 3:                          # spikes
        z = np.zeros(3*p); z[rng.integers(p)] = 5*rng.normal(); z[p+rng.integers(p)] = 5*rng.normal(); z[2*p+rng.integers(p)] = 5*rng.normal(); z += 0.1*rng.normal(size=3*p)
    for tau in (2.0, 8.0, 30.0, 120.0, 500.0):
        sc = np.abs(z).max(); z = z/sc
        z = minimize(lambda w: ratio(w, d, tau), z, method='L-BFGS-B', options={'maxiter': 500}).x
    val = ratio(z, d, None)
    if val < best[0]: best = (val, z, d)
print(f"p={p}: min ratio found = {best[0]:.5f}  (d={best[2]})")
f, g, h = split(best[1]); np.set_printoptions(precision=3, suppress=True)
s = np.abs(np.concatenate([f,g,h])).max()
print("f", f/s); print("g", g/s); print("h", h/s)
print("vars", (f**2).mean()/s**2, (g**2).mean()/s**2, (h**2).mean()/s**2)
