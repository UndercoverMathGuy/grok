# Class C, n=2: worst-case ratio  R = Pay^2 / Cert  over interference moduli x_{l,T} = |beta_{l,T}|^2 >= 0.
#   Cert (lower bound for mbar_d^2, proved):  max( max_T V_T ,  max_T (E - E2_T) )
#       V_T = sum_l x_{l,T} k_l,  E = sum_T V_T,  E2_T = sum_l x_{l,T} k_l^2,   k_l = kappa_l(d)
#   Pay (upper bound for sum_{l cheap} bonus_l k_l, proved):  sum_{l in L'} k_l * min( R_l/(2 sqrt 3), c1 * M_l )
#       R_l = sqrt(sum_T x_{l,T}),  M_l = max_T sqrt(x_{l,T}),  c1 = 1 - 3^{1-s} (s=3/2: 0.4226; Theorem A alone if mode='A')
# Non-cheap interference y_{l,T} >= 0 enters Cert only.  Class C at n=2 is closed at prime p (all b in [2,3]) if max R <= 1.
# The multiset {kappa_l(d)} does not depend on d (relabelling), so d=1 suffices.
import numpy as np, sys
from scipy.optimize import minimize
p = int(sys.argv[1]); mode = sys.argv[2]; reps = int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(seed)
F = np.arange(1, (p-1)//2 + 1); K = len(F)
k = 1 - np.cos(2*np.pi*F/p)
c1 = 1 - 3**(1-1.5); cA = 1/(2*np.sqrt(3))
def unpack(z):
    x = z[:3*K].reshape(3, K)**2; y = z[3*K:].reshape(3, K)**2; return x, y
def cert(x, y, tau):
    w = x + y
    V = (w*k).sum(1); E = V.sum(); E2 = (w*k**2).sum(1)
    c = np.concatenate([V, E - E2])
    if tau is None: return c.max()
    return c.max() + np.log(np.exp(tau*(c - c.max())).sum())/tau      # smooth max (upper bound on max)
def pay(x, tau):
    R = np.sqrt(x.sum(0) + 1e-300)
    if mode == 'A': return (k*cA*R).sum()
    if tau is None: M = np.sqrt(x.max(0))
    else:
        sx = np.sqrt(x + 1e-300); M = sx.max(0) - np.log(np.exp(-tau*(sx.max(0) - sx)).sum(0))/tau*0   # exact max for value
        M = sx.max(0)
    return (k*np.minimum(cA*R, c1*M)).sum()
def negratio(z, tau):
    x, y = unpack(z)
    C = cert(x, y, tau)
    if C <= 0: return 0.0
    return -pay(x, tau)**2/C
best = (0, None)
for r in range(reps):
    z0 = np.abs(rng.normal(size=6*K))
    if r % 3 == 0: z0[3*K:] = 0
    if r % 3 == 1: z0[:3*K] *= np.repeat(rng.random(3) < 0.7, K)
    z = z0
    for tau in (5, 30, 200, 2000):
        z = minimize(negratio, z, args=(tau,), method='Nelder-Mead' if K*6 <= 12 else 'Powell', options={'maxiter': 20000, 'xtol': 1e-10, 'ftol': 1e-12}).x
    val = -negratio(z, None)
    if val > best[0]: best = (val, z)
x, y = unpack(best[1])
print(f"p={p} mode={mode}: worst R = Pay^2/Cert = {best[0]:.4f}  -> {'CLOSED' if best[0] <= 1 else 'not closed'}  "
      f"(sum_F kappa = {k.sum():.2f})")
np.set_printoptions(precision=3, suppress=True)
print("  kappa:", k); print("  x (cheap) by T rows:\n", x/ max(x.max(),1e-300)); print("  y (noncheap):\n", y/max(x.max(),1e-300))
