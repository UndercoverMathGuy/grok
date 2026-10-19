# Claim R caveat, sharp version: decoupled networks = two relabelled copies (features 1, 2 at p = 5) of ONE IF single
# feature theta_1 taken from the non-aligned IF manifold (3 atoms, nodes alpha, n = 2).  For such unions (exactly IF):
#   gamma_1/B_p = e * M(yhat),  e = |A|/L,  H_{2/3}(m) = H_{2/3}(yhat) + H_{2/3}(rho/L).
# Maximise over (alpha, t) subject to H <= H(theta_v) = 1.537697 and compare with KM2 (0.818485) and theta_v (0.869933).
# The best point is then rebuilt as a genuine network and evaluated directly.
import numpy as np, math
from scipy.optimize import minimize
from vlib import *
from v3_ifmanifold import block_specs
p, n, b = 5, 2, 2.0; r = 2 / 3; B = Bp(p, n); cap = 1.537697; K = kappa(p)[:2]
def zatoms(al):
    V = np.array([[np.exp(1j * j * a) for a in al] for j in range(1, n + 1)])
    _, _, vh = np.linalg.svd(V); z = np.conj(vh[-1]); return z
def renyi_(q): q = q / q.sum(); return 3 * math.log((q ** r).sum())
def val(x):
    al = np.array([0.0, x[0], x[1]]); t = 1 / (1 + math.exp(-x[2]))
    z = zatoms(al); rho = np.abs(z); e = abs(z.sum()) / rho.sum()
    H = renyi_(np.array([1 - t, t])) + renyi_(rho)
    M = (K @ np.array([1 - t, t])).min()
    return e * M, H, e, renyi_(rho), t
best = (0, None)
rng = np.random.default_rng(0)
for trial in range(400):
    x0 = np.array([2 * np.pi / 3, 4 * np.pi / 3, -2]) + rng.normal(size=3) * [0.6, 0.6, 1.0]
    f = lambda x: -(val(x)[0] - 200 * max(0, val(x)[1] - cap) ** 1)
    res = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
    v = val(res.x)
    if v[1] <= cap + 1e-9 and v[0] > best[0]: best = (v[0], res.x, v)
g, x, v = best
print('IF-manifold copies: best e*M = %.6f  (e = %.4f, H(theta_1) = %.4f vs log3 = %.4f, t = %.4f, H total = %.6f)'
      % (g, v[2], v[3], math.log(3), v[4], v[1]))
# rebuild as a genuine network and evaluate exactly
al = np.array([0.0, x[0], x[1]]); t = v[4]
specs = []
for l, wt in [(1, 1 - t), (2, t)]:
    sp, z = block_specs(l, n, al, wt); specs += sp
net = classC(p, n, specs, b); a = net.masses(); m = a / a.sum()
print('rebuilt network: h>0: %s, gamma_1/B = %.6f, H_2/3(m) = %.6f ; KM2 best 0.818485, theta_v 0.869933'
      % (net.gaps().min() > 0, net.gaps().min() / a.sum() / B, renyi(m, r)))
