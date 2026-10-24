# Referee probe of largen R7c (CONDITIONAL).  The conclusion needs UPPER bounds k*(ceil(2s-1), s) < k_p, while only
# numerical LOWER bounds (local searches) exist.  Here: (1) independent searches for k*(7,4) (b=2, the value R7c uses
# against k_5 = 0.94574), slot-symmetric and general atoms; (2) continuation searches for k*(n, nu/2) at n = 25..31,
# i.e. s = 13..16, against k_7 = 1.81987 (R7c: "at p=7 the same holds for s up to about 15").  Any value found above
# the threshold is re-evaluated in 200-bit interval arithmetic (v_R5.kstar_iv, Gram form).
import sys, math, os
import numpy as np
from math import comb
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(99)
D = '/home/user/grok/heracles/conjecture20/unify/largen/'
K5 = math.sqrt(1 / math.tan(math.pi / 5) ** 2 - 1); K7 = math.sqrt(1 / math.tan(math.pi / 7) ** 2 - 1)

def ratio_sym(v, m, n, s):
    rho, al, ph = v[:m], v[m:2 * m], v[2 * m:3 * m]
    beta = np.exp(rho + 1j * al); xi = np.exp(1j * ph)
    A = beta.sum(); Q = np.exp(rho / s).sum(); nu = n + 1
    K = np.arange(1, n + 1)
    mk = (beta[None, :] * xi[None, :] ** K[:, None]).sum(1)
    E = np.sum(np.array([comb(n, k) for k in K]) * np.abs(mk) ** 2)
    return (A.real - nu ** (1 - s) * Q ** s) / math.sqrt(E / 2 + 1e-300)

def ratio_gen(v, m, n, s):
    rho, al = v[:m], v[m:2 * m]; ph = v[2 * m:].reshape(m, n)
    beta = np.exp(rho + 1j * al); xi = np.exp(1j * ph)
    A = beta.sum(); Q = np.exp(rho / s).sum(); nu = n + 1
    G = np.prod(1 + xi[:, None, :] * np.conj(xi[None, :, :]), axis=2) - 1
    E = np.real(beta @ G @ np.conj(beta))
    return (A.real - nu ** (1 - s) * Q ** s) / math.sqrt(max(E, 1e-300) / 2)

def search(fun, dim_fn, m, n, s, starts, init=None):
    best, bx = -np.inf, None
    for st in range(starts):
        if init is not None and st == 0:
            v0 = init
        else:
            v0 = dim_fn(m, n)
        r = minimize(lambda v: -fun(v, m, n, s), v0, method='L-BFGS-B', options=dict(maxiter=4000))
        if -r.fun > best: best, bx = -r.fun, r.x
    return best, bx

print(f"k_5 = {K5:.6f}, k_7 = {K7:.6f}")
print("(1) k*(7,4): slot-symmetric atoms")
n, s = 7, 4.0
glob_best = 0
for m in range(3, 13):
    init = lambda m, n: np.concatenate([0.5 * rng.standard_normal(m), rng.uniform(0, 2 * np.pi, m), rng.uniform(0, 2 * np.pi, m)])
    b, x = search(ratio_sym, init, m, n, s, 30)
    glob_best = max(glob_best, b)
    print(f"   m={m}: best {b:.5f}", flush=True)
print("    general atoms (xi_i in T^7)")
for m in range(3, 9):
    init = lambda m, n: np.concatenate([0.5 * rng.standard_normal(m), rng.uniform(0, 2 * np.pi, m), rng.uniform(0, 2 * np.pi, m * n)])
    b, x = search(ratio_gen, init, m, n, s, 16)
    glob_best = max(glob_best, b)
    print(f"   m={m}: best {b:.5f}", flush=True)
print(f"  overall best k*(7,4) lower bound found: {glob_best:.5f}  (R7c uses 0.928 < k_5 = {K5:.5f} as if it were an upper bound)")

print("(2) k*(n, nu/2) for n = 25..31 by continuation from the stored n=24 and n=32 atom systems")
def load(f):
    d = np.load(f); m = int(d[0]); z = d[1:]; return m, z
res = {}
m24, z24 = load(D + 'arcpolish_n24.npy'); m32, z32 = load(D + 'arcpolish_n32.npy')
z = z24.copy(); m = m24
for nn in range(24, 32):
    ss = (nn + 1) / 2
    r = minimize(lambda v: -ratio_sym(v, m, nn, ss), z, method='L-BFGS-B', options=dict(maxiter=20000))
    z = r.x; res[nn] = max(res.get(nn, -np.inf), -r.fun)
    print(f"   up from 24:  n={nn} (s={ss}): {-r.fun:.5f}", flush=True)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'up_n{nn}.npy'), np.concatenate([[m], z]))
z = z32.copy(); m = m32
for nn in range(32, 23, -1):
    ss = (nn + 1) / 2
    r = minimize(lambda v: -ratio_sym(v, m, nn, ss), z, method='L-BFGS-B', options=dict(maxiter=20000))
    z = r.x
    if -r.fun > res.get(nn, -np.inf):
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'down_n{nn}.npy'), np.concatenate([[m], z]))
    res[nn] = max(res.get(nn, -np.inf), -r.fun)
    print(f"   down from 32: n={nn} (s={ss}): {-r.fun:.5f}  (> k_7: {-r.fun > K7})", flush=True)
