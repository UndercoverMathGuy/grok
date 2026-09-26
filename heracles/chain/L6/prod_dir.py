# Depth restricted to lone-atom junk directions: z_k = c_k (v1, v2, v1 v2) (all three patterns, equal size, product phases),
# the only junk shape that earns the full sqrt3 discount. Flip-enlarged support function, exact over all rows; search = UPPER est.
import sys, json, numpy as np
from scipy.optimize import minimize
p = int(sys.argv[1]); T = [int(x) for x in sys.argv[2].split(',')]; K = len(T); rng = np.random.default_rng(6)
x = np.arange(p); ze = np.exp(-2j * np.pi * 2 * np.outer(T, x) / p); ab = (x[:, None] + x[None, :]) % p
def H(zc):
    best = -np.inf
    for s1, s2 in ((1, 1), (-1, 1), (1, -1)):
        F = s1 * np.real(zc[0] @ ze)[:, None] + s2 * np.real(zc[1] @ ze)[None, :] + s1 * s2 * np.real(zc[2] @ ze)[ab]
        best = max(best, F.max())
    return best
def obj(v):
    c = np.exp(v[:K]); a1 = v[K:2 * K]; a2 = v[2 * K:3 * K]
    zc = -np.vstack([c * np.exp(1j * a1), c * np.exp(1j * a2), c * np.exp(1j * (a1 + a2))])   # exposure = H(-J), J = lone-atom junk
    return H(zc) / (np.sqrt(3) * c.sum())
best = 9
for s in range(int(sys.argv[3]) if len(sys.argv) > 3 else 12):
    v0 = np.r_[rng.normal(0, 0.3, K), rng.uniform(0, 2 * np.pi, 2 * K)]
    if s == 0: v0 = np.r_[np.zeros(K), np.zeros(2 * K)]
    if s == 1: v0 = np.r_[np.zeros(K), np.pi * np.ones(2 * K)]
    r = minimize(obj, v0, method='Powell', options=dict(maxfev=6000, xtol=1e-7, ftol=1e-9))
    best = min(best, r.fun)
print(json.dumps(dict(p=p, K=K, lone_atom_direction_depth_upper=round(best, 4))), flush=True)
