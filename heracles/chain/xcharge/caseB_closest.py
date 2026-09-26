# How close to junk-free can a case-B network be?  minimise s = max_k eta_k/|G_k| (relative junk, worst frequency)
# subject to F (X-free bound / target) >= 1 + 1e-4, |theta| = 1, unrestricted width-9 networks.
# Starts: case-B endpoints from caseB3 (already feasible).
import sys, glob, numpy as np
from scipy.optimize import minimize
_a = sys.argv; sys.argv = [_a[0], '9']
from caseB import gap3, xblind_vals, gs, p
sys.argv = _a
N = 9
def rel(v):
    Zh = np.fft.fftn(np.einsum('ia,ib,ic->abc', *v.reshape(3, N, p)))
    out = []
    for k in (1, 2):
        m = (-k) % p; G = abs(Zh[k, k, m]); eta = np.sqrt(abs(Zh[m, k, m]) ** 2 + abs(Zh[k, m, m]) ** 2 + abs(Zh[m, m, m]) ** 2)
        out.append(eta / max(G, 1e-9))
    return np.array(out)
best = []
for f in sorted(glob.glob('caseB3_d*_s*.npy')):
    v = np.load(f); v /= np.linalg.norm(v)
    z0 = np.r_[v, rel(v).max()]
    cons = [dict(type='ineq', fun=lambda z: z[-1] - rel(z[:-1])),
            dict(type='ineq', fun=lambda z: xblind_vals(gap3(z[:-1])) - (1 + 1e-4) * gs),
            dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
    r = minimize(lambda z: z[-1], z0, constraints=cons, method='SLSQP', options=dict(maxiter=800, ftol=1e-12))
    vv = r.x[:-1] / np.linalg.norm(r.x[:-1]); feas = xblind_vals(gap3(vv)).min() / gs
    best.append((rel(vv).max(), feas, f)); print(f"{f}: max relative junk {rel(vv).max():.4f} (per k {np.round(rel(vv), 3)}), F {feas:.4f}", flush=True)
ok = [b for b in best if b[1] >= 1 + 5e-5]
print("closest case-B (min over feasible of max_k eta_k/|G_k|):", min(ok) if ok else None)
