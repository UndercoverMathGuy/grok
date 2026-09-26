# Case B at p = 5, two phases.  Phase 1: push deep into case B (maximise the X-free bound B_XB / gamma_sym at |theta| = 1,
# unrestricted networks).  Phase 2: from that feasible point, maximise the TRUE score subject to staying in case B with
# margin delta:  every X-blind value >= (1 + delta) gs.
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.')
from caseB import gap3, xblind_vals, gs, N, p
delta = float(sys.argv[3]); seeds = int(sys.argv[2])
rng = np.random.default_rng(int(1000 * delta) + 3); out = []
for s in range(seeds):
    v = rng.normal(size=3 * N * p); v /= np.linalg.norm(v)
    c1 = [dict(type='ineq', fun=lambda z: xblind_vals(gap3(z[:-1])) - z[-1] * gs), dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
    r1 = minimize(lambda z: -z[-1], np.r_[v, xblind_vals(gap3(v)).min() / gs], constraints=c1, method='SLSQP', options=dict(maxiter=800, ftol=1e-12))
    v1 = r1.x[:-1] / np.linalg.norm(r1.x[:-1]); bxb = xblind_vals(gap3(v1)).min() / gs
    if bxb < 1 + delta:
        print(f"seed {s}: phase 1 reached B_XB {bxb:.4f} < 1+delta, skip", flush=True); continue
    c2 = [dict(type='ineq', fun=lambda z: gap3(z[:-1])[:, :, 1:].ravel() - z[-1] * gs),
          dict(type='ineq', fun=lambda z: xblind_vals(gap3(z[:-1])) - (1 + delta) * gs),
          dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
    r2 = minimize(lambda z: -z[-1], np.r_[v1, gap3(v1)[:, :, 1:].min() / gs], constraints=c2, method='SLSQP', options=dict(maxiter=1500, ftol=1e-12))
    v2 = r2.x[:-1] / np.linalg.norm(r2.x[:-1]); g = gap3(v2)
    T = g[:, :, 1:].min() / gs; feas = xblind_vals(g).min() / gs
    out.append((T, feas, s)); np.save(f'caseB2_d{delta}_s{s}.npy', v2)
    print(f"seed {s}: phase-1 B_XB {bxb:.4f} | phase 2: true T {T:.4f}, B_XB {feas:.4f} (need >= {1 + delta})", flush=True)
ok = [o for o in out if o[1] >= (1 + delta) - 1e-6]
print("MAX true T in case B (feasible):", max(ok) if ok else None)
