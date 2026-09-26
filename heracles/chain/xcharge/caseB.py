# Case B at p = 5: networks whose X-free bound FAILS (every X-blind test value >= gamma_sym P^{3/2}).
# How good can their TRUE score be?  maximise t s.t. all gaps >= t * gs, all X-blind values >= gs, |theta|^2 = 1.
import sys, numpy as np
from scipy.optimize import minimize
p = 5; N = int(sys.argv[1]) if len(sys.argv) > 1 else 9; seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 0; Gam = 0.8838834764831844; delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
gs = 3 ** -0.5 * 2 * (6 * p) ** -1.5 * Gam; x = np.arange(p); S = (x[:, None] + x[None, :]) % p; i2 = pow(2, -1, p)
def gap3(v):
    E1, E2, W = v.reshape(3, N, p); Z = np.einsum('ia,ib,ic->abc', E1, E2, W); corr = Z[x[:, None], x[None, :], S]
    return np.stack([corr - Z[x[:, None], x[None, :], (S - D) % p] for D in range(p)], -1)
def xblind_vals(g):
    Ds = range(1, p); out = [g[:, :, D].mean() for D in Ds]
    for s in range(p):
        out.append(np.mean([g[(s + D * i2) % p, :, D].mean() for D in Ds]))
        out.append(np.mean([g[:, (s + D * i2) % p, D].mean() for D in Ds]))
        out.append(np.mean([np.mean([g[a, (s + D * i2 - a) % p, D] for a in range(p)]) for D in Ds]))
    return np.array(out)
if __name__ == '__main__':
    rng = np.random.default_rng(2); best = []
    for s in range(seeds):
        v = rng.normal(size=3 * N * p); v /= np.linalg.norm(v)
        cons = [dict(type='ineq', fun=lambda z: gap3(z[:-1])[:, :, 1:].ravel() - z[-1] * gs),
                dict(type="ineq", fun=lambda z: xblind_vals(gap3(z[:-1])) - (1 + delta) * gs),
                dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
        g0 = gap3(v)
        r = minimize(lambda z: -z[-1], np.r_[v, g0[:, :, 1:].min() / gs], constraints=cons, method='SLSQP', options=dict(maxiter=1500, ftol=1e-12))
        vv = r.x[:-1] / np.linalg.norm(r.x[:-1]); g = gap3(vv)
        feas = xblind_vals(g).min() / gs
        best.append((g[:, :, 1:].min() / gs, feas, s))
        print(f"seed {s}: true T {g[:, :, 1:].min() / gs:.4f}   min X-blind value / target {feas:.4f} (>= 1 means case B)", flush=True)
    print("MAX true T over case-B-feasible:", max([b for b in best if b[1] >= (1 + delta) * 0.999], default=None))
