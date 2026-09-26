# Exact max-min search (SLSQP): maximise t s.t. every gap >= t * RHS(theta), |theta| = 1.   RHS: T or CA (smoothed).
# usage: global_slsqp.py p N mode seeds kind    kind in {lmax, blockmix, random}
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '..')
from common import programme, block_net
from global_eval import evaluate
p, N, mode, seeds, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
h = (p - 1) // 2; Gam, ystar = programme(p, 1.5); gs = 3 ** -0.5 * 2 * (6 * p) ** -1.5 * Gam
x = np.arange(p); S = (x[:, None] + x[None, :]) % p
def gaps(v):
    E1, E2, W = v.reshape(3, N, p); Z = np.einsum('ia,ib,ic->abc', E1, E2, W)
    corr = Z[x[:, None], x[None, :], S]
    return np.concatenate([(corr - Z[x[:, None], x[None, :], (S - D) % p]).ravel() for D in range(1, p)])
def rhs(v):
    if mode == 'T': return gs * (v @ v) ** 1.5
    F = np.fft.fft(v.reshape(3, N, p), axis=2)
    c2 = np.abs(F[0, :, 1:h + 1] * F[1, :, 1:h + 1] * F[2, :, 1:h + 1]) ** 2
    return 2 / p ** 3 * Gam * (((c2 + 1e-9) ** (1 / 3)).sum() / 3 ** (1 / 3)) ** 1.5
rng = np.random.default_rng(7)
def seed_vec(s):
    if kind == 'lmax':
        f = {5: 3, 7: 9}[p]
        v = np.load(f'../../claude0923/endpoint_p{p}_s{f}.npy').reshape(3, -1, p)
        v = np.concatenate([v, np.zeros((3, max(0, N - v.shape[1]), p))], 1)[:, :N]
        return (v + (0.0 if s == 0 else 0.01 * s) * rng.normal(size=v.shape)).ravel()
    if kind == 'blockmix':
        v = np.zeros((3, N, p))
        for i, (e1, e2, w) in enumerate(block_net(p, ystar)[:N]): v[:, i] = [e1, e2, w]
        eps = [0.03, 0.08, 0.15, 0.3][s % 4]
        pert = eps * rng.normal(size=v.shape) * np.abs(v).max()
        if s % 2: pert[:, :, :] += eps * np.abs(v).max() * rng.normal(size=(3, N, 1))   # add DC components
        return (v + pert).ravel()
    return rng.normal(size=3 * N * p)
res = []
for s in range(seeds):
    v = seed_vec(s); v /= np.linalg.norm(v)
    z0 = np.r_[v, gaps(v).min() / rhs(v)]
    cons = [dict(type='ineq', fun=lambda z: gaps(z[:-1]) - z[-1] * rhs(z[:-1])), dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
    r = minimize(lambda z: -z[-1], z0, constraints=cons, method='SLSQP', options=dict(maxiter=1500, ftol=1e-13))
    vn = r.x[:-1] / np.linalg.norm(r.x[:-1]); V = vn.reshape(3, N, p)
    ev = evaluate(p, [(V[0, i], V[1, i], V[2, i]) for i in range(N)])
    res.append((ev[mode], s)); np.save(f'sq_{p}_{mode}_{kind}_s{s}.npy', vn)
    print(f"p={p} {mode} {kind} seed {s}: T {ev['T']:.6f} CA {ev['CA']:.6f} CAb {ev['CAb']:.6f} CAoff {ev['CAoff']:.6f}", flush=True)
print("MAX", max(res))
