# SLSQP search at p = 5 for violations of
#  (CA)  margin <= (2/p^3) Gamma (sum_k Y_k^{2/3})^{3/2},  Y_k^{2/3} = cost_k / 3^{1/3},  cost_k = sum_i |c_ik|^{2/3}
#        i.e. the diagonal-budget programme ceiling (drops DC / imbalance slack, so it is STRONGER than (T));
#  (T)   margin <= gamma_sym P^{3/2}  (control).
# maximise t s.t. every gap >= t * RHS(theta), |theta| = 1.
import sys, numpy as np
from scipy.optimize import minimize
p = 5; N = int(sys.argv[1]); seeds = int(sys.argv[2]); mode = sys.argv[3]
Gam = 0.8838834764831844; gs = 3 ** -0.5 * 2 * (6 * p) ** -1.5 * Gam
x = np.arange(p); S = (x[:, None] + x[None, :]) % p
def unpack(v): return v.reshape(3, N, p)
def gaps(v):
    E1, E2, W = unpack(v); Z = np.einsum('ia,ib,ic->abc', E1, E2, W)
    corr = Z[x[:, None], x[None, :], S]
    return np.concatenate([(corr - Z[x[:, None], x[None, :], (S - D) % p]).ravel() for D in range(1, p)])
def rhs(v):
    if mode == 'T': return gs * (v @ v) ** 1.5
    E1, E2, W = np.fft.fft(unpack(v), axis=2)
    cost = sum(((np.abs(E1[:, k] * E2[:, k] * W[:, k]) ** 2 + 1e-14) ** (1 / 3)).sum() for k in (1, 2))
    return 2 / p ** 3 * Gam * (cost / 3 ** (1 / 3)) ** 1.5
def ratio(v): return gaps(v).min() / rhs(v)
rng = np.random.default_rng(0); best = []
v0 = np.load('../../claude0923/endpoint_p5_s3.npy').reshape(3, 9, p)
for s in range(seeds):
    if s < 4:
        v = np.concatenate([v0, 0.02 * rng.normal(size=(3, max(0, N - 9), p))], 1)[:, :N].ravel() + 0.02 * s * rng.normal(size=3 * N * p)
    else:
        v = rng.normal(size=3 * N * p)
    v /= np.linalg.norm(v)
    z0 = np.r_[v, ratio(v)]
    cons = [dict(type='ineq', fun=lambda z: gaps(z[:-1]) - z[-1] * rhs(z[:-1])), dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
    r = minimize(lambda z: -z[-1], z0, constraints=cons, method='SLSQP', options=dict(maxiter=600, ftol=1e-12))
    val = ratio(r.x[:-1] / np.linalg.norm(r.x[:-1]))
    best.append((val, s)); print(f"seed {s}: ratio {val:.6f}", flush=True)
    np.save(f'ca_{mode}_s{s}.npy', r.x[:-1])
print("MAX", max(best))
