# Can the PROVED eta-form cost bound certify case A?  Maximise, over pure profiles (Nper atoms per frequency),
#   Ftilde = min(X-blind values) / (gamma_sym ((6/p) sum_k 3^{1/3} (|G_k| - 0.272 eta_k)_+^{2/3})^{3/2}).
# If sup Ftilde <= 1 (only at the block), case A holds for EVERY network whose diagonal profile is in the searched range.
import sys, numpy as np
from scipy.optimize import minimize
_a = sys.argv; sys.argv = [_a[0], '9']
from caseB import xblind_vals, gs, p
sys.argv = _a
Nper = int(sys.argv[1]); seeds = int(sys.argv[2]); cap = float(sys.argv[3])     # cap: max relative junk allowed
K = 2; M = K * Nper; x = np.arange(p); S = (x[:, None] + x[None, :]) % p
ks = np.repeat(np.arange(1, K + 1), Nper)
C = np.cos(2 * np.pi * np.outer(ks, x) / p); Sn = np.sin(2 * np.pi * np.outer(ks, x) / p)
def build(par):
    a = par.reshape(M, 3, 2); return [a[:, j, 0:1] * C + a[:, j, 1:2] * Sn for j in range(3)]
def gap3(E):
    Z = np.einsum('ia,ib,ic->abc', *E); corr = Z[x[:, None], x[None, :], S]
    return np.stack([corr - Z[x[:, None], x[None, :], (S - D) % p] for D in range(p)], -1)
def prof(E):
    Zh = np.fft.fftn(np.einsum('ia,ib,ic->abc', *E)); out = []
    for k in (1, 2):
        m = (-k) % p; G = abs(Zh[k, k, m]); eta = np.sqrt(abs(Zh[m, k, m]) ** 2 + abs(Zh[k, m, m]) ** 2 + abs(Zh[m, m, m]) ** 2)
        out.append((G, eta))
    return out
def denom(E):
    return gs * ((6 / p) * sum(3 ** (1 / 3) * max(1e-12, G - 0.272 * eta) ** (2 / 3) for G, eta in prof(E))) ** 1.5
rng = np.random.default_rng(4); best = []
for s in range(seeds):
    par = rng.normal(size=M * 6)
    if s % 2 == 0:   # start near a block: phases pi l/3 (Nper >= 3), then perturb
        a = np.zeros((M, 3, 2))
        for i in range(M):
            l = i % Nper; ph = np.pi * l / 3; r0 = 0.7
            a[i, 0] = a[i, 1] = r0 * np.array([np.cos(ph), -np.sin(ph)]); a[i, 2] = r0 * np.array([np.cos(2 * ph), -np.sin(2 * ph)])
        par = a.ravel() + 0.2 * rng.normal(size=M * 6)
    cons = [dict(type='ineq', fun=lambda z: xblind_vals(gap3(build(z[:-1]))) - z[-1] * denom(build(z[:-1]))),
            dict(type='ineq', fun=lambda z: np.array([cap * G - eta for G, eta in prof(build(z[:-1]))])),
            dict(type='eq', fun=lambda z: z[:-1] @ z[:-1] - 1)]
    par /= np.linalg.norm(par)
    E0 = build(par); z0 = np.r_[par, xblind_vals(gap3(E0)).min() / denom(E0)]
    r = minimize(lambda z: -z[-1], z0, constraints=cons, method='SLSQP', options=dict(maxiter=1500, ftol=1e-12))
    E = build(r.x[:-1]); val = xblind_vals(gap3(E)).min() / denom(E)
    rel = [eta / G for G, eta in prof(E)]
    best.append((val, s)); print(f"Nper={Nper} cap={cap} seed {s}: Ftilde {val:.5f}  relative junk {np.round(rel, 3)}", flush=True)
print("MAX Ftilde:", max(best))
