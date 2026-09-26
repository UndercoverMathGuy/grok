# Adversarial search for the true graded curve: R(a) = min over populations of cost / (|G| - a eta)_+^{2/3}.
# Conjecture (lone atom): R(a) = kappa(a) = min(3^{1/3}, (1 - sqrt3 a)^{-2/3}).
import sys, json, numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
def unpack(par, N):
    x = np.exp(par[:N]); c = x * np.exp(1j * par[N:2 * N])
    return x, c, np.exp(1j * par[2 * N:3 * N]), np.exp(1j * par[3 * N:4 * N])
def ratio(par, N, a):
    x, c, v1, v2 = unpack(par, N)
    G = c.sum(); J1 = (c * v1).sum(); J2 = (c * v2).sum(); J12 = (c * v1 * v2).sum()
    eta = np.sqrt(abs(J1) ** 2 + abs(J2) ** 2 + abs(J12) ** 2)
    d = abs(G) - a * eta
    if d <= 1e-12: return 50.0
    return (x ** (2 / 3)).sum() / d ** (2 / 3)
def seeds(N):
    out = []
    for _ in range(40):
        spread = rng.choice([0.05, 0.3, 1.0, np.pi])
        out.append(np.r_[rng.normal(0, 0.8, N), rng.normal(0, spread, N), rng.uniform(0, 2 * np.pi, 2 * N)])
    if N >= 3:   # near junk-free block: phases pi l/3 in both slots (v = e^{2 i pi l/3}), amplitudes equal
        for _ in range(20):
            ph = 2 * np.pi * np.arange(N) / 3
            out.append(np.r_[rng.normal(0, 0.3, N), rng.normal(0, 0.3, N), ph + rng.normal(0, 0.3, N), ph + rng.normal(0, 0.3, N)])
    return out
def R(a, Ns=(7, 8)):
    best = (np.inf, None)
    for N in Ns:
        for s in seeds(N):
            r = minimize(ratio, s, args=(N, a), method='Nelder-Mead', options=dict(maxiter=8000, xatol=1e-10, fatol=1e-12))
            r = minimize(ratio, r.x, args=(N, a), method='Powell', options=dict(maxiter=8000, xtol=1e-10, ftol=1e-12))
            if r.fun < best[0]: best = (r.fun, (N, r.x.tolist()))
    return best
for a in [float(t) for t in sys.argv[2].split(',')]:
    val, arg = R(a)
    kap = min(3 ** (1 / 3), (1 - np.sqrt(3) * a) ** (-2 / 3))
    print(json.dumps(dict(a=a, R=round(val, 7), kappa=round(kap, 7), R_over_kappa=round(val / kap, 7), N=arg[1] and arg[0], par=arg[1])), flush=True)
