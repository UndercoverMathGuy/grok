# f.3 (faster): for each population, A1*(pop; A2) = smallest radius A1 at which
#   sum|c|^{2/3} >= 3^{1/3} (min_{|t1|=A1,|t2|=A2} Re h)_+^{2/3}
# holds (the inequality only gets easier as A1 grows). Threshold(A2) = sup over populations of A1*.
import sys, json, numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0)
E = np.exp(1j * np.linspace(0, 2 * np.pi, 1025)[:-1])

def unpack(par, N):
    x = np.exp(par[:N]); c = x * np.exp(1j * par[N:2 * N])
    return x, c, np.exp(1j * par[2 * N:3 * N]), np.exp(1j * par[3 * N:4 * N])

def a1star(par, N, A2):
    x, c, v1, v2 = unpack(par, N)
    G = c.sum(); J1 = (c * v1).sum(); J2 = (c * v2).sum(); J12 = (c * v1 * v2).sum()
    base = np.real(G + J2 * A2 * E); slope = np.abs(J1 + J12 * A2 * E)
    cost = (x ** (2 / 3)).sum()
    f = lambda A1: cost - 3 ** (1 / 3) * max(0.0, (base - A1 * slope).min()) ** (2 / 3)
    if f(0.0) >= 0: return 0.0
    lo, hi = 0.0, 2.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if f(mid) >= 0: hi = mid
        else: lo = mid
    return hi

def threshold(A2, restarts):
    best = (0.0, None)
    for N in (1, 2, 3, 4):
        for _ in range(restarts):
            spread = rng.choice([0.2, 0.6, np.pi])   # mostly phase-aligned atoms: those are the ones that save cost
            par = np.r_[rng.normal(0, 0.7, N), rng.normal(0, spread, N), rng.uniform(0, 2 * np.pi, 2 * N)]
            r = minimize(lambda q: -a1star(q, N, A2), par, method='Nelder-Mead',
                         options=dict(maxiter=3000, xatol=1e-8, fatol=1e-10))
            if -r.fun > best[0]:
                best = (-r.fun, (N, r.x.tolist()))
    return best

if __name__ == '__main__':
    for A2 in [float(a) for a in sys.argv[2].split(',')]:
        thr, arg = threshold(A2, int(sys.argv[3]))
        single = max(0.0, 1 - 3 ** -0.5 / (1 - A2))
        print(json.dumps(dict(A2=A2, threshold_A1=round(thr, 5), single_atom=round(single, 5), argmax_N=arg[0] if arg else None,
                              argmax_par=arg[1] if arg else None)), flush=True)
