# f.3: for which torus radii (A1, A2) does  sum_l |c_l|^{2/3} >= 3^{1/3} m_+^{2/3}  hold for every population?
# h(t1,t2) = sum_l c_l (1 + v_l1 t1)(1 + v_l2 t2);  m = min over |t1| = A1, |t2| = A2 of Re h.
# For fixed t2 the min over t1 is Re(G + J2 t2) - A1 |J1 + J12 t2|, so m is a 1-D min over arg t2.
import sys, json, numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
BETA = np.linspace(0, 2 * np.pi, 721)[:-1]

def m_of(c, v1, v2, A1, A2):
    G = c.sum(); J1 = (c * v1).sum(); J2 = (c * v2).sum(); J12 = (c * v1 * v2).sum()
    t2 = A2 * np.exp(1j * BETA)
    f = np.real(G + J2 * t2) - A1 * np.abs(J1 + J12 * t2)
    i = f.argmin()
    g = lambda b: np.real(G + J2 * A2 * np.exp(1j * b)) - A1 * abs(J1 + J12 * A2 * np.exp(1j * b))
    r = minimize(lambda b: g(b[0]), [BETA[i]], method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-14))
    return min(f[i], r.fun)

def ratio(par, N, A1, A2):
    x = np.exp(par[:N]); c = x * np.exp(1j * par[N:2 * N])
    v1 = np.exp(1j * par[2 * N:3 * N]); v2 = np.exp(1j * par[3 * N:4 * N])
    m = m_of(c, v1, v2, A1, A2)
    if m <= 1e-12:
        return 10.0
    return (x ** (2 / 3)).sum() / (3 ** (1 / 3) * m ** (2 / 3))

def best_ratio(A1, A2, restarts=24):
    best = (9, None)
    for N in (1, 2, 3, 4, 5):
        for _ in range(restarts):
            par = np.r_[rng.normal(0, 0.7, N), rng.uniform(0, 2 * np.pi, 3 * N)]
            if N >= 3 and rng.uniform() < 0.5:     # start near the junk-free block (phases pi l/3 pattern), perturbed
                par = np.r_[np.zeros(N), rng.normal(0, 0.3, N), np.pi * np.arange(N) / 3 * 2 + rng.normal(0, 0.3, N),
                            np.pi * np.arange(N) / 3 * 2 + rng.normal(0, 0.3, N)]
            r = minimize(ratio, par, args=(N, A1, A2), method='Nelder-Mead',
                         options=dict(maxiter=6000, xatol=1e-9, fatol=1e-11))
            if r.fun < best[0]:
                best = (r.fun, (N, r.x.tolist()))
    return best

if __name__ == '__main__':
    out = {}
    for A2 in [float(a) for a in sys.argv[2].split(',')]:
        lo, hi = max(0.0, 1 - 3 ** -0.5 / (1 - A2)) if A2 < 1 else 0.0, 0.75   # single-atom curve is a lower bound
        val_lo = best_ratio(lo, A2)[0]
        for _ in range(9):
            mid = (lo + hi) / 2
            v = best_ratio(mid, A2)[0]
            if v >= 1 - 1e-7: hi = mid
            else: lo = mid
        out[A2] = dict(single_atom_A1=max(0.0, 1 - 3 ** -0.5 / (1 - A2)), search_A1_threshold=hi, ratio_at_single_atom_curve=val_lo)
        print(json.dumps({A2: out[A2]}), flush=True)
