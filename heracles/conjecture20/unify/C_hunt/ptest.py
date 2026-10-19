# Falsification search for (P), the L1 payment of a class-C feature:  G * (1 - eps/2) <= 1, where for an atom system
#   (b_i complex, xi_it unimodular), A = sum b_i, a_T = sum_i b_i prod_{t in T} xi_it (T nonempty),
#   eps = sum_T |a_T| / |A|,  G = |A| / (nu^{1-s} (sum |b_i|^r)^s),  r = 1/s.
# (Only eps < 2 matters.)  Maximise G (1 - eps/2) by L-BFGS from many random / IF-perturbed / cheap starts.
# Usage: python ptest.py n s nstarts seed
import sys, itertools, json, time
import numpy as np
from scipy.optimize import minimize

n = int(sys.argv[1]); s = float(sys.argv[2]); nst = int(sys.argv[3]); seed = int(sys.argv[4])
mode = sys.argv[5] if len(sys.argv) > 5 else 'P'   # 'theta': payment constant theta_P = sup (1-1/G)/eps
nu = n + 1; r = 1 / s; rng = np.random.default_rng(seed)
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
Tm = np.array(Ts, float)


def parts(z, m):
    b = z[:m] + 1j * z[m:2 * m]; ph = z[2 * m:].reshape(m, n)
    return b, ph


def val(z, m, smooth=1e-12):
    b, ph = parts(z, m)
    A = abs(b.sum())
    aT = (b[None, :] * np.exp(1j * Tm @ ph.T)).sum(1)              # (nT,)
    eps = np.sum(np.sqrt(np.abs(aT) ** 2 + smooth)) / A
    Q = np.sum((np.abs(b) ** 2 + smooth) ** (r / 2))
    G = A / (nu ** (1 - s) * Q ** s)
    return G * (1 - eps / 2), G, eps


t0 = time.time(); best = (-np.inf,)
for it in range(nst):
    m = int(rng.integers(2, 2 * nu + 2))
    kind = it % 3
    if kind == 0:   # IF-perturbed
        b = np.ones(m) / m * np.exp(0.3 * rng.standard_normal(m)); ph = 2 * np.pi * np.outer(np.arange(m), np.ones(n)) / m
        ph = ph + 0.3 * rng.standard_normal((m, n))
    elif kind == 1:  # concentrated (cheap) masses
        b = 10 ** rng.uniform(-2, 0, m); ph = rng.uniform(0, 2 * np.pi, (m, n))
    else:
        b = rng.standard_normal(m) + 1j * rng.standard_normal(m); ph = rng.uniform(0, 2 * np.pi, (m, n))
    b = np.asarray(b, complex)
    z = np.r_[b.real, b.imag, ph.ravel()]
    def f(zz):
        v_, G_, e_ = val(zz, m)
        return -((1 - 1 / G_) / max(e_, 1e-9)) if G_ > 1 else -(G_ - 1)    # maximise theta = (1 - 1/G)/eps on G > 1
    if mode != 'theta': f = lambda zz: -val(zz, m)[0]
    res = minimize(f, z, method='L-BFGS-B', options={'maxiter': 3000})
    res = minimize(f, res.x, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-13})
    v, G, eps = val(res.x, m, 0.0)
    if mode == 'theta': v = (1 - 1 / G) / eps if G > 1 else -1.0
    if v > best[0]: best = (v, G, eps, m, res.x)
if mode == 'theta':
    b_, ph_ = parts(best[4], best[3])
    print(json.dumps(dict(n=n, s=s, starts=nst, theta_P=best[0], at_G=best[1], at_eps=best[2], m=best[3],
                          masses=np.round(np.sort(np.abs(b_))[::-1] / np.abs(b_).sum(), 4).tolist(),
                          time=round(time.time() - t0, 1))), flush=True)
else:
    print(json.dumps(dict(n=n, s=s, starts=nst, max_G_times_1_minus_eps_over_2=best[0], at_G=best[1], at_eps=best[2], m=best[3],
                          P_holds=bool(best[0] <= 1 + 1e-9), time=round(time.time() - t0, 1))), flush=True)
