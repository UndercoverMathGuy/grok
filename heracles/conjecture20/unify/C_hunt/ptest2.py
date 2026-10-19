# L2 payment constant of a class-C feature:  theta2(n, s; Lambda) = sup (1 - 1/G) / eta  over atom systems
#   (b_i complex, xi_it unimodular) whose fluctuation lives on the character family Lambda (a_T = 0 for T not in Lambda),
#   A = sum b_i,  a_T = sum_i b_i prod_{t in T} xi_it,  eta = (sum_T |a_T|^2)^{1/2} / |A|,  G = |A| / (nu^{1-s} Q^s).
# Also the gain curve Gmax(eta0) = max G subject to eta <= eta0 (same Lambda).
# Usage: python ptest2.py n s nstarts seed LAMBDA     LAMBDA in {all, label, singles, 1-2, 1-12, 2-12}
import sys, itertools, json, time
import numpy as np
from scipy.optimize import minimize

n = int(sys.argv[1]); s = float(sys.argv[2]); nst = int(sys.argv[3]); seed = int(sys.argv[4]); lam = sys.argv[5]
nu = n + 1; r = 1 / s; rng = np.random.default_rng(seed)
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
Tm = np.array(Ts, float)
if lam == 'all': inL = np.ones(len(Ts), bool)
elif lam == 'label': inL = np.array([all(T) for T in Ts])
elif lam == 'singles': inL = np.array([sum(T) == 1 for T in Ts])
else:
    want = {'1-2': [(1, 0), (0, 1)], '1-12': [(1, 0), (1, 1)], '2-12': [(0, 1), (1, 1)]}[lam]
    inL = np.array([tuple(T) in want for T in Ts])
PEN = 1e4


def parts(z, m):
    return z[:m] + 1j * z[m:2 * m], z[2 * m:].reshape(m, n)


def quant(z, m, sm=1e-14):
    b, ph = parts(z, m)
    A = abs(b.sum())
    aT = (b[None, :] * np.exp(1j * Tm @ ph.T)).sum(1)
    eta = np.sqrt(np.sum(np.abs(aT[inL]) ** 2) + sm) / A
    off = np.sum(np.abs(aT[~inL]) ** 2) / A ** 2
    Q = np.sum((np.abs(b) ** 2 + sm) ** (r / 2))
    G = A / (nu ** (1 - s) * Q ** s)
    return G, eta, off


def seed_sys():
    m = int(rng.integers(2, 2 * nu + 2)); kind = rng.integers(0, 4)
    if kind == 0:
        b = np.ones(m) / m * np.exp(0.3 * rng.standard_normal(m)); ph = 2 * np.pi * np.outer(np.arange(m), np.ones(n)) / m + 0.3 * rng.standard_normal((m, n))
    elif kind == 1:
        b = 10 ** rng.uniform(-2, 0, m); ph = rng.uniform(0, 2 * np.pi, (m, n))
    elif kind == 2:   # punctured / removed-atom blocks
        k = int(rng.integers(2, nu + 1)); m = k
        b = np.ones(m) / m * np.exp(0.05 * rng.standard_normal(m)); ph = 2 * np.pi * np.outer(np.arange(m), np.ones(n)) / k + 0.05 * rng.standard_normal((m, n))
    else:
        b = rng.standard_normal(m) + 1j * rng.standard_normal(m); ph = rng.uniform(0, 2 * np.pi, (m, n))
    b = np.asarray(b, complex)
    return m, np.r_[b.real, b.imag, ph.ravel()]


t0 = time.time()
# (1) theta2
best = (-np.inf,)
for it in range(nst):
    m, z = seed_sys()
    def f(zz):
        G, eta, off = quant(zz, m)
        base = -((1 - 1 / G) / max(eta, 1e-9)) if G > 1 else -(G - 1)
        return base + PEN * off
    res = minimize(f, z, method='L-BFGS-B', options={'maxiter': 3000})
    res = minimize(f, res.x, method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-14})
    zz = res.x
    if not inL.all():   # exact constraints a_T = 0 off Lambda: SLSQP polish
        def cons(zc):
            b, ph = parts(zc, m); aT = (b[None, :] * np.exp(1j * Tm @ ph.T)).sum(1)[~inL]
            return np.r_[aT.real, aT.imag]
        def f0(zc):
            G, eta, off = quant(zc, m)
            return -((1 - 1 / G) / max(eta, 1e-9)) if G > 1 else -(G - 1)
        r2 = minimize(f0, zz, method='SLSQP', constraints=[{'type': 'eq', 'fun': cons}], options={'maxiter': 2000, 'ftol': 1e-14})
        zz = r2.x
    G, eta, off = quant(zz, m, 0.0)
    v = (1 - 1 / G) / eta if G > 1 else -1.0
    if off < 1e-12 and v > best[0]: best = (v, G, eta, m, zz, off)
if len(best) == 1: print(json.dumps(dict(n=n, s=s, Lambda=lam, theta2=None, note='no feasible cheap system found'))); sys.exit()
b_, ph_ = parts(best[4], best[3])
print(json.dumps(dict(n=n, s=s, Lambda=lam, starts=nst, theta2=best[0], at_G=best[1], at_eta=best[2], m=best[3], off=best[5],
                      masses=np.round(np.sort(np.abs(b_))[::-1] / np.abs(b_).sum(), 4).tolist(),
                      ref_c=1 - (n / nu) ** (s - 1), time=round(time.time() - t0, 1))), flush=True)
# (2) gain curve
curve = {}
for eta0 in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]:
    bg = (-np.inf, None)
    for it in range(max(6, nst // 4)):
        m, z = seed_sys()
        def f(zz):
            G, eta, off = quant(zz, m)
            return -G + PEN * (max(eta - eta0, 0) ** 2 + off)
        res = minimize(f, z, method='L-BFGS-B', options={'maxiter': 3000})
        G, eta, off = quant(res.x, m, 0.0)
        if eta <= eta0 * (1 + 1e-4) and off < 1e-10 and G > bg[0]: bg = (G, eta)
    curve[eta0] = bg
print(json.dumps(dict(n=n, s=s, Lambda=lam, gain_curve={k: [round(v[0], 6), round(v[1], 6)] if v[1] is not None else None for k, v in curve.items()},
                      linear_bound={k: round(1 / (1 - best[0] * k), 6) if best[0] * k < 1 else None for k in curve},
                      time=round(time.time() - t0, 1))), flush=True)
