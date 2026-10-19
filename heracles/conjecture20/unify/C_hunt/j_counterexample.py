# Explicit counterexample to (J) with constant 1/2 (paired joint L1 detection), and the detection constant
# theta*(p,n) = rho*/2 recomputed with the optimal phases, re-evaluated exactly (mpmath, 40 digits) on the full grid.
# Usage: python j_counterexample.py p n nphase seed
import sys, itertools, json
import numpy as np
import mpmath as mp
from scipy.optimize import linprog

p = int(sys.argv[1]); n = int(sys.argv[2]); nph = int(sys.argv[3]); seed = int(sys.argv[4])
F = (p - 1) // 2; rng = np.random.default_rng(seed); d0 = 1
X = np.indices((p,) * n).reshape(n, -1).T
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
cols = [(l, T) for l in range(1, F + 1) for T in Ts]
chi = np.array([np.exp(2j * np.pi * 2 * l * (X @ np.array(T)) / p) for (l, T) in cols])
cv = {d: np.array([1 - np.exp(-2j * np.pi * l * d / p) for (l, T) in cols]) for d in (d0, p - d0)}
w = np.abs(cv[d0]) / 2


def rho_of(psi):
    A = np.vstack([-np.real((np.exp(1j * psi) * cv[d])[:, None] * chi).T for d in (d0, p - d0)])
    res = linprog(np.r_[np.zeros(len(cols)), 1.0], A_ub=np.hstack([A, -np.ones((A.shape[0], 1))]), b_ub=np.zeros(A.shape[0]),
                  A_eq=np.r_[w, 0.0][None, :], b_eq=[1.0], bounds=[(0, None)] * len(cols) + [(None, None)], method='highs')
    return res.fun, res.x[:len(cols)]


best = (np.inf, None, None)
for it in range(nph):
    psi = np.zeros(len(cols)) if it == 0 else rng.uniform(0, 2 * np.pi, len(cols))
    v, r = rho_of(psi); step = 0.6
    for sweep in range(6):
        for k in range(len(cols)):
            for dl in (step, -step):
                ps = psi.copy(); ps[k] += dl; v2, r2 = rho_of(ps)
                if v2 < v - 1e-12: psi, v, r = ps, v2, r2
        step *= 0.5
    if v < best[0]: best = (v, psi, r)
v, psi, r = best
# exact re-evaluation: a_{l,T} = r e^{i psi}; D(x, +-d0) on the whole grid in mpmath
mp.mp.dps = 40
pi = mp.pi
a = [mp.mpf(float(r[k])) * mp.expj(mp.mpf(float(psi[k]))) for k in range(len(cols))]
L1 = mp.fsum(abs(a[k]) * abs(1 - mp.expj(-2 * pi * cols[k][0] * d0 / p)) for k in range(len(cols)))
worst = {}
for d in (d0, p - d0):
    mn = None
    for x in X:
        val = mp.fsum(mp.re(a[k] * (1 - mp.expj(-2 * pi * cols[k][0] * d / p)) *
                            mp.expj(2 * pi * 2 * cols[k][0] * int(np.dot(x, cols[k][1])) / p)) for k in range(len(cols)) if r[k] > 0)
        mn = val if mn is None or val < mn else mn
    worst[d] = mn
loss = max(-worst[d0], -worst[p - d0])
print(json.dumps(dict(p=p, n=n, rho_lp=v, theta_star=v / 2, exact_loss_over_L1=float(loss / L1),
                      J_half_requires=0.5, J_half_fails=bool(loss / L1 < 0.5),
                      coefficients=[(cols[k][0], cols[k][1], float(r[k]), float(psi[k] % (2 * np.pi)))
                                    for k in range(len(cols)) if r[k] > 1e-12]), default=str))
