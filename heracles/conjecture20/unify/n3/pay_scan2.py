"""Numerical sup of n=3 single-feature payment ratios (evidence only).

which = 'pt'  : bonus / max_T |beta_T|              (pointwise single-T, c_1^1(3,s))
        'L2'  : bonus / sqrt(sum_T |beta_T|^2)       (L2, c_L2(3,s))
        'cf'  : bonus / max over collision-free families of sum |beta_T|
mode = 'gen' (general atoms) or 'sym' (slot-symmetric atoms xi_it = e^{i phi_i}).
usage: python pay_scan2.py which s Nmax restarts seed mode
"""
import sys
import numpy as np
from scipy.optimize import minimize
from n3lib import coeffs, unpack, unpack_sym, TNAMES

which, s = sys.argv[1], float(sys.argv[2])
Nmax, R, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
mode = sys.argv[6] if len(sys.argv) > 6 else 'gen'
theta = 4.0 ** (1 - s)
# collision-free families for n=3 (Hn/result.tex): 7 singletons, 3 pairs of 2-sets, {12,13,23}
idx = {nm: k for k, nm in enumerate(TNAMES)}
FAMS = [[k] for k in range(7)] + [[idx['12'], idx['13']], [idx['12'], idx['23']], [idx['13'], idx['23']],
                                   [idx['12'], idx['13'], idx['23']]]


def den(bT, smooth=None):
    m = np.abs(bT)
    if which == 'pt':
        if smooth is None:
            return m.max()
        M = m.max()
        return M + np.log(np.sum(np.exp(smooth * (m - M) / M))) * M / smooth
    if which == 'L2':
        return np.sqrt(np.sum(m ** 2))
    vals = np.array([m[f].sum() for f in FAMS])
    if smooth is None:
        return vals.max()
    M = vals.max()
    return M + np.log(np.sum(np.exp(smooth * (vals - M) / M))) * M / smooth


def ratio(v, N, smooth=None):
    beta, ang = (unpack_sym if mode == 'sym' else unpack)(v, N)
    A, bT = coeffs(beta, ang)
    Q = np.sum(np.abs(beta) ** (1.0 / s))
    d = den(bT, smooth)
    return (abs(A) - theta * Q ** s) / max(d, 1e-14)


rng = np.random.default_rng(seed)
best = (-np.inf, None, None)
for N in range(1, Nmax + 1):
    bN = -np.inf
    for t in range(R):
        k = 3 * N if mode == 'sym' else 5 * N
        v = np.concatenate([rng.uniform(0.2, 1.0, N), rng.uniform(-np.pi, np.pi, k - N)])
        for sm in ([None] if which == 'L2' else [20.0, 80.0, 400.0]):
            res = minimize(lambda w: -ratio(w, N, sm), v, method='L-BFGS-B', options=dict(maxiter=3000))
            v = res.x
        res = minimize(lambda w: -ratio(w, N), v, method='Nelder-Mead',
                       options=dict(maxiter=20000, xatol=1e-11, fatol=1e-14, adaptive=True))
        val = ratio(res.x, N)
        if val > bN:
            bN = val
        if val > best[0]:
            best = (val, N, res.x)
    print(f"which={which} mode={mode} s={s} N={N}: best {bN:.6f}", flush=True)
val, N, v = best
beta, ang = (unpack_sym if mode == 'sym' else unpack)(v, N)
A, bT = coeffs(beta, ang)
rot = np.conj(A) / abs(A)
print(f"BEST which={which} mode={mode} s={s} value={val:.6f} N={N}  single-atom: pt={1 - 4 ** (1 - s):.6f} "
      f"L2={(1 - 4 ** (1 - s)) / np.sqrt(7):.6f}")
print("  |beta_i|/|A| =", np.round(np.abs(beta) / abs(A), 5).tolist())
print("  arg(beta_i/A) =", np.round(np.angle(beta * rot), 5).tolist())
print("  xi angles =", np.round(np.mod(ang, 2 * np.pi), 4).tolist())
print("  |beta_T|/|A| =", dict(zip(TNAMES, np.round(np.abs(bT) / abs(A), 5).tolist())))
