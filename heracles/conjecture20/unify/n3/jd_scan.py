"""True joint detection constant at n=3 (class C profiles), numerically.

f_T : Z_p -> R mean zero for the 7 non-empty T in [3].
F(u)   = sum_T f_T(s_T(u)),  F^d(u) = sum_T f_T(s_T(u) + d),  u in Z_p^3.
A = -min F, A' = max F^d, E = sum_T Var f_T.
kappa_3(p) = inf A A' / E   (n=2 analogue: (p-1)/(p+1), sawtooth).
Local search (evidence only): soft-min/max + L-BFGS, then exact evaluation.
usage: python jd_scan.py p d restarts seed [restrict]
 restrict: 'all' (7 sets), 'single' (singletons only), 'pairs', 'no123', ...
"""
import sys, itertools
import numpy as np
from scipy.optimize import minimize

p = int(sys.argv[1]); d = int(sys.argv[2]); R = int(sys.argv[3]); seed = int(sys.argv[4])
restrict = sys.argv[5] if len(sys.argv) > 5 else 'all'
n = 3
TS_all = [T for k in range(1, n + 1) for T in itertools.combinations(range(n), k)]
if restrict == 'all':
    TS = TS_all
elif restrict == 'single':
    TS = [T for T in TS_all if len(T) == 1]
elif restrict == 'pairs':
    TS = [T for T in TS_all if len(T) == 2]
elif restrict == 'no123':
    TS = [T for T in TS_all if len(T) < 3]
elif restrict == 'n2':
    TS = [(0,), (1,), (0, 1)]
else:
    TS = [tuple(int(c) - 1 for c in t) for t in restrict.split(',')]
U = np.array(list(itertools.product(range(p), repeat=n)))
S = np.stack([U[:, list(T)].sum(axis=1) % p for T in TS], axis=1)  # p^3 x |TS|
nT = len(TS)


def build(x):
    f = x.reshape(nT, p - 1)
    f = np.concatenate([f, -f.sum(axis=1, keepdims=True)], axis=1)  # mean zero
    return f


def evalF(f):
    Fv = sum(f[k][S[:, k]] for k in range(nT))
    Fd = sum(f[k][(S[:, k] + d) % p] for k in range(nT))
    return Fv, Fd


def ratio(x):
    f = build(x)
    Fv, Fd = evalF(f)
    E = (f ** 2).mean(axis=1).sum()
    return (-Fv.min()) * Fd.max() / E


def soft(x, k):
    f = build(x)
    Fv, Fd = evalF(f)
    E = (f ** 2).mean(axis=1).sum()
    sc = np.sqrt(E)
    a = -Fv / sc; b = Fd / sc
    A = a.max() + np.log(np.exp(k * (a - a.max())).sum()) / k
    B = b.max() + np.log(np.exp(k * (b - b.max())).sum()) / k
    return A * B


rng = np.random.default_rng(seed)
best = (np.inf, None)
for rr in range(R):
    x = rng.normal(size=nT * (p - 1))
    if rr % 3 == 1:   # sawtooth-like seeds
        saw = np.array([((-a) % p) - (p - 1) / 2 for a in range(p)], dtype=float)
        x = np.concatenate([saw[:p - 1] * rng.choice([-1, 1]) + 0.3 * rng.normal(size=p - 1) for _ in range(nT)])
    for k in (5.0, 20.0, 80.0, 300.0):
        res = minimize(soft, x, args=(k,), method='L-BFGS-B', options=dict(maxiter=2000))
        x = res.x
    res = minimize(ratio, x, method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-9, fatol=1e-12))
    v = ratio(res.x)
    if v < best[0]:
        best = (v, res.x)
print(f"p={p} d={d} restrict={restrict} sets={len(TS)} best AA'/E = {best[0]:.6f}  (p-1)/(p+1)={(p-1)/(p+1):.6f}  slicing k/(2+k)={((p-1)/(p+1))/(2+(p-1)/(p+1)):.6f}")
f = build(best[1])
E = (f ** 2).mean(axis=1).sum()
for k, T in enumerate(TS):
    print('  T=', ''.join(str(t + 1) for t in T), ' Var share=%.4f' % ((f[k] ** 2).mean() / E), ' f=', np.round(f[k] / np.sqrt(E), 4))
