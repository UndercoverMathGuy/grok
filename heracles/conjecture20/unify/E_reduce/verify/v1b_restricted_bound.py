# R2(a) adversarial: maximise gamma_lam over a family of N_S networks that carry cancelling interference at j notin S
# (each base neuron has a twin whose frequency-j output, j notin S, is negated, so beta_j = 0 exactly).
import numpy as np, sys
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_reduce/verify')
from vlib import *
rng = np.random.default_rng(7)

def build(x, m, n, p, flip):
    U = x[:m * n * p].reshape(m, n, p); W = x[m * n * p:].reshape(m, p)
    return np.concatenate([U, U]), np.concatenate([W, W @ flip])

def obj(x, m, n, p, flip, lam, b, tau):
    U, W = build(x, m, n, p, flip); f = table(U, W); F = (p - 1) // 2
    g = sum(lam[l - 1] * Gl(f, p, n, l) for l in range(1, F + 1)).ravel()
    s = (n + 1) / b; c = cost(U, W, b) ** s
    g = g / c; gm = g.min()
    return -(gm - tau * np.log(np.exp(-(g - gm) / tau).sum()))

out = []
for case in range(10):
    p = 5; n = 2; nu = 3; F = 2
    S = [[1], [2], [1, 2]][case % 3]
    lam = rng.uniform(0.7, 1.4, F)
    b = [2.0, 2.4, 3.0, 2.0, 2.7][case % 5]; s = nu / b
    flip = proj(p, 0) + sum((1 if l in S else -1) * proj(p, l) for l in range(1, F + 1))
    m = 4
    best = -np.inf
    for start in range(3):
        x = rng.normal(size=m * n * p + m * p)
        for tau in [3e-2, 1e-2, 3e-3, 1e-3]:
            sc = Bp(p, n)
            res = minimize(obj, x, args=(m, n, p, flip, lam, b, tau * sc), method='L-BFGS-B', options={'maxiter': 400})
            x = res.x
        U, W = build(x, m, n, p, flip); f = table(U, W)
        be = beta(f, p, n)
        assert max([abs(be[j - 1]) for j in range(1, F + 1) if j not in S] + [0]) < 1e-9
        h = sum(lam[l - 1] * Gl(f, p, n, l) for l in range(1, F + 1)).min()
        Gam, M = programme(p, s, lam=lam, S=S)
        best = max(best, h / (Bp(p, n) * Gam * cost(U, W, b) ** s))
    # interference mass on j notin S of the best net (last one)
    out.append((S, np.round(lam, 3), b, best))
    print('S=%s lam=%s b=%.2f: best h_lam/(B_p Gamma^{lam,(S)} P^s) = %.6f' % (S, np.round(lam, 3), b, best), flush=True)
print('max ratio %.6f (must be <= 1)' % max(o[3] for o in out))
