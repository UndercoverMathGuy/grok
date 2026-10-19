# Independent referee check of largen R4 (robust width from subordination; slot-symmetric class C).
# Field of slot-symmetric atoms: P(z) = sum_i beta_i prod_t (1 + xi_i z_t), z in mu_p^n; depends only on counts N_a of z=w^a.
import math, itertools
import numpy as np
from math import comb
from scipy.optimize import minimize
import mpmath as mp
rng = np.random.default_rng(4242)

def counts(n, p):
    out = []
    def rec(prefix, left, k):
        if k == p - 1: out.append(prefix + [left]); return
        for v in range(left + 1): rec(prefix + [v], left - v, k + 1)
    rec([], n, 0); return np.array(out, float)

def field(beta, xi, p, Ns):
    w = np.exp(2j * np.pi * np.arange(p) / p)
    L = np.log(1 + xi[:, None] * w[None, :] + 0j)          # (m,p); 1+xi w != 0 generically
    return np.exp(Ns @ L.T) @ beta

def cmin(beta, xi, p, Ns):
    P = field(beta, xi, p, Ns)
    return (P.real - np.abs(P.imag) / math.tan(math.pi / p)).min()

def rhs_min_over_q(beta, xi, A, c, n, p):
    # minimise over q (deg<=n, q(0)=1) of sum_i |beta_i||q(xi_i)| + (4/p)|A-c| sum_k |q_k|/(C(n,k) R^k)  (convex)
    R = math.cos(math.pi / p)
    V = np.vander(xi, n + 1, increasing=True)               # (m, n+1)
    def obj(v):
        q = np.concatenate([[1], v[:n] + 1j * v[n:]])
        return np.sum(np.abs(beta) * np.abs(V @ q)) + (4 / p) * abs(A - c) * sum(abs(q[k]) / (comb(n, k) * R ** k) for k in range(1, n + 1))
    best = obj(np.zeros(2 * n))
    for s in range(4):
        v0 = 0.3 * rng.standard_normal(2 * n) if s else np.zeros(2 * n)
        r = minimize(obj, v0, method='Powell', options=dict(maxiter=20000, xtol=1e-10, ftol=1e-13))
        best = min(best, r.fun)
    return best

print("(i) R4 inequality |A| <= min_q RHS(q) on random slot-symmetric class-C atom systems (any margin, c = min cm)")
worst = 0; cnt = 0
for (p, n, m) in [(3, 3, 3), (5, 3, 4), (5, 4, 3), (7, 4, 5), (7, 5, 3), (11, 4, 5), (13, 5, 4), (5, 6, 7)]:
    Ns = counts(n, p)
    for t in range(12):
        if t % 2 == 0:   # near IF block
            nu = n + 1
            beta = np.ones(nu, complex) / nu * np.exp(0.4 * (rng.standard_normal(nu) + 1j * rng.standard_normal(nu)))
            xi = np.exp(2j * np.pi * np.arange(nu) / nu + 0.4j * rng.standard_normal(nu))
        else:
            beta = rng.standard_normal(m) + 1j * rng.standard_normal(m); xi = np.exp(2j * np.pi * rng.random(m))
        A = beta.sum(); c = cmin(beta, xi, p, Ns)
        r = abs(A) / rhs_min_over_q(beta, xi, A, c, n, p)
        worst = max(worst, r); cnt += 1
print(f"  {cnt} systems: max |A| / min_q RHS = {worst:.4f} (must be <= 1)")

print("(ii) R4(a): positive margin forces m > n pR^2/(4+pR); search for positive margin with m <= bound")
def best_margin(p, n, m, tries=40):
    Ns = counts(n, p); best = -np.inf
    for t in range(tries):
        v0 = np.concatenate([rng.standard_normal(m) * 0.5, rng.uniform(0, 2 * np.pi, m), rng.uniform(0, 2 * np.pi, m)])
        f = lambda v: -cmin(np.exp(v[:m] + 1j * v[m:2 * m]), np.exp(1j * v[2 * m:]), p, Ns) / np.exp(v[:m]).sum()
        r = minimize(f, v0, method='Powell', options=dict(maxiter=6000))
        best = max(best, -r.fun)
    return best
for (p, n) in [(5, 4), (5, 7), (7, 6), (11, 6)]:
    R = math.cos(math.pi / p); bd = n * p * R * R / (4 + p * R)
    mmax = math.floor(bd) if bd != int(bd) else int(bd)
    out = []
    for m in range(max(1, mmax - 0), n + 2):
        out.append((m, best_margin(p, n, m, tries=10)))
    print(f"  p={p} n={n}: bound says m > {bd:.3f}; best c/L by width m: " + ", ".join(f"m={m}:{v:+.4f}" for m, v in out), flush=True)

print("(iii) R4(b): p cos(pi/p)^(n+1) > 4n for all p >= 4n+5 (interval arithmetic at p = 4n+5, n <= 3000) + n-atom searches")
mp.iv.dps = 30
bad = 0; minslack = None
for n in range(1, 3001):
    p = 4 * n + 5
    val = mp.iv.mpf(p) * mp.iv.cos(mp.iv.pi / p) ** (n + 1) - 4 * n
    if not (val.a > 0): bad += 1
    s = float(val.a); minslack = s if minslack is None else min(minslack, s)
print(f"  at p=4n+5: failures {bad}, min lower-endpoint slack {minslack:.4f}; (p - (n+1)pi^2/(2p) increasing in p covers p > 4n+5)")
for (p, n) in [(13, 2), (17, 3), (23, 4), (11, 2)]:
    R = math.cos(math.pi / p); cond = p * R ** (n + 1) > 4 * n
    print(f"  p={p} n={n}: condition {cond}; best c/L with n atoms = {best_margin(p, n, n, tries=12):+.5f} (theorem: <= 0 if condition)", flush=True)

print("(iv) R4(c): equally spaced phases, nu atoms, arbitrary complex beta: c <= nu min|beta| under the condition")
for (p, n) in [(11, 3), (13, 4), (11, 5), (7, 2)]:
    R = math.cos(math.pi / p); NR = sum(1 / (comb(n, k) * R ** k) for k in range(1, n + 1)); cond = p * R >= 4 * NR
    Ns = counts(n, p); nu = n + 1; best = -np.inf
    def f(v):
        beta = np.exp(v[:nu] + 1j * v[nu:2 * nu]); xi = np.exp(1j * (2 * np.pi * np.arange(nu) / nu + v[-1]))
        return -cmin(beta, xi, p, Ns) / (nu * np.abs(beta).min())
    for t in range(15):
        v0 = np.concatenate([0.6 * rng.standard_normal(nu), 0.6 * rng.standard_normal(nu), [rng.uniform(0, 2 * np.pi)]])
        r = minimize(f, v0, method='Powell', options=dict(maxiter=8000))
        best = max(best, -r.fun)
    print(f"  p={p} n={n}: condition pR >= 4N_R: {cond} (pR={p*R:.2f}, 4N_R={4*NR:.2f}); max c/(nu min|beta|) found = {best:.4f}", flush=True)
