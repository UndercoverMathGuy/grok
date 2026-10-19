# (i) equally spaced phases, nu atoms, arbitrary complex beta: claim c <= nu*min|beta_j| when p cos(pi/p) >= 4 N_R,
#     N_R = sum_{k=1}^n 1/(C(n,k) R^k), R=cos(pi/p).   (ii) sharp width: p cos^{n+1}(pi/p) > 4n  =>  no positive margin with <= n atoms.
import math, numpy as np
from math import comb
rng = np.random.default_rng(11)
def counts(n, p):
    out = []
    def rec(prefix, left, k):
        if k == p - 1: out.append(prefix + [left]); return
        for v in range(left + 1): rec(prefix + [v], left - v, k + 1)
    rec([], n, 0); return np.array(out, float)
def margin(beta, xi, n, p, N):
    om = np.exp(4j*np.pi*np.arange(p)/p)
    P = np.exp(N @ np.log(1 + xi[:, None]*om[None, :]).T) @ beta
    return (P.real - abs(P.imag)/math.tan(math.pi/p)).min()
print("(i) equally spaced phases")
for p, n in [(11, 3), (11, 6), (13, 5), (13, 8), (17, 6), (7, 2), (5, 2)]:
    R = math.cos(math.pi/p); NR = sum(1/(comb(n, k)*R**k) for k in range(1, n+1)); cond = p*R >= 4*NR
    N = counts(n, p); nu = n+1; worst = -np.inf; cnt = 0
    for t in range(3000):
        beta = (1/nu)*np.exp(rng.uniform(0, 0.6)*rng.standard_normal(nu) + 1j*rng.uniform(0, 0.6)*rng.standard_normal(nu))
        xi = np.exp(2j*np.pi*np.arange(nu)/nu + 1j*rng.uniform(0, 2*np.pi))
        c = margin(beta, xi, n, p, N)
        if c > 0:
            cnt += 1; worst = max(worst, c/(nu*np.abs(beta).min()))
    print(f"  p={p} n={n}: condition p R >= 4 N_R: {cond} (pR={p*R:.2f}, 4N_R={4*NR:.2f});  {cnt} nets with c>0, max c/(nu min|beta|) = {worst:.4f}")
print("(ii) sharp width (<= n atoms, slot-symmetric class C): search for positive margin")
for p, n in [(13, 2), (17, 3), (23, 4), (29, 5), (11, 3)]:
    R = math.cos(math.pi/p); cond = p*R**(n+1) > 4*n
    N = counts(n, p); best = -np.inf
    for t in range(4000):
        m = n
        beta = np.exp(rng.standard_normal(m)*0.7 + 1j*rng.uniform(0, 2*np.pi, m))
        xi = np.exp(1j*rng.uniform(0, 2*np.pi, m))
        c = margin(beta, xi, n, p, N); best = max(best, c/np.abs(beta).sum())
    print(f"  p={p} n={n}: theorem applies: {cond};  best c/L over 4000 random n-atom systems = {best:.4f} (theorem: <= 0)")
