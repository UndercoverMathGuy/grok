# Check of R4 (robust width from diagonal subordination, slot-symmetric class C):
#   |A| <= sum_i |beta_i||q(xi_i)| + (4/p)|A-c| N_R(q),  N_R(q) = sum_{k>=1} |q_k| / (C(n,k) R^k),  for every q, deg<=n, q(0)=1.
# On positive-margin slot-symmetric class-C networks (perturbed IF blocks), with q = random and q = prod(z-xi_i) (m<=n).
import math, itertools, numpy as np
from math import comb
rng = np.random.default_rng(7)
def counts(n, p):
    out = []
    def rec(prefix, left, k):
        if k == p - 1: out.append(prefix + [left]); return
        for v in range(left + 1): rec(prefix + [v], left - v, k + 1)
    rec([], n, 0); return np.array(out, float)
worst = 0; tested = 0
for p, n in [(5, 4), (7, 4), (7, 6), (11, 5), (13, 6)]:
    N = counts(n, p); om = np.exp(4j*np.pi*np.arange(p)/p); R = math.cos(math.pi/p)
    for trial in range(150):
        nu = n + 1
        beta = np.ones(nu, complex)/nu * np.exp(rng.uniform(0, .3)*(rng.standard_normal(nu)+1j*rng.standard_normal(nu)))
        xi = np.exp(2j*np.pi*np.arange(nu)/nu + 1j*rng.uniform(0, .3)*rng.standard_normal(nu))
        P = np.exp(N @ np.log(1 + xi[:, None]*om[None, :]).T) @ beta
        c = (P.real - abs(P.imag)/math.tan(math.pi/p)).min()
        if c <= 0: continue
        tested += 1; A = beta.sum()
        for t in range(20):
            if t == 0:
                # q vanishing on all but one atom (m-1 <= n roots), normalised q(0)=1
                roots = xi[1:]; q = np.poly(roots)[::-1]; q = q / q[0]
            else:
                q = np.concatenate([[1], (rng.standard_normal(n) + 1j*rng.standard_normal(n)) * rng.uniform(0, 1) / np.arange(1, n+1)])
            qx = np.array([np.polyval(q[::-1], x) for x in xi])
            NR = sum(abs(q[k])/(comb(n, k)*R**k) for k in range(1, len(q)))
            rhs = np.sum(abs(beta)*abs(qx)) + (4/p)*abs(A - c)*NR
            worst = max(worst, abs(A)/rhs)
print(f"R4 check: {tested} positive-margin slot-symmetric class-C nets, max |A|/RHS = {worst:.4f} (must be <= 1)")
# width corollary numbers
for p in [5, 7, 11, 13, 101]:
    R = math.cos(math.pi/p); print(f"  p={p}: positive margin forces m > n * {p*R*R/(4+p*R):.4f}   (T1: m >= ~ n/2*(1-1/p) = n*{0.5*(1-1/p):.4f})")
