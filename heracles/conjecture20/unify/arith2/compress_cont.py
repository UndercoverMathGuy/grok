# Continuous-limit compression ratio of a uniform progression block (p -> infinity, dilation invariant):
#   rho_q = min { ||b'|| : sum_l b'_l (1-cos(l x)) >= sum_{j=1}^q (1-cos(j x)) for all x in [0,pi], b' >= 0 } / ||1_{1..q}||,
#   ||b|| = (sum b^r)^s.  Frequencies l range over 1..Lmax (integers); at any prime p > 2 Lmax and any dilation d,
#   d*b' then dominates the profile of d*{1..q} at every offset, so the same certificate works at every such p.
# Reweighted LP on a fine x-grid (local minimum => an upper bound on rho_q once the certificate is verified).
# Verification: the difference polynomial is checked on a much finer grid plus a Lipschitz/derivative bound.
import numpy as np, sys
from scipy.optimize import linprog

def solve(q, s, Lmax, ngrid=4000, iters=60, seed=0, starts=4):
    r = 1/s; x = np.linspace(0, np.pi, ngrid+1)[1:]
    L = np.arange(1, Lmax+1)
    Kx = 1-np.cos(np.outer(x, L))                          # ngrid x Lmax
    rhs = (1-np.cos(np.outer(x, np.arange(1, q+1)))).sum(1)
    rng = np.random.default_rng(seed); best = (np.inf, None)
    for st in range(starts):
        w = np.ones(Lmax) if st == 0 else rng.uniform(0.3, 1.7, Lmax)
        b = None
        for it in range(iters):
            res = linprog(w, A_ub=-Kx, b_ub=-rhs, bounds=(0, None), method='highs')
            if res.status != 0: break
            b = res.x; wn = (b + 1e-9*b.max())**(r-1)
            if np.allclose(wn, w, rtol=1e-7): break
            w = wn
        if b is None: continue
        b = b*(b > 1e-10*b.max()); Q = np.sum(b[b > 0]**r)
        if Q < best[0]: best = (Q, b.copy())
    Q, b = best
    return (Q/q)**s, b

def verify(q, b, nfine=400000):
    L = np.nonzero(b)[0]+1; bw = b[L-1]
    x = np.linspace(0, np.pi, nfine+1)
    lhs = (bw[None, :]*(1-np.cos(np.outer(x, L)))).sum(1)
    rhs = (1-np.cos(np.outer(x, np.arange(1, q+1)))).sum(1)
    diff = lhs - rhs
    # derivative bound of diff: |d/dx| <= sum b_l l + sum_j j ; grid spacing h -> min over [0,pi] >= min_grid - Lip*h/2
    Lip = (bw*L).sum() + q*(q+1)/2; h = np.pi/nfine
    # near x=0 both sides vanish quadratically: check second-order coefficient separately
    c2 = 0.5*((bw*L**2).sum() - (np.arange(1, q+1)**2).sum())
    return diff.min(), Lip*h/2, c2

if __name__ == '__main__':
    s = float(sys.argv[1]); qs = [int(a) for a in sys.argv[2].split(',')]
    for q in qs:
        rho, b = solve(q, s, Lmax=max(40, 6*q), ngrid=3000)
        dmin, lipslack, c2 = verify(q, b)
        L = np.nonzero(b)[0]+1
        print(f"s={s} q={q}: rho_q <= {rho:.4f} (threshold (2/3)^(s-1) = {(2/3)**(s-1):.4f}); |spt b'|={len(L)} "
              f"maxfreq={L.max()}; fine-grid min(lhs-rhs)={dmin:.3e} (Lipschitz slack {lipslack:.1e}); x^2-coef={c2:.3e}", flush=True)
