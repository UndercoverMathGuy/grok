# Compression ratio of a clean block b on Z_p:
#   rho(b) = min{ ||b'|| : K b' >= K b pointwise (all delta != 0), b' >= 0 } / ||b||,   ||x|| = (sum x^r)^s.
# If rho(b) <= (2/3)^{s-1}, a dirty block with allocation b (any interference, any phases, any p) can be replaced by a
# cheaper clean block whose coverage dominates (3/2)^{s-1} K b, so it cannot beat IF.  Computed by reweighted LP
# (concave minimisation over a polyhedron: local minima only, so rho_found >= rho_true; an UPPER bound on rho).
import numpy as np, sys
from scipy.optimize import linprog

def kmat(p):
    F = (p-1)//2; j = np.arange(1, F+1)
    return 1 - np.cos(2*np.pi*np.outer(j, j)/p)          # rows delta, cols l (both in 1..F)

def compress(p, b, s=1.5, restarts=8, seed=0, lam=1.0):
    r = 1/s; K = kmat(p); F = K.shape[0]; rng = np.random.default_rng(seed)
    rhs = lam * (K @ b)
    best = (np.inf, None)
    starts = [np.ones(F)] + [rng.uniform(0.2, 2, F) for _ in range(restarts-1)]
    starts.append((b + 1e-3)**(r-1))
    for w in starts:
        for it in range(80):
            res = linprog(w, A_ub=-K, b_ub=-rhs, bounds=(0, None), method='highs')
            if res.status != 0: break
            x = res.x; wn = (x + 1e-10*max(1, x.max()))**(r-1)
            if np.allclose(wn, w, rtol=1e-7): break
            w = wn
        x = x*(x > 1e-12*x.max()); Q = np.sum(x**r)
        if Q < best[0]: best = (Q, x.copy())
    Qb = np.sum(b[b > 0]**r)
    return (best[0]/Qb)**s, best[1]

if __name__ == '__main__':
    s = 1.5; thr = (2/3)**(s-1)
    for p in [int(a) for a in sys.argv[1].split(',')]:
        F = (p-1)//2
        for N in [int(a) for a in sys.argv[2].split(',')]:
            if N > F: continue
            b = np.zeros(F); b[:N] = 1.0
            rho, x = compress(p, b, s)
            S = np.nonzero(x)[0] + 1
            print(f"p={p} progression N={N}: rho={rho:.4f}  (threshold {thr:.4f})  |spt b'|={len(S)}  spt={list(S[:12])}{'...' if len(S)>12 else ''}", flush=True)
