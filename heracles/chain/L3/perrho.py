# L3-A with a separate rho_k per frequency (still PROVED: same proof, lambda_k = rho_k conj(z_k)/|z_k|, rho_k <= 1/2):
#   A* >= min_k (rho_k - E_k) / (1 + E_0),  E = sum over patterns of prod_{i in supp} rho_i.
# Optimised numerically; the final value is re-evaluated exactly at a rational rho vector (CERTIFIED per set).
import sys, math, itertools, numpy as np
from fractions import Fraction as Fr
from scipy.optimize import minimize
def patterns(p, H):
    K = len(H); m = [(2 * k) % p for k in H]
    rel, rep = [], [[] for _ in H]
    for eps in itertools.product((-1, 0, 1), repeat=K):
        s = sum(e * mk for e, mk in zip(eps, m)) % p; supp = tuple(i for i in range(K) if eps[i])
        if not supp: continue
        if s == 0: rel.append((supp, len(supp) % 2))
        for k in range(K):
            if s == m[k] and not (len(supp) == 1 and eps[k] == 1): rep[k].append((supp, len(supp) % 2))
    return rel, rep
def value(r, rel, rep, two=False):
    pr = lambda supp: np.prod([r[i] for i in supp])
    E0 = sum(pr(s) for s, par in rel if (not two or par == 0))
    return min(r[k] - sum(pr(s) for s, par in rep[k] if (not two or par == 1)) for k in range(len(r))) / (1 + E0)
def optimise(p, H, two=False):
    rel, rep = patterns(p, H); K = len(H)
    f = lambda u: -value(0.5 / (1 + np.exp(-u)), rel, rep, two)
    best = min((minimize(f, np.full(K, u0), method='Nelder-Mead', options=dict(maxiter=4000)) for u0 in (-1.0, 0.0, 1.5, 4.0)), key=lambda r: r.fun)
    r = [Fr(int(round(0.5 / (1 + math.exp(-u)) * 10 ** 6)), 10 ** 6) for u in best.x]
    return float(value(r, rel, rep, two)), len(rel)
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for p in (1009, 10007):
        h = (p - 1) // 2; K = round(1.3 * math.log(p)); res = []
        for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
            H = sorted(rng.choice(np.arange(1, h + 1), K, replace=False).tolist())
            res.append((optimise(p, H)[0], optimise(p, H, True)[0]))
        a = np.array(res)
        print(f"p={p} K={K}: per-rho one-sided median {np.median(a[:, 0]):.3f} (max {a[:, 0].max():.3f}); two-sided median {np.median(a[:, 1]):.3f} (max {a[:, 1].max():.3f})", flush=True)
