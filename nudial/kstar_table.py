"""M_K table + K*(nu) + equal-vs-free-amplitude check (reconstruction of scratch kstar.py/ktable.py).

M_K(p) = max_{|S|=K} min_{Delta!=0} sum_{k in S} (1 - cos 2pi k Delta / p).
Exhaustive over subsets (WLOG 1 in S: the unit group acts transitively on
frequencies and permutes Delta) while C(nf-1,K-1) <= 4e6, else local search.
unequal_mass: softmin ascent over amplitudes on the fixed support, sphere
constraint sum n_k^2 = 1, objective min_D sum_k n_k^nu g_k(D).
Usage: uv run python nudial/kstar_table.py 113 [211 ...]
"""
import numpy as np, itertools, json, sys
rng = np.random.default_rng(0)

def G(p):
    k = np.arange(1, (p-1)//2 + 1); d = np.arange(1, p)
    return 1 - np.cos(2*np.pi*np.outer(k, d)/p)

def best_subset(p, K, exhaustive_limit=4_000_000, restarts=200):
    g = G(p); nf = g.shape[0]; others = np.arange(1, nf)
    if K == 1: return float(g[0].min()), (1,), 'exact'
    from math import comb
    if comb(nf-1, K-1) <= exhaustive_limit:
        best, bestS = -1, None
        it = itertools.combinations(others, K-1)
        while True:
            block = list(itertools.islice(it, 200_000))
            if not block: break
            idx = np.array(block)
            m = (g[0][None, :] + g[idx].sum(1)).min(1)
            j = m.argmax()
            if m[j] > best: best, bestS = m[j], (1,)+tuple(int(x)+1 for x in idx[j])
        return float(best), bestS, 'exhaustive'
    best, bestS = -1, None
    for r in range(restarts):
        S = set([0]) | set(rng.choice(others, K-1, replace=False).tolist())
        cur = g[list(S)].sum(0).min(); improved = True
        while improved:
            improved = False
            for out in list(S):
                if out == 0: continue
                base = g[[x for x in S if x != out]].sum(0)
                cand = (base[None, :] + g).min(1); cand[list(S)] = -1
                j = int(cand.argmax())
                if cand[j] > cur + 1e-12:
                    S.remove(out); S.add(j); cur = cand[j]; improved = True; break
        if cur > best: best, bestS = float(cur), tuple(sorted(x+1 for x in S))
    return best, bestS, 'local'

def unequal_mass(p, S, L, steps=3000):
    g = G(p)[[s-1 for s in S]]; K = len(S)
    n = np.ones(K)/np.sqrt(K); lr, beta = 0.02, 60.0
    best = ((n**L) @ g).min()
    for t in range(steps):
        f = (n**L) @ g
        w = np.exp(-beta*(f - f.min())); w /= w.sum()
        grad = L * n**(L-1) * (g @ w)
        grad -= n * (grad @ n)
        n = np.abs(n + lr*grad); n /= np.linalg.norm(n)
        best = max(best, ((n**L) @ g).min())
    return float(best)

if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:] or ['113']]:
        rows = {}
        for K in range(1, 13):
            M, S, how = best_subset(p, K)
            rows[K] = (M, S, how)
            print(f"p={p} K={K:2d} M_K={M:.4f} c={K-M:.3f} {how} S={S}", flush=True)
        for nu in (2, 3, 4, 5, 6):
            mg = {K: K**(-nu/2)*rows[K][0] for K in rows}
            mu = {K: unequal_mass(p, rows[K][1], nu) for K in rows}
            Kd = (p-1)//2; dense = Kd**(-nu/2)*(p/2)
            ks = max(mg, key=mg.get)
            print(f"p={p} nu={nu}: K*={ks} margin_eq={mg[ks]:.4g} (free-amp argmax {max(mu,key=mu.get)}) dense={dense:.3g} sparse/dense={mg[ks]/dense:.1f}x")
