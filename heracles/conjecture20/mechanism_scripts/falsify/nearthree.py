# Near b=3 (s = 1+eps): dense-capable programme (reweighted LP on all of F from several starts, + coset supports)
# and the realisable all-pair network (two aligned neurons per feature, sharp payment) on structured supports:
#   full F with Legendre phases, multiplicative cosets (index k) with real-character phases, QR half.
# Ratio of an all-pair configuration (v = PAIR*a, PAIR=(3/2)^{s-1}) = model ratio at c_sharp, exact for networks (task 3).
import numpy as np, sys, math, json
sys.path.insert(0, '..')
from prog2 import kappa, rlp
from sympy import primitive_root, legendre_symbol
from relmodel import Model, lp_polish

def programme(p, s, extra_supports=(), starts=6, seed=0):
    r = 1 / s; K = kappa(p); F = K.shape[1]; rng = np.random.default_rng(seed); best = (np.inf, None)
    def val(y): yy = y[y > 1e-12]; return np.sum(yy ** r)
    inits = [np.ones(F)] + [rng.exponential(1, F) for _ in range(starts)]
    for S in extra_supports:
        w = np.full(F, 1e3); w[np.array(S)] = 1.0; inits.append(w)
    for w in inits:
        y = rlp(K, r, w)
        if y is None: continue
        if val(y) < best[0]: best = (val(y), y.copy())
    Q, y = best; y = y * (y > 1e-12) / Q ** s
    return Q ** (-s), y

def coset_sets(p, k):
    g = primitive_root(p); F = (p - 1) // 2; out = []
    if (p - 1) % k: return out
    d = (p - 1) // k; h = pow(g, k, p)       # subgroup H of order d (index k)
    for co in range(k):
        gco = pow(g, co, p); ls = [(gco * pow(h, j, p)) % p for j in range(d)]
        idx = {}; ph = {}
        for j, l in enumerate(ls):
            i = (l if l <= F else p - l) - 1; sg = 1 if l <= F else -1
            if i in idx: continue
            idx[i] = (j, sg)
        out.append((co, idx))
    return out

if __name__ == '__main__':
    s = float(sys.argv[1]); ps = [int(a) for a in sys.argv[2].split(',')]
    CS = 1 - (2 / 3) ** (s - 1); PAIR = 1.5 ** (s - 1)
    for p in ps:
        F = (p - 1) // 2
        cos_sup = []
        for k in [2, 4, 6, 8, 10, 12, 16, 20]:
            for co, idx in coset_sets(p, k)[:2]: cos_sup.append(sorted(idx.keys()))
        G, y = programme(p, s, cos_sup)
        K = int(np.sum(y > 0)); M = Model(p, s, CS, G)
        res = []
        chi = np.array([int(legendre_symbol(l, p)) for l in range(1, F + 1)])
        thL = np.where(chi > 0, 0.0, np.pi) if p % 4 == 1 else -chi * np.pi / 2
        a = np.ones(F); res.append(('full-Legendre', M.ratio(a, PAIR * a, thL, CS), F))
        best = (-9, None)
        for k in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30]:
            for co, idx in coset_sets(p, k)[:2]:
                S = np.array(sorted(idx.keys())); a = np.zeros(F); a[S] = 1
                d = (p - 1) // k
                for m in sorted(set(list(range(min(d, 10))) + [d // 2, d // 4, d // 3])):   # character phases on H
                    th = np.zeros(F)
                    for i, (j, sg) in idx.items(): th[i] = sg * 2 * np.pi * m * j / d
                    R = M.ratio(a, PAIR * a, th, CS)
                    if R > best[0]: best = (R, (k, co, m, len(S), a.copy(), th.copy()))
        R, (k, co, m, n, a, th) = best
        R2, a2, v2 = lp_polish(M, a, PAIR * a, th, fixv=PAIR, c=CS, rounds=6)
        print(f's={s} p={p} Gamma={G:.5f} K*={K}/{F}  full-Legendre pairs: {res[0][1]:.4f}   best coset(k={k},co={co},m={m},|S|={n}) pairs: {R:.4f} -> LP-reweighted {R2:.4f}'
              f'   [c_sharp={CS:.4f}; full-F needs eps>4.93/sqrt(p)={4.93 / math.sqrt(p):.3f}]', flush=True)
