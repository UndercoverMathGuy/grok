import numpy as np, math
from sympy import primitive_root, legendre_symbol, divisors

def fold(ls, p):
    """map residues to frequencies in 1..F, returning (index, conj) with conj=-1 if l was replaced by p-l."""
    F = (p - 1) // 2; out = {}
    for l in ls:
        l %= p
        if l == 0: continue
        if l <= F: out.setdefault(l - 1, 1)
        else: out.setdefault(p - l - 1, -1)
    return out

def mk(p, idxmap, ph=None, a=None):
    F = (p - 1) // 2; A = np.zeros(F); TH = np.zeros(F)
    S = np.array(sorted(idxmap.keys()))
    A[S] = 1.0 if a is None else a
    if ph is not None:
        for k, (i, sg) in enumerate(idxmap.items()): TH[i] = sg * ph[k]
    return S, A, TH

def seeds(p, rng, ystar=None, nrand=2):
    F = (p - 1) // 2; L = math.log(p); out = []
    # programme support
    if ystar is not None:
        S, y = ystar
        S = np.array(S) - 1; A = np.zeros(F); A[S] = y
        for t in range(nrand): out.append(('ystar', S, A.copy(), rng.uniform(0, 2 * np.pi, F)))
        for k in [int(L), int(3 * L)]:
            ex = rng.choice(np.setdiff1d(np.arange(F), S), min(k, F - len(S)), replace=False)
            S2 = np.sort(np.concatenate([S, ex])); A2 = A.copy(); A2[ex] = A[S].mean() * 0.5
            out.append((f'ystar+{k}', S2, A2, rng.uniform(0, 2 * np.pi, F)))
    # full support, Legendre phases (flat +-1 profile) and random
    chi = np.array([int(legendre_symbol(l, p)) for l in range(1, F + 1)])
    th = np.where(chi > 0, 0.0, np.pi) if p % 4 == 1 else -chi * np.pi / 2
    out.append(('full-legendre', np.arange(F), np.ones(F), th))
    out.append(('full-rand', np.arange(F), np.ones(F), rng.uniform(0, 2 * np.pi, F)))
    # random sets of log size
    for mult in [1.5, 2.5, 4, 6]:
        k = min(F, max(3, int(round(mult * L))))
        for t in range(nrand):
            S = np.sort(rng.choice(F, k, replace=False)); A = np.zeros(F); A[S] = 1
            out.append((f'rand{mult}L', S, A, rng.uniform(0, 2 * np.pi, F)))
    # arithmetic progressions with chirp phases (dilated)
    for mult in [2, 4, 8]:
        K = min(F, int(round(mult * L)))
        for t in range(nrand):
            d = int(rng.integers(1, p)); l0 = int(rng.integers(1, p))
            ls = [(l0 + k * d) % p for k in range(K)]
            im = fold(ls, p)
            if len(im) < K: continue
            ph = np.array([math.pi * k * k / K for k in range(K)])
            S, A, TH = mk(p, im, ph); out.append((f'AP{mult}L-chirp', S, A, TH))
    # Bohr sets of rank 2,3
    for rank in [2, 3]:
        for t in range(nrand):
            al = rng.integers(1, p, rank)
            target = int(round(4 * L)); eps = 0.5
            for _ in range(60):
                ls = [l for l in range(1, F + 1) if all(abs(((x * l) % p) / p - round(((x * l) % p) / p)) < eps for x in al)]
                if len(ls) > 1.3 * target: eps *= 0.9
                elif len(ls) < 0.7 * target: eps *= 1.08
                else: break
            if len(ls) < 3: continue
            im = fold(ls, p)
            S, A, TH = mk(p, im); TH[S] = rng.uniform(0, 2 * np.pi, len(S))
            out.append((f'Bohr{rank}', S, A, TH))
    # multiplicative subgroups / cosets
    g = primitive_root(p)
    ds = [d for d in divisors(p - 1) if 0.8 * L <= d <= 8 * L]
    for K in ds[:4]:
        h = pow(g, (p - 1) // K, p)
        for t in range(nrand):
            co = int(rng.integers(1, p))
            ls = [(co * pow(h, j, p)) % p for j in range(K)]
            im = fold(ls, p)
            m = int(rng.integers(0, K))
            ph = np.array([2 * math.pi * m * j / K for j in range(len(im))])
            S, A, TH = mk(p, im, ph); out.append((f'coset{K}', S, A, TH))
    return out
