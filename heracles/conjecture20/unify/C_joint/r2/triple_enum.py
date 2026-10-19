# Enumerate all zero-sum ordered triples of signed class-C characters eps*2l*1_T for random families; classify (A)/(B).
import numpy as np, itertools
rng = np.random.default_rng(9); bad = 0; total = 0; typeA = typeB = 0
for trial in range(400):
    p = int(rng.choice([3, 5, 7, 11, 13])); n = int(rng.choice([2, 3, 4])); F = (p-1)//2
    sets = [T for k in range(1, n+1) for T in itertools.combinations(range(n), k)]
    fam = list({(int(rng.integers(1, F+1)), sets[rng.integers(len(sets))]) for _ in range(int(rng.integers(2, 9)))})
    chars = []
    for (l, T) in fam:
        v = np.zeros(n, int); v[list(T)] = 2*l
        chars.append((1, l, T, v % p)); chars.append((-1, l, T, (-v) % p))
    for a, b, c in itertools.product(chars, repeat=3):
        if np.any((a[3]+b[3]+c[3]) % p): continue
        total += 1
        A = (a[2] == b[2] == c[2]) and (a[0]*a[1] + b[0]*b[1] + c[0]*c[1]) % p == 0
        B = False
        if a[1] == b[1] == c[1]:
            for x, y, z in itertools.permutations([a, b, c]):
                if set(x[2]) == set(y[2]) | set(z[2]) and not (set(y[2]) & set(z[2])) and y[0] == z[0] == -x[0]: B = True
        typeA += A; typeB += B; bad += not (A or B)
print(f"zero-sum triples enumerated: {total}; type A: {typeA}; type B: {typeB}; neither: {bad} (must be 0)")
# balanced families: skew at -d equals minus skew at d
worst = 0
for trial in range(60):
    p = int(rng.choice([5, 7, 11])); n = 2; F = (p-1)//2; X = np.indices((p,)*n).reshape(n, -1).T
    T = [(0, 1)]; ls = rng.choice(np.arange(1, F+1), size=min(F, 4), replace=False)
    coef = {(int(l), T[0]): rng.normal()+1j*rng.normal() for l in ls}
    for d in range(1, p):
        def dd(dl):
            tot = np.zeros(len(X))
            for (l, TT), Bc in coef.items():
                tot += np.real(Bc*np.exp(2j*np.pi*2*l*X[:, list(TT)].sum(1)/p)*(1-np.exp(-2j*np.pi*l*dl/p)))
            return tot
        a, b = dd(d), dd(p-d); worst = max(worst, abs(np.mean(a**3) + np.mean(b**3))/np.mean(a**2)**1.5)
print(f"balanced (common-set) families: max |E d_d^3 + E d_-d^3| / sigma^3 = {worst:.2e} (must be 0)")
