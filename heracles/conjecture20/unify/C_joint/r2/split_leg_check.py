import numpy as np, itertools
from pw_scan import legendre_phases
rng = np.random.default_rng(3)
# (a) Kronecker splitting: groups with independent character modules -> depth adds, and min_x of sums of group functions adds.
worst = 0
for trial in range(100):
    p = int(rng.choice([5, 7, 11, 13])); n = 3; F = (p-1)//2
    X = np.indices((p,)*n).reshape(n, -1).T
    # group 1 uses sets inside {0,1}; group 2 uses set {2}: modules span{e0,e1} and span{e2} independent
    g1 = [(int(rng.integers(1, F+1)), T) for T in [(0,), (1,), (0, 1)] for _ in range(2)]
    g2 = [(int(rng.integers(1, F+1)), (2,)) for _ in range(3)]
    d = int(rng.integers(1, p))
    def dd(fam):
        tot = np.zeros(len(X))
        for (l, T) in fam:
            B = rng.normal()+1j*rng.normal(); xT = X[:, list(T)].sum(1)
            tot += np.real(B*np.exp(2j*np.pi*2*l*xT/p)*(1-np.exp(-2j*np.pi*l*d/p)))
        return tot
    D1 = dd(g1); D2 = dd(g2)
    worst = max(worst, abs((D1+D2).min() - (D1.min() + D2.min())))
print(f"(a) independent modules: max |min(d1+d2) - (min d1 + min d2)| = {worst:.2e}")
# (b) Legendre single-T network, p = 3 mod 4: field-row l_r sums take exactly three values |F|, S_+/2, S_-/2
for p in [7, 11, 19, 23, 43, 227, 347, 499]:
    if p % 4 != 3: continue
    F = (p-1)//2; l = np.arange(1, F+1); w = np.arange(p); r = 2/3
    phi = legendre_phases(p)
    vals = ((1 + np.cos(phi[None, :] + 2*np.pi*np.outer(w, l)/p))**r).sum(1)
    x = np.arange(1, p); qr = set((x**2) % p); chi = np.array([1 if a in qr else -1 for a in x])
    Sp = np.sum((1 + chi*np.sin(2*np.pi*x/p))**r); Sm = np.sum((1 - chi*np.sin(2*np.pi*x/p))**r)
    distinct = np.unique(np.round(vals, 9))
    pred = np.unique(np.round([F, Sp/2, Sm/2], 9))
    print(f"(b) p={p}: distinct field-row sums {distinct}  predicted {pred}  phases in {{-pi/2,+pi/2}}: {np.allclose(np.abs(phi), np.pi/2)}")
