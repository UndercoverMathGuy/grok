# Referee checks for R9 (slicing lemma): own LP for the guaranteed constants, and brute-force checks at n=3,4.
import numpy as np, itertools
from scipy.optimize import linprog
rng = np.random.default_rng(41)
def subsets(n): return [T for k in range(1, n + 1) for T in itertools.combinations(range(n), k)]
def lp(n, kap):
    Ts = subsets(n); m = len(Ts)       # variables: V_T (m), t ; minimise t
    Aub = []; bub = []
    for i, j in itertools.combinations(range(n), 2):
        row = np.zeros(m + 1)
        for k, T in enumerate(Ts):
            row[k] = kap if (i in T or j in T) else -1.0      # t >= kap*meets - avoids  <=>  kap*meets - avoids - t <= 0
        row[-1] = -1; Aub.append(row); bub.append(0)
    for k in range(m):
        row = np.zeros(m + 1); row[k] = 1; row[-1] = -1; Aub.append(row); bub.append(0)
    res = linprog(np.r_[np.zeros(m), 1], A_ub=np.array(Aub), b_ub=bub, A_eq=[np.r_[np.ones(m), 0]], b_eq=[1], bounds=[(0, None)]*(m + 1), method='highs')
    return res.fun
for kap in [1.0, 5/6, 2/3]:
    print(f"kappa={kap:.3f}: " + "  ".join(f"n={n}: {lp(n, kap):.4f}" for n in range(2, 8)) + f"   kappa/(2+kappa)={kap/(2+kap):.4f}")
def check(n, p, trials):
    Ts = subsets(n); X = np.array(list(itertools.product(range(p), repeat=n)))
    sT = {T: X[:, list(T)].sum(1) % p for T in Ts}
    worst = np.inf; worst_ratio = np.inf
    for t in range(trials):
        D = {}
        for T in Ts:
            kind = rng.integers(4)
            if kind == 0: v = rng.normal(size=p)
            elif kind == 1: v = ((-np.arange(p) + rng.integers(p)) % p).astype(float)*rng.uniform(0.2, 2)
            elif kind == 2: v = np.zeros(p); v[rng.integers(p)] = rng.normal()*3
            else: v = rng.normal(size=p)*rng.uniform(0, 0.1)
            D[T] = v - v.mean()
        V = {T: (D[T]**2).mean() for T in Ts}; E = sum(V.values())
        Z = sum(D[T][sT[T]] for T in Ts); A = -Z.min()
        for d in range(p):
            Zd = sum(D[T][(sT[T] + d) % p] for T in Ts); Ap = Zd.max()
            kap = (p - 1)/(p + 1)
            Cd = {T: (D[T]*D[T][(np.arange(p) + d) % p]).mean() for T in Ts}
            best = max(V.values())
            for i, j in itertools.combinations(range(n), 2):
                bnd = kap*sum(V[T] for T in Ts if (i in T or j in T)) + sum(Cd[T] for T in Ts if not (i in T or j in T))
                best = max(best, bnd)
            worst = min(worst, (A*Ap - best)/E); worst_ratio = min(worst_ratio, A*Ap/E)
    return worst, worst_ratio
for (n, p, tr) in [(3, 3, 300), (3, 5, 200), (3, 7, 60), (4, 5, 20)]:
    w, wr = check(n, p, tr)
    print(f"slicing n={n} p={p}: min (AA' - best slicing bound)/E = {w:.3e} (>=0);  min AA'/E = {wr:.4f};  LP guarantee at kappa_p = {lp(n, (p-1)/(p+1)):.4f}")
