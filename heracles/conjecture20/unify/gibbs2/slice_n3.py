# General-n slicing lemma (class C profiles), checked on exact minima.
#  D_T : Z_p -> R mean zero for T != empty in [n];  Z(x) = sum_T D_T(s_T(x)),  Z^d(x) = sum_T D_T(s_T(x) + d).
#  A = -min Z,  A' = max Z^d,  V_T = Var D_T,  C_T(d) = E_w D_T(w) D_T(w+d),  kap = (p-1)/(p+1).
#  LEMMA (slicing with a 2-dim core {i,j}):  A A' >= kap * sum_{T meets {i,j}} V_T + sum_{T avoids {i,j}} C_T(d)
#  and per-T Bhatia-Davis:                  A A' >= max_T V_T.
#  For n = 3 these give A A' >= kap/(2+kap) * sum_T V_T.
import numpy as np, itertools, sys
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
p = int(sys.argv[1]); n = int(sys.argv[2])
Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]
X = np.indices((p,)*n).reshape(n, -1)
S = [(np.array(T) @ X) % p for T in Ts]
kap = (p - 1)/(p + 1)
def check(D, d):
    Z = sum(D[k][S[k]] for k in range(len(Ts))); Zd = sum(D[k][(S[k] + d) % p] for k in range(len(Ts)))
    A = -Z.min(); Ap = Zd.max()
    V = np.array([(Dk**2).mean() for Dk in D]); C = np.array([(Dk*np.roll(Dk, -d)).mean() for Dk in D])
    bounds = [V.max()]
    for i, j in itertools.combinations(range(n), 2):
        meet = np.array([T[i] or T[j] for T in Ts], bool)
        bounds.append(kap*V[meet].sum() + C[~meet].sum())
    return A*Ap, max(bounds), V.sum()
worst = np.inf; worst_ratio = np.inf; worst_n3 = np.inf
for t in range(int(sys.argv[4]) if len(sys.argv) > 4 else 300):
    kind = t % 4
    D = []
    for k, T in enumerate(Ts):
        if kind == 0: v = rng.normal(size=p)*rng.exponential()
        elif kind == 1: v = (((np.arange(p) + rng.integers(p))*int(rng.integers(1, p))) % p).astype(float)*rng.exponential()
        elif kind == 2: v = rng.normal(size=p)*(rng.random() < 0.4)
        else: v = ((-np.arange(p)) % p).astype(float)*(sum(T) == 1) + 0.1*rng.normal(size=p)
        D.append(v - v.mean())
    d = int(rng.integers(1, p))
    prod, b, E = check(D, d)
    if E < 1e-12: continue
    worst = min(worst, (prod - b)/E); worst_ratio = min(worst_ratio, prod/E)
print(f"p={p} n={n}: min (AA' - best slicing bound)/E = {worst:.3e} (>=0);  min AA'/E seen = {worst_ratio:.4f};  "
      f"n=3 guaranteed kap/(2+kap) = {kap/(2+kap):.4f};  Note Thm loc: 1/(2^n-1) = {1/(2**n-1):.4f}")
