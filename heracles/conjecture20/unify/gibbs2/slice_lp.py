# Guaranteed joint-detection constant from (per-T Bhatia-Davis) + (slicing lemma with 2-dim cores), worst case C_T = -V_T:
#   c_n(kap) = min_{V >= 0, sum V = 1} max( max_T V_T,  max_{i<j} [ kap * sum_{T meets {i,j}} V_T - sum_{T avoids {i,j}} V_T ] )
# (for n = 2 the core is everything and c_2 = kap).  Compare with the Note's 1/(2^n - 1).
import numpy as np, itertools
from scipy.optimize import linprog
def c_n(n, kap):
    Ts = [T for T in itertools.product([0, 1], repeat=n) if any(T)]; N = len(Ts)
    # variables V_1..V_N, t ; minimise t
    A = []; b = []
    for k in range(N):
        row = np.zeros(N + 1); row[k] = 1; row[-1] = -1; A.append(row); b.append(0)   # V_T <= t
    # max(...) <= t means: for the optimum we need t >= max_T V_T AND t >= every pair bound?  No: the bound is a MAX of
    # certificates, so the adversary must make ALL certificates small:  V_T <= t and pair_ij(V) <= t for all pairs.
    for i, j in itertools.combinations(range(n), 2):
        row = np.zeros(N + 1)
        for k, T in enumerate(Ts):
            row[k] = kap if (T[i] or T[j]) else -1.0
        row[-1] = -1; A.append(row); b.append(0)
    Aeq = [np.r_[np.ones(N), 0]]; beq = [1]
    res = linprog(np.r_[np.zeros(N), 1], A_ub=np.array(A), b_ub=b, A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)]*(N + 1), method='highs')
    return res.fun, res.x[:N], Ts
for kap in [1.0, 5/6, 2/3]:
    for n in [2, 3, 4, 5, 6, 7]:
        val, V, Ts = c_n(n, kap)
        print(f"kap={kap:.3f} n={n}: guaranteed AA'/E >= {val:.4f}   (Note Thm loc: 1/(2^n-1) = {1/(2**n-1):.4f}, gain x{val*(2**n-1):.2f})")
