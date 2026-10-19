# Exact programme by vertex enumeration of the coverage polytope  P_p = {v >= 0 : sum_j v_j kappa_j(delta) >= 1, delta = 1..F}.
# Gamma^{(S)}_s = max over vertices v with spt(v) subset of S of (sum_j v_j^{1/s})^{-s}  (concave minimisation -> vertex).
# Recomputes transition points and the support gap delta_min(s) exactly (up to float roundoff), for p <= 13.
import sys, itertools, numpy as np
from lib import kappa_mat

def vertices(p):
    F = (p - 1) // 2; K = kappa_mat(p)[:F]                    # rows delta = 1..F, cols j
    A = np.vstack([K, np.eye(F)]); bvec = np.concatenate([np.ones(F), np.zeros(F)])   # A v >= b
    V = []
    for act in itertools.combinations(range(2 * F), F):
        M = A[list(act)]
        if abs(np.linalg.det(M)) < 1e-12: continue
        v = np.linalg.solve(M, bvec[list(act)])
        if np.all(A @ v >= bvec - 1e-10):
            v[np.abs(v) < 1e-12] = 0
            if not any(np.allclose(v, w, atol=1e-10) for w in V): V.append(v)
    return V

def gamma_S(V, S, s):
    r = 1 / s; best = 0.0
    for v in V:
        if all((v[j - 1] == 0) or (j in S) for j in range(1, len(v) + 1)):
            best = max(best, np.sum(v[v > 0] ** r) ** (-s))
    return best

if __name__ == "__main__":
    p = int(sys.argv[1]); grid = [float(x) for x in sys.argv[2].split(",")]; F = (p - 1) // 2
    V = vertices(p); print(f"p={p}: {len(V)} vertices", flush=True)
    allS = [S for k in range(1, F + 1) for S in itertools.combinations(range(1, F + 1), k)]
    for s in grid:
        vals = {S: gamma_S(V, S, s) for S in allS}; G = vals[tuple(range(1, F + 1))]
        # optimal supports = supports of optimal vertices
        r = 1 / s
        opt = sorted({tuple(j + 1 for j in np.nonzero(v)[0]) for v in V if np.sum(v[v > 0] ** r) ** (-s) >= G * (1 - 1e-9)})
        nonc = [S for S in allS if not any(set(T) <= set(S) for T in opt)]
        worst = max(nonc, key=lambda S: vals[S]) if nonc else None
        dmin = 1 - vals[worst] / G if worst else float('inf')
        print(f"p={p} s={s:.3f}: Gamma={G:.6f} K*={sorted(set(len(T) for T in opt))} delta_min={dmin:.4f} (S={worst})", flush=True)
