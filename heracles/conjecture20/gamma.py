# Exact Gamma_s(p) via vertex enumeration of P_p = {v>=0 : sum_j v_j kappa_j(delta) >= 1, delta in F}
import numpy as np, itertools, math
def kappa(p):
    F = list(range(1, (p - 1) // 2 + 1))
    K = np.array([[1 - math.cos(2 * math.pi * j * d / p) for j in F] for d in F])  # rows delta, cols j
    return F, K
def Gamma(p, s, return_y=False):
    F, K = kappa(p); f = len(F); r = 1 / s
    # constraints: K v >= 1 (f rows), v >= 0 (f rows). vertex: f active among 2f
    A = np.vstack([K, np.eye(f)]); bvec = np.concatenate([np.ones(f), np.zeros(f)])
    best = None
    for act in itertools.combinations(range(2 * f), f):
        M = A[list(act)]
        if abs(np.linalg.det(M)) < 1e-12: continue
        v = np.linalg.solve(M, bvec[list(act)])
        if (v < -1e-12).any() or (K @ v < 1 - 1e-10).any(): continue
        v = np.maximum(v, 0)
        phi = (v ** r).sum()
        if best is None or phi < best[0]: best = (phi, v)
    G = best[0] ** (-s)
    if return_y:
        y = G * best[1]
        return G, y
    return G
if __name__ == "__main__":
    for p in [3, 5, 7, 11]:
        for s in [1, 1.25, 1.5, 2, 2.5, 3]:
            G, y = Gamma(p, s, True)
            print(p, s, round(G, 6), np.round(y, 4))
