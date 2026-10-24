# Restricted programme Gamma_s^{(S)}(p) by vertex enumeration; range of s where the count lower bound is exact.
import numpy as np, itertools, math
def kappa(p):
    F = list(range(1, (p - 1) // 2 + 1))
    return F, np.array([[1 - math.cos(2 * math.pi * j * d / p) for j in F] for d in F])
def gamma_S(p, s, S):
    F, K = kappa(p); idx = [F.index(j) for j in S]; K = K[:, idx]; f = len(idx); r = 1 / s
    A = np.vstack([K, np.eye(f)]); b = np.concatenate([np.ones(K.shape[0]), np.zeros(f)])
    best = None
    for act in itertools.combinations(range(A.shape[0]), f):
        M = A[list(act)]
        if abs(np.linalg.det(M)) < 1e-12: continue
        v = np.linalg.solve(M, b[list(act)])
        if (v < -1e-12).any() or (K @ v < 1 - 1e-10).any(): continue
        phi = (np.maximum(v, 0) ** r).sum()
        if best is None or phi < best: best = phi
    return best ** (-s)
def s1(p, nu):
    F, _ = kappa(p); full = F; minus = F[:-1]
    lo, hi = 1.0, 3.0
    for _ in range(50):
        mid = (lo + hi) / 2
        ok = nu ** (mid - 1) * gamma_S(p, mid, minus) < gamma_S(p, mid, full)
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo
for p in [5, 7, 11, 13]:
    F, _ = kappa(p)
    for nu in [3, 4, 5]:
        s = s1(p, nu); S_II = 1 + 1 / (len(F) * (2 * nu + 1) - 2)
        print("p=%2d nu=%d: count = F proved for s <= %.3f (b >= %.3f);  Part II interval s <= %.3f" % (p, nu, s, nu / s, S_II))
