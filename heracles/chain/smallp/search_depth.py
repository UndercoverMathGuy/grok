# NUMERICAL: upper estimates of the l2-polyball depth of the flip-enlarged body (explicit bad directions).
import sys, numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
def gens(p, T):
    x = np.arange(p); a, b = np.meshgrid(x, x, indexing='ij'); a = a.ravel(); b = b.ravel()
    T = np.array(T)
    E = lambda u: np.exp(-2j * np.pi * 2 * np.outer(u, T) / p)          # rows x K
    V1, V2, V3 = E(a), E(b), E((a + b) % p)
    G = []
    for s1, s2 in ((1, 1), (-1, 1), (1, -1)):
        G.append(np.concatenate([s1 * V1, s2 * V2, s1 * s2 * V3], axis=1))  # generator rows: (t_{k1}, t_{k2}, t_{k12})
    return np.vstack(G)                                                   # n x 3K complex
def make_obj(p, T):
    G = gens(p, T); K = len(T)
    def obj(v):
        z = v[:3 * K] + 1j * v[3 * K:]
        H = np.real(G @ z).max()                                          # support function, pairing Re sum z t
        n = np.linalg.norm(z.reshape(3, K), axis=0).sum()
        return H / n
    return obj
def search(p, T, starts=200):
    K = len(T); obj = make_obj(p, T); best = (9, None); I = np.arange(K)
    S = []
    for pat in range(3):
        for ph in (np.zeros(K), np.pi * np.ones(K), np.pi * (I % 2), np.pi * I ** 2 / K):
            z = np.zeros((3, K), complex); z[pat] = np.exp(1j * ph); S.append(np.r_[z.real.ravel(), z.imag.ravel()])
    for _ in range(12):     # single-atom-like shapes: equal moduli, random phases
        z = np.exp(2j * np.pi * rng.uniform(size=(3, K))); S.append(np.r_[z.real.ravel(), z.imag.ravel()])
    S += [rng.normal(size=6 * K) for _ in range(starts)]
    for s in S:
        r = minimize(obj, s, method='Powell', options=dict(maxiter=60000, xtol=1e-9, ftol=1e-11))
        if r.fun < best[0]: best = (r.fun, r.x)
    return best
if __name__ == '__main__':
    p = int(sys.argv[1]); T = [int(t) for t in sys.argv[2].split(',')]
    v, x = search(p, T)
    K = len(T); z = (x[:3 * K] + 1j * x[3 * K:]).reshape(3, K)
    print(f"p={p} T={T}: enlarged l2-depth <= {v:.4f}; worst direction moduli per k:", np.round(np.abs(z) / np.linalg.norm(z, axis=0), 3).T.tolist())
