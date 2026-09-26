# NUMERICAL upper estimates: l2-polyball depth of the D-dependent shift body
#   B'_D = conv{ ( m(s1,k) t_k1, m(s2,k) t_k2, m(s1+s2,k) t_k12 )_k : rows (a,b), shifts (s1,s2) in Z_p^2 },
#   m(s,k) = W(s; th_k) / cov_k.   (plain = s=(0,0); flips = s1 or s2 = 1/2)
import sys, numpy as np
from scipy.optimize import minimize
def gens(p, T, D, shifts):
    x = np.arange(p); a, b = np.meshgrid(x, x, indexing='ij'); a = a.ravel(); b = b.ravel(); T = np.array(T)
    th = 2 * np.pi * T * D / p; cov = 1 - np.cos(th)
    m = lambda s: (np.cos(2 * s * th) - np.cos((1 - 2 * s) * th)) / cov
    E = lambda u: np.exp(-2j * np.pi * 2 * np.outer(u, T) / p)
    V1, V2, V3 = E(a), E(b), E((a + b) % p)
    return np.vstack([np.concatenate([m(s1) * V1, m(s2) * V2, m(s1 + s2) * V3], axis=1) for s1, s2 in shifts])
def search(G, K, rng, starts=150):
    def obj(v):
        z = v[:3 * K] + 1j * v[3 * K:]
        return np.real(G @ z).max() / np.linalg.norm(z.reshape(3, K), axis=0).sum()
    I = np.arange(K); S = []
    for pat in range(3):
        for ph in (np.zeros(K), np.pi * np.ones(K), np.pi * (I % 2), np.pi * I ** 2 / K):
            z = np.zeros((3, K), complex); z[pat] = np.exp(1j * ph); S.append(np.r_[z.real.ravel(), z.imag.ravel()])
    for _ in range(12):
        z = np.exp(2j * np.pi * rng.uniform(size=(3, K))); S.append(np.r_[z.real.ravel(), z.imag.ravel()])
    S += [rng.normal(size=6 * K) for _ in range(starts)]
    best = 9
    for s in S:
        best = min(best, minimize(obj, s, method='Powell', options=dict(maxiter=60000, xtol=1e-9, ftol=1e-11)).fun)
    return best
if __name__ == '__main__':
    p = int(sys.argv[1]); T = [int(t) for t in sys.argv[2].split(',')]; D = int(sys.argv[3]); rng = np.random.default_rng(0)
    half = (p + 1) // 2
    fam = {'plain+flips': [(0, 0), (half, 0), (0, half)], 'all shifts': [(s1, s2) for s1 in range(p) for s2 in range(p)]}
    for name, sh in fam.items():
        print(f"p={p} T={T} D={D} {name:12s}: depth <= {search(gens(p, T, D, sh), len(T), rng):.4f}", flush=True)
