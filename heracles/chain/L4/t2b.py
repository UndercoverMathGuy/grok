# Depth side vs coverage side. For each set H: two-sided depth (search = UPPER estimate, with the flat direction z found),
# global coverage defect  m_glob = min_D mean_k cov_k(D)  (1 = typical),
# local defect of the flat direction  m_loc = min_D sum |z_k| cov_k(D) / sum |z_k|,
# Rudin-Shapiro bound sqrt(2/L) for progressions of length L = 2^m.
import numpy as np, math
from scipy.optimize import minimize
from l4lib import *
rng = np.random.default_rng(2); p = 1009

def depth_and_dir(H, starts=10):
    K = len(H); ze = np.exp(-2j * np.pi * 2 * np.outer(H, np.arange(p)) / p)
    f = lambda v: np.abs(np.real((v[:K] + 1j * v[K:]) @ ze)).max() / np.abs(v[:K] + 1j * v[K:]).sum()
    st = [np.r_[np.cos(np.pi * np.arange(K) ** 2 / K), np.sin(np.pi * np.arange(K) ** 2 / K)], np.r_[np.ones(K), np.zeros(K)]]
    st += [rng.normal(size=2 * K) for _ in range(starts)]
    best = (9, None)
    for s in st:
        r = minimize(f, s, method='Powell', options=dict(maxiter=30000, xtol=1e-9, ftol=1e-11))
        if r.fun < best[0]: best = (r.fun, r.x)
    z = best[1][:K] + 1j * best[1][K:]
    return best[0], np.abs(z)

def row(name, H):
    H = sorted(set(min(k % p, p - k % p) for k in H if k % p))
    d, az = depth_and_dir(H)
    C = cov_matrix(p, H)[1:]
    mg = (C.mean(1)).min(); ml = (C @ az / az.sum()).min()
    print(f"{name:34s} K={len(H):2d}  depth<= {d:.3f}  global defect {mg:.3f}  flat-direction local defect {ml:.3f}", flush=True)

def ap(s, L): return [(j * s) % p for j in range(1, L + 1)]
pass
0 and print(f"   (Rudin-Shapiro bound sqrt(2/L): L=8 {math.sqrt(2/8):.3f}, L=16 {math.sqrt(2/16):.3f})")
g = 11  # 11 is a primitive root mod 1009? use order check below
def subgroup(n):
    for x in range(2, p):
        if pow(x, n, p) == 1 and all(pow(x, n // q, p) != 1 for q in (2, 3, 5, 7) if n % q == 0):
            return [pow(x, j, p) for j in range(n)]
for n in (12, 14, 16, 18, 24):
    row(f"multiplicative subgroup size {n}", subgroup(n))
row("2-D GAP 4x4 (generic)", [(i * 37 + j * 411) % p for i in range(1, 5) for j in range(0, 4)])
row("2-D GAP 3x5 (generic)", [(i * 37 + j * 411) % p for i in range(1, 4) for j in range(0, 5)])
row("geometric {3^i}", [pow(3, i, p) for i in range(12)])
for K in (8, 12, 16):
    row(f"random K={K}", [int(x) for x in rng.integers(1, p, K)])
