# Calibration for the additive-combinatorics pieces: two-sided (flip) K-mode depth of doubled sets 2H in Z_p,
# for arithmetic progressions vs random sets vs distinct-sums sets. Search = UPPER estimates (explicit flat directions).
import itertools, numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(11)
def two_sided(p, S, starts=12):
    K = len(S); ze = np.exp(-2j * np.pi * 2 * np.outer(S, np.arange(p)) / p)
    f = lambda v: np.abs(np.real((v[:K] + 1j * v[K:]) @ ze)).max() / np.abs(v[:K] + 1j * v[K:]).sum()
    st = [np.r_[np.cos(np.pi * np.arange(K) ** 2 / K), np.sin(np.pi * np.arange(K) ** 2 / K)],   # chirp (Rudin-Shapiro-like flat)
          np.r_[np.ones(K), np.zeros(K)]] + [rng.normal(size=2 * K) for _ in range(starts)]
    return min(minimize(f, s, method='Powell', options=dict(maxiter=20000, xtol=1e-9, ftol=1e-11)).fun for s in st)
def distinct(p, S):
    s = [sum(e * 2 * k for e, k in zip(eps, S)) % p for eps in itertools.product((-1, 0, 1), repeat=len(S))]
    return len(set(s)) == len(s)
p = 1009
print("APs {1..L} at p=1009 (two-sided depth, upper est.):")
for L in (2, 3, 4, 6, 8, 10, 12):
    print(f"  L={L:2d}: {two_sided(p, list(range(1, L + 1))):.3f}   (1/sqrt(L) = {L ** -0.5:.3f})", flush=True)
print("random sets at p=1009:")
for K in (4, 6, 9, 12):
    vals = []; nd = 0
    for _ in range(6):
        S = sorted(rng.choice(np.arange(1, (p - 1) // 2 + 1), K, replace=False).tolist())
        vals.append(two_sided(p, S, starts=6)); nd += distinct(p, S) if K <= 9 else 0
    print(f"  K={K:2d}: min {min(vals):.3f} median {np.median(vals):.3f}   distinct-sums {nd}/6" + ("" if K <= 9 else " (not checked)"), flush=True)
