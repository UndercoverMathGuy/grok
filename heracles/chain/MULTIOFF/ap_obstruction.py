# Is there, on a progression {1..L}, a single-pattern junk direction whose PAIR ratio is small at EVERY offset?
# Then every offset weighting lam gives multi-offset depth <= max_D pair ratio (mediant inequality).
import numpy as np, torch
from mo import Body, cov
from scipy.optimize import minimize
p = 1009; rng = np.random.default_rng(2)
Ds = np.arange(1, (p - 1) // 2 + 1)
def per_offset_ratios(T, z):
    J = torch.tensor(np.stack([z, 0 * z, 0 * z]).astype(complex))
    out = []
    for D in Ds:
        b = Body(p, T, [D, p - D], [0.5, 0.5], mode='1')
        out.append(float(b.ratio(J)))
    return np.array(out)
for L in (4, 8, 16, 32):
    T = list(range(1, L + 1)); k = np.arange(L)
    cands = {'chirp': np.exp(1j * np.pi * k ** 2 / L), 'chirp2': np.exp(1j * np.pi * k * (k + 1) / L)}
    # Rudin-Shapiro signs for L a power of 2
    rs = np.array([1.0]); s = np.array([1.0])
    while len(rs) < L: rs, s = np.r_[rs, s], np.r_[rs, -s]
    cands['rudin_shapiro'] = rs[:L].astype(complex)
    res = {n: per_offset_ratios(T, z).max() for n, z in cands.items()}
    best = min(res, key=res.get)
    print(f"L={L:2d}: max over offsets of pair ratio: " + ", ".join(f"{n} {v:.3f}" for n, v in res.items())
          + f"   -> every lam gives depth <= {res[best]:.3f};  1/sqrt(L) = {L ** -0.5:.3f}", flush=True)

print("--- restricted to offsets where the progression covers at least half its maximum (sum_k cov_k(D) >= L/2) ---")
for L in (4, 8, 16, 32):
    T = list(range(1, L + 1)); k = np.arange(L)
    rs = np.array([1.0]); s = np.array([1.0])
    while len(rs) < L: rs, s = np.r_[rs, s], np.r_[rs, -s]
    good = cov(p, np.array(T), Ds).sum(0) >= L / 2
    for n, z in (('chirp', np.exp(1j * np.pi * k ** 2 / L)), ('rudin_shapiro', rs[:L].astype(complex))):
        r = per_offset_ratios(T, z)[good]
        print(f"L={L:2d} {n:13s}: over {good.sum()} well-covered offsets: max {r.max():.3f}  median {np.median(r):.3f}  min {r.min():.3f}"
              f"   (Parseval floor 1/sqrt(2L) = {1 / np.sqrt(2 * L):.3f})", flush=True)
