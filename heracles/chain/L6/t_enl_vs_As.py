# Check: the flip-enlarged l2-polyball depth is <= the two-sided K-mode depth A_s (single-pattern directions).
# Take the flattest two-sided direction z (from the polar LP at the worst theta), put it on pattern 1 only, evaluate the
# enlarged body's support function H(z) exactly (all rows, all three variants).
import numpy as np
from scipy.optimize import linprog
from tools import depth_polar
rng = np.random.default_rng(5)
for p in (11, 13, 17):
    H = list(range(1, (p - 1) // 2 + 1)); K = len(H); x = np.arange(p)
    As, th = depth_polar(p, H, True, rng, starts=20, iters=400)
    chi = np.exp(-2j * np.pi * 2 * np.outer(H, x) / p); rows = np.c_[chi.real.T, -chi.imag.T]
    r = linprog(-np.r_[np.cos(th), np.sin(th)], A_ub=np.vstack([rows, -rows]), b_ub=np.ones(2 * p), bounds=[(None, None)] * (2 * K), method='highs')
    z1 = r.x[:K] + 1j * r.x[K:]
    zc = np.zeros((3, K), complex); zc[0] = z1
    ze = chi; ab = (x[:, None] + x[None, :]) % p; Hval = -np.inf
    for s1, s2 in ((1, 1), (-1, 1), (1, -1)):
        F = s1 * np.real(zc[0] @ ze)[:, None] + s2 * np.real(zc[1] @ ze)[None, :] + s1 * s2 * np.real(zc[2] @ ze)[ab]
        Hval = max(Hval, F.max())
    print(f"p={p} dense: A_s ~ {As:.4f};  enlarged H(z)/sum|z_k| on that direction = {Hval / np.abs(z1).sum():.4f}  (<= 0.272 fails the proven cost side)")
