"""Exact numbers quoted in the paper's new Section 4 text (plateaus, bottlenecks, ablation).
Reuses the vertex enumeration of check_consequences.py (exact for p <= 23)."""
import numpy as np
from scipy.optimize import brentq
from check_consequences import vertices, fold

def canon(v, p):
    h = len(v); best = None
    for a in range(1, p):
        img = np.zeros(h)
        for j in np.nonzero(v > 1e-12)[0]:
            img[fold(a * (j + 1), p) - 1] = v[j]
        key = tuple(np.round(img, 7)); best = key if best is None or key < best else best
    return best

for p in [13, 23]:
    A, V = vertices(p)
    cost = lambda s, v: (v ** (1 / s)).sum()
    grid = np.linspace(1.0005, 2.5, 3000)
    win = [int(np.argmin([(V[i] ** (1 / s)).sum() for i in range(len(V))])) for s in grid]
    keys = [canon(V[w], p) for w in win]
    print(f'p={p}: switch points in (1, 2.5] (exact root of cost equality between consecutive optimal vertices):')
    for t in range(1, len(grid)):
        if keys[t] != keys[t - 1]:
            va, vb = V[win[t - 1]], V[win[t]]
            sw = brentq(lambda s: cost(s, va) - cost(s, vb), grid[t - 1], grid[t])
            sa = [j + 1 for j in np.nonzero(va > 1e-12)[0]]; sb = [j + 1 for j in np.nonzero(vb > 1e-12)[0]]
            print(f'   s = {sw:.4f}: K {len(sa)} -> {len(sb)}   ({sa} -> {sb})')

print('\nablation at exact maximisers: ratio (actual relative loss)/(1-(1-w)^s), and loss <= 2y/m check; tight offsets vs K')
for s in [1.5, 2.0, 2.5]:
    rat, ok_up, tight_ok = [], True, True
    for p in [7, 11, 13, 17, 19, 23]:
        A, V = vertices(p)
        c = (V ** (1 / s)).sum(1); i0 = c.argmin(); y = V[i0] * c[i0] ** (-s); m = (A @ y).min()
        act = np.nonzero(y > 1e-12)[0]
        tight = int((np.abs(A @ y - m) < 1e-9).sum()); tight_ok &= tight >= len(act)
        for j in act:
            yj = y.copy(); yj[j] = 0; loss = 1 - (A @ yj).min() / m
            w = y[j] ** (1 / s); rat.append(loss / (1 - (1 - w) ** s)); ok_up &= loss <= 2 * y[j] / m + 1e-12
    print(f'  s={s}: loss/lower-bound in [{min(rat):.3f}, {max(rat):.3f}] over {len(rat)} (p, j) pairs; upper bound holds: {ok_up}; tight >= K at all p: {tight_ok}')

print('\nexact end of the dense window s_c(p): largest s at which the all-frequency vertex is optimal')
for p in [7, 11, 13, 17, 19, 23]:
    A, V = vertices(p)
    h = A.shape[0]
    dense = [i for i in range(len(V)) if (V[i] > 1e-12).sum() == h]
    assert len(dense) == 1
    vd = V[dense[0]]
    f = lambda s: (vd ** (1 / s)).sum() - min((V[i] ** (1 / s)).sum() for i in range(len(V)) if i != dense[0])
    grid = np.linspace(1.0005, 2.5, 3000)
    t = next(k for k in range(len(grid)) if f(grid[k]) > 1e-12)
    sc = brentq(f, grid[t - 1], grid[t])
    print(f'  p={p}: s_c = {sc:.6f}  (s_c - 1)*ln p = {(sc - 1) * np.log(p):.3f}')
