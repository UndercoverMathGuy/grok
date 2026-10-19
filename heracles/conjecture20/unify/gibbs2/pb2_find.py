# Locate the PB2 violations in the test families of gibbs/check_planebd.py.
# PB2 (conjecture): (-min_{Pi_c} F)(max_{Pi_c'} F) >= V for all planes c,c'.
import numpy as np
rng = np.random.default_rng(11)
def planes(f, p):
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    return {c: f[0][x1] + f[1][x2] + f[2][(c - x1 - x2) % p] for c in range(p)}
def gen(p, kind):
    fs = []
    for i in range(3):
        if kind == 'gauss': v = rng.normal(size=p)
        elif kind == 'spike': v = np.zeros(p); v[rng.integers(p)] = rng.normal()*p; v += 0.05*rng.normal(size=p)
        elif kind == 'twopoint': v = np.zeros(p); a, b = rng.choice(p, 2, replace=False); v[a] = 1; v[b] = -1; v *= rng.exponential()
        elif kind == 'ramp': d = int(rng.integers(1, p)); v = (np.arange(p) * pow(d, -1, p) % p).astype(float)
        elif kind == 'line': v = rng.normal(size=p) if i == 2 else np.zeros(p)
        elif kind == 'mixed': v = rng.normal(size=p)**3
        fs.append(v - v.mean())
    return fs
worst = {}
for p in [5, 7, 11, 13]:
    for kind in ['gauss', 'spike', 'twopoint', 'ramp', 'line', 'mixed']:
        for t in range(150):
            f = gen(p, kind); V = sum((g**2).mean() for g in f)
            if V < 1e-12: continue
            P = planes(f, p)
            mins = np.array([-P[c].min() for c in range(p)]); maxs = np.array([P[c].max() for c in range(p)])
            r = mins.min()*maxs.min()/V
            key = (p, kind)
            if key not in worst or r < worst[key][0]: worst[key] = (r, f, mins, maxs, V)
for k, v in sorted(worst.items()):
    print(k, f"min ratio {v[0]:.4f}")
k = min(worst, key=lambda k: worst[k][0]); r, f, mins, maxs, V = worst[k]
np.set_printoptions(precision=4, suppress=True)
print("worst", k, r); [print(" f%d" % i, f[i]) for i in range(3)]
print(" depths by plane", mins); print(" heights by plane", maxs); print(" V", V, [ (g**2).mean() for g in f])
