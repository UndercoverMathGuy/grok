# Adversarial minimisation of  R(f) = (min_c depth_c)(min_c' height_c') / V  over mean-zero f1,f2,f3 on Z_p.
# Theorem PBD#: R >= (p-1)/(p+1), equality at the ramp.  Exact objective, stochastic hill climbing with
# shrinking steps from random, spiky and near-ramp starts.
import numpy as np, sys
p = int(sys.argv[1]); reps = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed)
x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); x1 = x1.ravel(); x2 = x2.ravel()
idx3 = [((c - x1 - x2) % p) for c in range(p)]
def R(z):
    f = z.reshape(3, p); f = f - f.mean(1, keepdims=True)
    V = (f**2).mean(1).sum()
    if V < 1e-14: return np.inf
    base = f[0][x1] + f[1][x2]
    m = min(-(base + f[2][ix]).min() for ix in idx3); M = min((base + f[2][ix]).max() for ix in idx3)
    return m*M/V
best = np.inf
ramp = ((-np.arange(p)) % p).astype(float)
for r in range(reps):
    kind = r % 4
    if kind == 0: z = rng.normal(size=3*p)
    elif kind == 1: z = np.tile(ramp, 3) + 0.5*rng.normal(size=3*p)
    elif kind == 2: z = rng.normal(size=3*p)**3
    else: z = np.concatenate([np.roll(ramp*int(rng.integers(1, p)) % p, int(rng.integers(p))) for _ in range(3)]) + rng.normal(size=3*p)
    val = R(z); step = 0.5*np.abs(z).max()
    for it in range(4000):
        cand = z + step*rng.normal(size=3*p)*(rng.random(3*p) < 0.3)
        v = R(cand)
        if v < val: z, val = cand, v
        else: step *= 0.997
        if step < 1e-6: break
    best = min(best, val)
print(f"p={p}: adversarial min of mM/V over {reps} runs = {best:.6f};  (p-1)/(p+1) = {(p-1)/(p+1):.6f};  "
      f"min - bound = {best - (p-1)/(p+1):.2e}")
