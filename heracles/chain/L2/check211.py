# Is the p = 211 "violation" at eps_clean real, or is Gamma*_lb(211) too low?  Re-optimise the heavy-only programme
# seeded from the light-augmented configuration's support.
import json, numpy as np
from progopt import solve, cover
rng = np.random.default_rng(7); p = 211; h = 105; A = cover(p)
hb = json.load(open('heavy_211.json')); G = hb['Gamma_lb']; x0 = np.array(hb['x'])
eps = G ** 2 / 12
f = A @ (x0 ** 1.5 / np.sqrt(3)); bind = np.nonzero(f <= f.min() * (1 + 1e-6))[0]
score = A[bind].min(0); score[x0 > 1e-7] = -1; order = np.argsort(-score)
sd = []
for m in (1, 2, 4, 8, 16, 32, 64, h):
    u = np.zeros(h); u[order[:m]] = eps
    if u.sum() < 1: sd.append(np.r_[x0 * (1 - u.sum()), u])
v, z = solve(p, eps, 1.0, 8, rng, x0s=sd, maxiter=400)
t = z[:h] + z[h:]
print("light-augmented ratio", v / (G / np.sqrt(3)))
allheavy = (A @ (t ** 1.5 / np.sqrt(3))).min() / (G / np.sqrt(3))
print("same shares, all at block rate:", allheavy)
seeds = [np.r_[t, np.zeros(h)]] + [np.r_[t * rng.uniform(0.8, 1.2, h), np.zeros(h)] for _ in range(5)]
seeds = [s / s.sum() for s in seeds]
v2, z2 = solve(p, 0.0, 1.0, 10, rng, x0s=seeds, maxiter=800)
print("re-optimised heavy-only Gamma_lb:", v2 * np.sqrt(3), "old", G, "support size", int((z2[:h] > 1e-7).sum()))
if v2 * np.sqrt(3) > G:
    json.dump(dict(Gamma_lb=float(v2 * np.sqrt(3)), x=z2[:h].tolist()), open('heavy_211.json', 'w')); print("updated heavy_211.json")
