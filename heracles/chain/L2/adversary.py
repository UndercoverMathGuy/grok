# Adversarial light-junk search: max V (relaxed programme, see progopt.py) vs the heavy-only value Gamma*/sqrt3.
# Seeds: best heavy spectrum scaled by (1 - sigma) + light shares eps on the m frequencies with the largest coverage at
# the heavy spectrum's binding offsets (min over binding offsets), plus random starts.
import sys, json, numpy as np
from progopt import solve, cover
from lemma_eps import eps_ref
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
p = int(sys.argv[1]); h = (p - 1) // 2
import os
if os.path.exists(f'heavy_{p}.json'):
    hb = json.load(open(f'heavy_{p}.json'))
else:
    known = {59: [1, 7, 11, 23, 27], 113: [1, 6, 16, 31, 35, 44, 55]}
    sd = []
    for _ in range(6):
        xx = np.zeros(h)
        if p in known: xx[np.array(known[p]) - 1] = rng.dirichlet(np.ones(len(known[p])) * 20)
        else: xx = rng.dirichlet(np.ones(h) * 0.3)
        sd.append(np.r_[xx, np.zeros(h)])
    v0, z0 = solve(p, 0.0, 1.0, 40 if p not in known else 4, rng, x0s=sd)
    hb = dict(Gamma_lb=float(v0 * np.sqrt(3)), x=z0[:h].tolist()); json.dump(hb, open(f'heavy_{p}.json', 'w'))
G = hb['Gamma_lb']; x0 = np.array(hb['x']); print('Gamma_lb', G, flush=True)
A = cover(p); f = A @ (x0 ** 1.5 / np.sqrt(3)); bind = np.nonzero(f <= f.min() * (1 + 1e-6))[0]
score = A[bind].min(0); score[x0 > 1e-7] = -1                      # light candidates: good at every binding offset
order = np.argsort(-score)
def seeds(eps):
    out = []
    for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, h):
        m = min(m, h)
        if m * eps >= 1: continue
        u = np.zeros(h); u[order[:m]] = eps
        out.append(np.r_[x0 * (1 - m * eps), u])
    return out
res = []
for R in (1.0, 25 / 16, 4.0):
    e_c = G ** 2 / (12 * R ** 2); e_r = eps_ref(p, G, R)
    for mult in (1, 1.5, 2, 3, 4, 6, 8, 12, 16):
        eps = e_c * mult
        if eps >= 1: continue
        v, z = solve(p, eps, R, 12, rng, x0s=seeds(eps), maxiter=400)
        ratio = v / (G / np.sqrt(3))
        nl = int((z[h:] > 1e-6).sum()); sl = float(z[h:].sum())
        res.append(dict(p=p, R=R, eps=eps, eps_over_clean=mult, eps_over_ref=eps / e_r, ratio=ratio, n_light=nl, sigma_light=sl))
        print(f"p={p} R={R:.3f} eps={eps:.5f} (x{mult} clean, x{eps / e_r:.2f} ref): max V/(Gamma*/sqrt3) = {ratio:.6f}  light: {nl} freqs, share {sl:.3f}", flush=True)
json.dump(res, open(f'adversary_{p}.json', 'w'), indent=1)
