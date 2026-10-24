"""Sigma-orbits of optimised mixed networks at p >= 5 (compare with the p = 3 isolation, p3_orbit.out).

For local optima found from random mixed starts (full tables), compute
  * gamma(theta), gamma(theta_split), and for the global flip group Sigma = {+-1}^{F x n} every orbit margin;
  * a sample of per-neuron flips;
  * the mixed part g_R on the binding offsets (max |g_R| relative to the gap scale) -- for IF-valued optima it must
    vanish (Corollary 'mixed modes cannot help an interference-free core').
usage: python orbit_p5.py p n b width seeds
"""
import sys, itertools, json
import numpy as np, torch
from netopt import *
import lib

p, n, b, m, seeds = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
nu = n + 1; s = nu / b
Gam, ystar = programme(p, s)
gIF = gamma_IF(p, n, b, Gam)
model = Full(p, n)
F = list(range(1, (p - 1) // 2 + 1))


def gam(U, W):
    h = lib.margin(U, W); P = lib.cost(U, W, b)
    return h / P ** s / gIF


rng = np.random.default_rng(11 + p + 100 * n)
out = []
for sd in range(seeds):
    U0 = rng.normal(size=(m, n, p)); W0 = rng.normal(size=(m, p))
    U, W, h, P = optimise(model, U0, W0, b)
    g = gam(U, W)
    Us, Ws = lib.split(U, W)
    gs = lib.margin(Us, Ws) / lib.cost(Us, Ws, b) ** s / gIF
    gsP = lib.margin(Us, Ws) / P ** s / gIF
    orb = []
    for sg in itertools.product([1, -1], repeat=len(F) * n):
        S = np.array(sg).reshape(len(F), n)
        U2, W2 = lib.flip(U, W, S)
        orb.append(gam(U2, W2))
    orb = np.array(orb)
    pn = []
    for k in range(200):
        Ss = [rng.choice([1, -1], size=(len(F), n)) for i in range(m)]
        U2, W2 = lib.flip_neuronwise(U, W, Ss)
        pn.append(gam(U2, W2))
    pn = np.array(pn)
    # mixed part on binding offsets
    f = lib.table(U, W); fs = lib.table(Us, Ws)
    G = lib.gaps_from_table(f, p, n); Gs = lib.gaps_from_table(fs, p, n)
    GR = G - Gs
    scale = np.abs(G[:, 1:]).mean()
    out.append(dict(seed=sd, gamma=g, gamma_split=gs, gamma_split_sameP=gsP, orbit_min=float(orb.min()),
                    orbit_mean=float(orb.mean()), orbit_sorted=sorted(orb.tolist(), reverse=True)[:6],
                    pn_mean=float(pn.mean()), pn_max=float(pn.max()), pn_frac_ge_99=float((pn >= 0.99 * g).mean()),
                    mixed_rel=float(np.abs(GR[:, 1:]).max() / scale)))
    o = out[-1]
    print(f"sd={sd} gamma={g:+.4f} split={gs:+.4f} (split at cost P: {gsP:+.4f}) | global orbit: top {np.round(o['orbit_sorted'], 4)}, "
          f"mean {o['orbit_mean']:+.4f}, min {o['orbit_min']:+.4f} | per-neuron flips mean {o['pn_mean']:+.4f} max {o['pn_max']:+.4f} "
          f"frac>=0.99gamma {o['pn_frac_ge_99']:.2f} | max|g_R|/mean|g| = {o['mixed_rel']:.2e}", flush=True)
json.dump(out, open(f'orbit_p{p}_n{n}_b{b}_m{m}.json', 'w'), indent=1)
