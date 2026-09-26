# End-to-end sanity on actual tables (p = 59): optimal heavy blocks at share 1 - sigma plus light LONE neurons (single atoms,
# maximal junk, phases free) on the frequencies best covering the binding offsets, each at share eps. Actual score / gamma_sym.
import sys, json, numpy as np
sys.path.insert(0, '..')
from common import table, gamma, block_net, gamma_sym, cosvec
from progopt import cover
p = 59; h = 29; rng = np.random.default_rng(5)
hb = json.load(open('heavy_59.json')); G = hb['Gamma_lb']; x0 = np.array(hb['x']); A = cover(p)
f = A @ (x0 ** 1.5 / np.sqrt(3)); bind = np.nonzero(f <= f.min() * (1 + 1e-6))[0]
score = A[bind].min(0); score[x0 > 1e-7] = -1; order = np.argsort(-score)
def net_for(sigma, m, eps):
    y = (x0 * (1 - sigma)) ** 1.5 / np.sqrt(3)          # block amplitudes: block cost 3^{1/3} y^{2/3} = share (atom-cost units)
    net = block_net(p, y)
    for k0 in order[:m]:
        c = (sigma / m) ** 1.5                           # lone atom with atom-cost share sigma/m: |c|^{2/3} = sigma/m
        rho = c ** (1 / 3) * 2 / p ** 0.5 * 1.0          # |ehat(k)| = rho p/2 for rho cos(.) ; product of three = c
        rho = (c / (p / 2) ** 3) ** (1 / 3)
        ph = rng.uniform(0, 2 * np.pi, 3)
        net.append((cosvec(p, k0 + 1, ph[0], rho), cosvec(p, k0 + 1, ph[1], rho), cosvec(p, k0 + 1, ph[2], rho)))
    return net
eps = G ** 2 / 12
worst = 0
for m in (1, 2, 4, 8, 16):
    for trial in range(20):
        sig = min(m * eps, 0.9) * rng.uniform(0.2, 1)
        net = net_for(sig, m, eps)
        worst = max(worst, gamma(p, net) / gamma_sym(p, G))
print(f"p=59, light lone neurons at share <= eps_clean, random phases: max actual score/gamma_sym = {worst:.8f}")
