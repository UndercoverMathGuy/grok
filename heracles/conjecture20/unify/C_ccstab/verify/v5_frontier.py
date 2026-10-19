# Referee check of claim R at (5,2,2), s = 3/2.
# (1) independent recomputation of H_{2/3}(m) and gamma_1/B_p for the stored SPLIT violators;
# (2) the KM2 two-block frontier;  (3) the caveat: decoupled networks = relabelled scaled copies of ONE single-feature
#     network theta_1 (any theta_1, not only the optimal IF block), and also unions of two independent single-feature
#     components.  Search for one with H_{2/3}(m) <= H(theta_v) and gamma_1/B_p > gamma_1(theta_v)/B_p.
import numpy as np, math
from scipy.optimize import differential_evolution
from vlib import *
p, n, b = 5, 2, 2.0; nu = 3; s = 1.5; r = 2 / 3; B = Bp(p, n)
d = np.load('/home/user/grok/heracles/conjecture20/unify/split/viol_p5_n2_b2.npz')
viol = []
for kk in range(3):
    net = Net(d['U%d' % kk], d['W%d' % kk], 2.0); a = net.masses(); m = a / a.sum()
    viol.append((renyi(m, r), net.gaps().min() / a.sum() / B))
    print('viol%d: H_2/3(m)=%.6f  gamma_1/B=%.6f' % (kk, *viol[-1]))
K = kappa(p)[:2]
def M(yh): return float((K @ yh).min())
ts = np.linspace(0, 0.5, 500001); best = 0
for t in ts:
    yh = np.array([1 - t, t]); H = math.log(3) + renyi(yh, r)
    if H <= viol[0][0]: best = max(best, M(yh))
print('KM2 frontier at cap %.6f: best M = %.6f' % (viol[0][0], best))

cap, target = viol[0]
def build(x, k, mode):
    specs = []
    for l in [1, 2]:
        for i in range(k):
            o = 6 * i if mode == 'copy' else 6 * (i + k * (l - 1))
            amps = np.exp(x[o:o + 3]); phis = x[o + 3:o + 5]; psi = x[o + 5]
            if l == 2: amps = amps * math.exp(x[-1] / 3)               # mass scale factor e^{x[-1]} on copy 2
            specs.append((l, amps, phis, psi))
    return classC(p, n, specs, b)
def obj(x, k, mode):
    net = build(x, k, mode); a = net.masses(); m = a / a.sum(); H = renyi(m, r)
    g1 = net.gaps().min() / a.sum() / B
    return -(g1 - 50 * max(0.0, H - cap))
for mode in ['copy', 'free']:
    for k in [1, 2, 3]:
        dim = 6 * k + 1 if mode == 'copy' else 12 * k + 1
        bounds = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)] * (k if mode == 'copy' else 2 * k) + [(-4, 1)]
        bestv = -np.inf; bx = None
        for seed in range(2):
            res = differential_evolution(obj, bounds, args=(k, mode), seed=seed, maxiter=300, popsize=12, tol=1e-12, polish=True)
            if -res.fun > bestv: bestv = -res.fun; bx = res.x
        net = build(bx, k, mode); a = net.masses(); m = a / a.sum()
        print('%s k=%d: best gamma_1/B = %.6f at H_2/3 = %.6f (cap %.6f, target to beat %.6f)'
              % (mode, k, net.gaps().min() / a.sum() / B, renyi(m, r), cap, target))
