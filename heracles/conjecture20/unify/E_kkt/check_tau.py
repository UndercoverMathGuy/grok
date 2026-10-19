# Check of R1(e),(f) on the stored local maximisers: for a KKT measure q (NNLS) with delta-marginal mu and x-marginal q_X:
#   nu^{nu/2} B_p max_l |K~_l(mu)| <= tau(q) <= sqrt(2) min(1, ||q_X||_2),   t_i <= tau(q),
#   t_i = nu^{nu/2} gamma w_i^{1-s}  (critical-value law),   w_i >= (nu^{nu/2} gamma / tau)^{1/(s-1)}.
import numpy as np, glob
from klib import *


def Ktilde(p, mu):
    F = (p - 1) // 2; d = np.arange(1, p)
    return np.array([np.sum(mu * (1 - np.exp(-2j * np.pi * l * d / p))) for l in range(1, F + 1)])


files = sorted(glob.glob('lm_p*_k*.npz'))
print(f'{len(files)} stored local maximisers')
worst = dict(low=-np.inf, up=-np.inf, crit=-np.inf, law=0.0, floor=-np.inf)
for f in files:
    d = np.load(f); net = Net(d['U'], d['W'], float(d['b']))
    p, n, nu, s = net.p, net.n, net.nu, net.s
    if net.margin() <= 0: continue
    act, q, res, h = kkt_measure(net, rtol=1e-6)
    if res > 1e-6: continue                                   # only well-polished KKT points
    Q = q_full(net, act, q); mu = Q.sum(0); qX = Q.sum(1)
    t, cres, M = critical_values(net, Q)
    tau, _ = injective_norm(M, starts=40, iters=200)
    lowb = nu ** (nu / 2) * Bp(p, n) * np.abs(Ktilde(p, mu)).max()
    upb = np.sqrt(2) * min(1.0, np.linalg.norm(qX))
    w = net.shares(); gam = net.gamma()
    law = np.max(np.abs(t / (nu ** (nu / 2) * gam * w ** (1 - s)) - 1))
    floor = (nu ** (nu / 2) * gam / tau) ** (1 / (s - 1))
    worst['low'] = max(worst['low'], lowb / tau - 1); worst['up'] = max(worst['up'], tau / upb - 1)
    worst['crit'] = max(worst['crit'], t.max() / tau - 1); worst['law'] = max(worst['law'], law)
    worst['floor'] = max(worst['floor'], floor / w.min() - 1)
    print(f'{f}: res {res:.0e}; tau {tau:.4e} in [{lowb:.4e}, {upb:.4e}]; max t/tau {t.max()/tau:.4f}; law err {law:.1e}; w_min/floor {w.min()/floor:.3f}')
print('worst (all must be <= 0 up to rounding, law error ~ KKT accuracy):', {k: float(f'{v:.2e}') for k, v in worst.items()})
