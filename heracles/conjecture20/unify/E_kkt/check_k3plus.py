# Check of K3+ (no-cheap maximisers are IF): for class-C networks at b > 2 with no cheap feature,
#   gamma <= nu^{1-s} B_p Gamma^{(S)}_s - (detection at delta*) / P^s,  detection >= max_{l,T}|beta_{l,T}| sqrt(kappa_l(delta*)/2),
# where delta* minimises the Q-profile sum_l Q_l^s kappa_l(delta).  So equality forces IF.  Valid at every b (class C).
import numpy as np, itertools
from klib import *
from check_lemmas import restricted_programme

rng = np.random.default_rng(11)
rows = []
for trial in range(300):
    p = int(rng.choice([5, 7])); n = int(rng.choice([2, 3])); nu = n + 1; b = float(rng.uniform(2.0, nu)); s = nu / b; r = 1 / s
    F = (p - 1) // 2; S = list(range(1, F + 1)) if rng.random() < 0.5 else sorted(rng.choice(np.arange(1, F + 1), size=max(1, F - 1), replace=False).tolist())
    G, y = restricted_programme(p, s, S)
    base = if_network(p, n, y, b)
    eps = 10 ** rng.uniform(-4, -1)
    a_ = np.arange(p); U = base.U.copy(); W = base.W.copy()
    for i in range(base.m):                     # frequency-preserving (class C) perturbation of every neuron
        j = int(np.argmax(np.abs(base.atoms()[i]))) + 1
        cs = np.cos(2 * np.pi * j * a_ / p); sn = np.sin(2 * np.pi * j * a_ / p)
        for t in range(n):
            U[i, t] += eps * (rng.standard_normal() * cs + rng.standard_normal() * sn)
        W[i] += eps * (rng.standard_normal() * cs + rng.standard_normal() * sn)
    net = Net(U, W, b)
    beta = net.clean(); Q = (np.abs(net.atoms()) ** r).sum(0)
    cheap = np.max(np.abs(beta) / (nu ** (1 - s) * Q ** s))
    if cheap > 1: continue
    P = net.cost(); g = net.gaps(); h = g.min()
    prof = kappa_mat(p) @ (Q ** s)                            # Q-profile sum_l Q_l^s kappa_l(delta), delta = 1..p-1
    d = int(np.argmin(prof)) + 1
    sf = net.signflip(); kap = kappa_mat(p)
    det = max(abs(sf[(l, T)]) * np.sqrt(kap[d - 1, l - 1] / 2) for l in range(1, F + 1) for T in itertools.product([0, 1], repeat=n) if any(T))
    bound = nu ** (1 - s) * Bp(p, n) * G * P ** s
    rows.append((h / bound, (h - (bound - det)) / bound, det / bound, cheap))
rows = np.array(rows)
print(f'no-cheap class-C perturbations of IF optima: {len(rows)} samples')
print(f'  max gamma/(nu^(1-s) B_p Gamma)          = {rows[:,0].max():.6f}  (<= 1)')
print(f'  max (h - [bound - detection])/bound      = {rows[:,1].max():.3e}  (<= 0: equality forces zero detection, i.e. IF)')
print(f'  detection/bound ranges over [{rows[:,2].min():.2e}, {rows[:,2].max():.2e}]; cheapness max {rows[:,3].max():.6f}')
