# Sanity checks (exact over Z_p, floating point) of the reduction chain in the relations case:
#   h_model <= min_d O1(d) <= min_d Pars(d) <= min_d TDP(d)
# where, for an allocation a, interference 0<=v<=tau' a, phases th, clean amplitudes A = a + c v (the largest P12 allows):
#   h_model = min_{w,d} [ sum_l A_l kappa_l(d) + phi(w) - phi(w-d) ],  phi(w) = sum_l v_l cos(2 pi l w/p + th_l)
#   O1(d)   = sum_l A_l kappa_l(d) - (1/2) osc_w (phi(w)-phi(w-d))                    (Note, Lemma osc)
#   Pars(d) = sum_l A_l kappa_l(d) - sqrt(sum_l v_l^2 kappa_l(d))                     (Bhatia-Davis + Parseval)
#   TDP(d)  = (K a)_d + c tau' * tail_d(a)   (tail_d = greedy kappa-budget 1/c^2, lib.tail_bound)   [this note]
# Also checks: waterfill excess E0 <= tail_bound/c (dual feasibility), and exactness of E0 by brute-force maximisation.
import numpy as np
from scipy.optimize import minimize
from lib import *

rng = np.random.default_rng(11)
worst = {'h<=O1': -1e9, 'O1<=Pars': -1e9, 'Pars<=TDP': -1e9, 'E0<=tail': -1e9, 'E0 brute': -1e9}
for trial in range(300):
    p = int(rng.choice([31, 37, 61, 67, 101, 131]))
    s = rng.uniform(1.0, 1.5); r, c, tp = consts(s)
    F = (p-1)//2; K = kmat(p)
    k = int(rng.integers(3, F+1))
    S = rng.choice(F, size=k, replace=False)
    a = np.zeros(F); a[S] = rng.exponential(1, k)**rng.uniform(0.3, 2.5)
    v = np.zeros(F); v[S] = tp*a[S]*rng.uniform(0, 1, k)**rng.uniform(0.2, 2)
    th = rng.uniform(0, 2*np.pi, F)
    A = a + c*v
    l = np.arange(1, F+1); w = np.arange(p)
    phi = (v[:, None]*np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])).sum(0)
    h_model = np.inf; o1 = np.inf; pars = np.inf; tdp = np.inf
    for d in range(1, F+1):
        kd = K[d-1]
        base = (A*kd).sum()
        for dd in (d, p-d):
            D = phi - np.roll(phi, dd)
            h_model = min(h_model, base + D.min())
        D = phi - np.roll(phi, d)
        o1 = min(o1, base - 0.5*(D.max()-D.min()))
        pars = min(pars, base - np.sqrt((v**2*kd).sum()))
        tdp = min(tdp, (a*kd).sum() + c*tp*tail_bound(a, kd, c))
    worst['h<=O1'] = max(worst['h<=O1'], h_model - o1)
    worst['O1<=Pars'] = max(worst['O1<=Pars'], o1 - pars)
    worst['Pars<=TDP'] = max(worst['Pars<=TDP'], pars - tdp)
    # excess checks at a random delta
    d = int(rng.integers(1, F+1)); kd = K[d-1]
    E0 = waterfill_excess(a, kd, c); TB = tail_bound(a, kd, c)
    worst['E0<=tail'] = max(worst['E0<=tail'], E0 - TB)
    if trial < 60 and (c*c*kd[S].sum() > 1):
        # brute force: maximise c sum u k - sqrt(sum u^2 k) over 0<=u<=a (concave): projected multistart
        m = (a > 0)
        def neg(z):
            u = a[m]/(1+np.exp(-z)); return -(c*(u*kd[m]).sum() - np.sqrt((u**2*kd[m]).sum()+1e-300))
        best = -np.inf
        for t in range(4):
            res = minimize(neg, rng.normal(0, 2, m.sum()), method='L-BFGS-B')
            best = max(best, -res.fun)
        worst['E0 brute'] = max(worst['E0 brute'], best - E0)
for kk, vv in worst.items(): print(f"max violation {kk:10s}: {vv:+.3e}   (must be <= ~0)")
