# Sanity check of the relaxation chain (relations case, exact over Z_p, floating point):
#   h_model <= O1 <= CP <= LR,
#   h_model = min_{w,d} [ (K A)_d + phi(w) - phi(w-d) ]      (gap of the reduced model, A = a + c v)
#   O1      = min_d [ (K A)_d - (1/2) osc_w D_d ]             (Note, Lemma osc)
#   CP      = min_{d,w} [ (K A)_d + psi_d(w) ],  psi_d(w) = sum_l v_l kappa_l(d) cos(2 pi l w/p + th_l) = (D_d(w) + D_{-d}(w))/2
#   LR      = Gamma * min_w || (A + v cos(2 pi l w/p + th))_+ ||   (label-row principle; Gamma = programme value, here the
#             value of the best allocation on file, a LOWER bound for Gamma_s(p), so LR is checked conservatively)
# and the label-row average bound  E_w sum_l row_l(w)_+^r <= Xi_p sum_l a_l^r  (Xi_p from xi_finite_p.out; here A = a + c v,
# v <= tau' a, so t_l = v_l/A_l <= 1 and a_l = (1 - c t_l) A_l: exactly the P12-extremal configurations covered by Xi).
import numpy as np
from lib import *

rng = np.random.default_rng(7)
worst = {'h<=O1': -1e9, 'O1<=CP': -1e9, 'CP<=LR': -1e9, 'rowavg<=Xi*sum a^r': -1e9, 'rowavg/sum a^r': -1e9}
Xi = {61: 1.07058, 101: 1.06911, 211: 1.06832}
for trial in range(240):
    p = int(rng.choice([61, 101, 211])); s = 1.5; r, c, tp = consts(s)
    G = load_support(p)[0]
    F = (p-1)//2; K = kmat(p); l = np.arange(1, F+1); w = np.arange(p)
    k = int(rng.integers(3, F+1)); S = rng.choice(F, k, replace=False)
    a = np.zeros(F); a[S] = rng.exponential(1, k)**rng.uniform(0.3, 2.5)
    v = np.zeros(F); v[S] = tp*a[S]*rng.uniform(0, 1, k)**rng.uniform(0.2, 2)
    th = rng.uniform(0, 2*np.pi, F)
    A = a + c*v
    cosm = np.cos(2*np.pi*np.outer(l, w)/p + th[:, None])
    phi = (v[:, None]*cosm).sum(0)
    h = np.inf; o1 = np.inf; cp = np.inf
    for d in range(1, F+1):
        base = (A*K[d-1]).sum()
        for dd in (d, p-d):
            h = min(h, base + (phi - np.roll(phi, dd)).min())
        D = phi - np.roll(phi, d)
        o1 = min(o1, base - 0.5*(D.max()-D.min()))
        psi = ((v*K[d-1])[:, None]*cosm).sum(0)
        cp = min(cp, base + psi.min())
    rows = np.maximum(A[:, None] + v[:, None]*cosm, 0)
    lr = G*min(np.sum(rows[:, j][rows[:, j] > 0]**r)**s for j in range(p))
    worst['h<=O1'] = max(worst['h<=O1'], h - o1)
    worst['O1<=CP'] = max(worst['O1<=CP'], o1 - cp)
    worst['CP<=LR'] = max(worst['CP<=LR'], cp - lr)
    rowavg = np.mean((rows**r).sum(0))
    worst['rowavg<=Xi*sum a^r'] = max(worst['rowavg<=Xi*sum a^r'], rowavg - Xi[p]*np.sum(a[S]**r))
    worst['rowavg/sum a^r'] = max(worst['rowavg/sum a^r'], rowavg/np.sum(a[S]**r))
for kk, vv in worst.items():
    if kk == "rowavg/sum a^r": print(f"max ratio     {kk:22s}: {vv:.4f}      (must be <= Xi_p, 1.0683..1.0706 here)")
    else: print(f"max violation {kk:22s}: {vv:+.3e}   (must be <= ~0)")
