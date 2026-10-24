# v4: the per-feature projection principle (= cost decoupling CD:  h <= pi * min_d sum_l Q_l^s kappa_l(d))
# is FALSE in class C.  Legendre network (C_joint J4/J6): feature l in F = n-atom single-set block with
#     Phi_l(x) = beta + 2 chihat(l) rho^{l w},  w = 2(x_1+...+x_n),  beta = |2 chihat(l)| = 2/sqrt(p).
# Then h = sqrt(p) + min_{w,d}[chi(w) - chi(w-d)] = sqrt(p) - 2,  Q_l = n^{1-r} beta^r, and
#     CD ratio = h / (pi M(Q^s)) = (1 - 2/sqrt p) n^{1-s} / pi.
# At n = 2, pi = 3^{1-s} (Note Thm q1n2), so CD fails iff p > 4/c_2(s)^2, c_2(s) = 1-(2/3)^{s-1}.
# Checks: (a) exact margins from the field formula for many p; (b) full explicit network at p = 127, n = 2
# (balanced neurons, E_width Net) -> margin, cost, gamma; (c) RD still holds: upper bound on the RD ratio
# with a reweighted-LP lower bound on Gamma_s(p).
import numpy as np, sys
from fmlib import *
from elib import Net, Bp
from cjlib import atom_tables, gamma_lp


out = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

def legendre(p):
    x = np.arange(p); qr = set((x[1:] ** 2) % p)
    return np.array([0] + [1 if t in qr else -1 for t in range(1, p)])

def chihat(p, l):
    chi = legendre(p); return np.sum(chi * np.exp(-1j * TAU * l * np.arange(p) / p)) / p

def legendre_feats(p, n):
    F = (p - 1) // 2; feats = {}
    for l in range(1, F + 1):
        c = 2 * chihat(p, l); beta = abs(c)
        feats[l] = singleset_atoms(n, beta, np.angle(c))
    return feats

def h_oned(p):
    """margin via the one-variable potential (valid for every n, since w = 2 sum x_t covers Z_p)"""
    F = (p - 1) // 2; w = np.arange(p)[:, None]; d = np.arange(1, p)[None, :]
    g = np.zeros((p, p - 1))
    for l in range(1, F + 1):
        c = 2 * chihat(p, l); beta = abs(c)
        g += beta * kap(p, l, d) + np.real(c * np.exp(1j * TAU * l * w / p) * (1 - np.exp(-1j * TAU * l * d / p)))
    return g.min()

# (a) exact margins, n = 2 and n = 3 via the full field on Z_p^n for small p, one-variable formula otherwise
say('(a) Legendre networks: h vs sqrt(p)-2; CD ratio at n=2 (pi = 3^{1-s})')
for p in [11, 13, 31, 101, 103, 107, 109, 113, 127, 131, 139, 151, 199, 401, 1009]:
    h1 = h_oned(p)
    if p <= 31:
        for n in (2, 3):
            net = CNet(p, n, legendre_feats(p, n)); assert abs(net.h() - h1) < 1e-9 * abs(h1) + 1e-9
    row = []
    for s in (1.5, 1.4, 1.3, 1.2):
        r = 1 / s; F = (p - 1) // 2; beta = 2 / np.sqrt(p); n = 2
        Q = n ** (1 - r) * beta ** r
        cd = h1 / (price_n2(s) * Q ** s * (p / 2))
        row.append(f's={s}: CD={cd:.4f}')
    say(f'  p={p:5d}: h={h1:.6f}  sqrt(p)-2={np.sqrt(p)-2:.6f}  ' + '  '.join(row))

# thresholds
for s in (1.5, 1.4, 1.3, 1.2, 1.1):
    c2 = 1 - (2 / 3) ** (s - 1)
    say(f'  n=2, s={s}: c_2(s)={c2:.5f}, CD fails for every p > 4/c_2^2 = {4/c2**2:.1f}')

# (b) explicit network at p = 127, n = 2
p, n = 127, 2
feats = legendre_feats(p, n)
Us, Ws = [], []
for l, (C, xi) in feats.items():
    U, W = atom_tables(p, n, l, C, xi); Us.append(U); Ws.append(W)
U, W = balance(np.concatenate(Us), np.concatenate(Ws))
net = Net(U, W)
h_tab = net.margin()
for b in (2.0, 2.5):
    s = (n + 1) / b; r = 1 / s
    gam = net.gamma(s)
    beta = 2 / np.sqrt(p); Qtot = ((p - 1) // 2) * n ** (1 - r) * beta ** r
    gam_pred = Bp(p, n) * (np.sqrt(p) - 2) / Qtot ** s
    Ql = n ** (1 - r) * beta ** r
    cd = h_tab / (price_n2(s) * Bp(p, n) ** 0 * Ql ** s * (p / 2))
    say(f'(b) explicit p=127 n=2 b={b}: width={net.m}, h(table)={h_tab:.10f} (sqrt(p)-2={np.sqrt(p)-2:.10f}), '
        f'gamma={gam:.6e} (predicted {gam_pred:.6e}),  CD ratio={cd:.6f}')

# (c) RD survives: Gamma_lp <= Gamma, so h/(pi Gamma_lp (sum Q)^s) >= RD ratio
for p in [127, 199]:
    for s in (1.5, 1.3):
        G_lb, y = gamma_lp(p, s, restarts=12, seed=1)
        r = 1 / s; n = 2; beta = 2 / np.sqrt(p); F = (p - 1) // 2
        Qtot = F * n ** (1 - r) * beta ** r
        rd_ub = (np.sqrt(p) - 2) / (price_n2(s) * G_lb * Qtot ** s)
        say(f'(c) p={p} s={s}: Gamma_lp={G_lb:.5f} (K={int((y>1e-12).sum())}), RD ratio <= {rd_ub:.4f}')
open('v4_legendre.out', 'w').write('\n'.join(out) + '\n')
