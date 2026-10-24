# Check S1 (slack-field form of the identity) and the slack Parseval identity on random and structured networks.
import numpy as np, sys
from sclib import *

rng = np.random.default_rng(7)
worst = dict(id=0, dint=0, dprog=0, par=0)
cases = [(5, 2, 2.0, 6), (7, 2, 2.5, 9), (5, 3, 2.0, 8), (3, 2, 2.2, 4), (11, 2, 2.0, 7), (7, 3, 3.0, 6)]
for (p, n, b, mw) in cases:
    for t in range(5):
        U = rng.normal(size=(mw, n, p)); W = rng.normal(size=(mw, p))
        if t % 2:      # class-C-like: project every neuron to one random frequency
            F = (p - 1) // 2
            for i in range(mw):
                P = G.freq_proj(p, rng.integers(1, F + 1)); U[i] = U[i] @ P; W[i] = W[i] @ P
        net = Net(U, W, b)
        d = decomp(net)
        s = net.s
        pred = Bp(p, n) * p / (p - 1) * (1 - d['Em_sig'] - d['dprog'] - d['dint']) * math.exp(-(s - 1) * renyi(d['m'], 1 / s))
        worst['id'] = max(worst['id'], abs(pred - net.gamma()) / abs(net.gamma()))
        worst['dint'] = max(worst['dint'], abs(d['dint'] - d['dint_slack']))
        worst['dprog'] = max(worst['dprog'], abs(d['dprog'] - d['dprog_slack']))
        l, r = slack_energy(net)
        worst['par'] = max(worst['par'], abs(l - r) / max(abs(l), 1e-300))
print('random networks (general and frequency-pure), cases', cases)
print('max rel err identity            :', worst['id'])
print('max |dint - min_delta sbar_sym/N|:', worst['dint'])
print('max |dprog - (mean-min)sbar/N|   :', worst['dprog'])
print('max rel err slack Parseval        :', worst['par'])
# IF optima: slack is x-independent; dint = 0
for (p, n, b) in [(5, 2, 2.0), (11, 2, 2.0), (7, 3, 2.0)]:
    s = (n + 1) / b; Gm, y, _ = G.programme(p, s)
    net = G.if_network(p, n, y, b); d = decomp(net); g = net.gaps()
    print('IF', (p, n, b), 'dint=%.2e dprog=%.4f  max_x-var of slack=%.2e' % (d['dint'], d['dprog'], (g.var(0)).max()))

# S2: all-network joint detection  sum_l E_l <= 2(1-1/p) sigma_max N (dprog+dint);  class C: sigma_max <= 2^{n+2} N / Gamma_1
print('\nS2 joint detection (every network) and the class-C spike bound:')
rng = np.random.default_rng(9); wj = np.inf; ws = np.inf; minratio = np.inf
for (p, n, b, mw) in cases + [(5, 4, 2.0, 6)]:
    for t in range(8):
        U = rng.normal(size=(mw, n, p)); W = rng.normal(size=(mw, p)); cc = t % 2
        if cc:
            F = (p - 1) // 2
            for i in range(mw):
                P = G.freq_proj(p, rng.integers(1, F + 1)); U[i] = U[i] @ P; W[i] = W[i] @ P
        net = Net(U, W, b); d = decomp(net)
        Phi = fields(net); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0).sum()
        sl = net.gaps() - net.margin(); smax = sl.max(); Nn = d['Nn']
        wj = min(wj, (2 * (1 - 1 / p) * smax * Nn * (d['dprog'] + d['dint']) - E) / Nn ** 2)
        if cc:
            ws = min(ws, (2 ** (n + 2) * Nn / (p / (p - 1)) - smax) / Nn)
            minratio = min(minratio, (d['dprog'] + d['dint']) * Nn ** 2 / E)
print('  min [2(1-1/p) smax N D\' - sum E]/N^2  (>=0) =', wj)
print('  min [2^{n+2} N/Gamma_1 - smax]/N       (>=0) =', ws)
print('  smallest observed D\' N^2 / sum E on class-C samples =', minratio)

# S2': uniform-offset variance identity (every network):  E_{x,delta} F^2 = (Gamma_1/2)[ sum_l E_l + 2 E_x (Re sum_l f_l)^2 ]
# and single-offset diagonality E_x F_delta^2 = sum_l kappa_l E_l fails off class C (cross-feature terms)
print('\nS2\' uniform-offset variance identity (general networks) and single-offset off-diagonal size:')
rng = np.random.default_rng(13); werr = 0; offd = 0
for (p, n, b, mw) in cases:
    for t in range(4):
        net = Net(rng.normal(size=(mw, n, p)), rng.normal(size=(mw, p)), b)
        Phi = fields(net); f = Phi - Phi.mean(0); E = (np.abs(f) ** 2).mean(0)
        g = net.gaps(); Fm = g - g.mean(0)
        lhs = (Fm ** 2).mean(); rhs = (p / (p - 1)) / 2 * (E.sum() + 2 * (np.real(f.sum(1)) ** 2).mean())
        werr = max(werr, abs(lhs - rhs) / lhs)
        K = kappa_mat(p); diag = K @ E
        offd = max(offd, np.max(np.abs((Fm ** 2).mean(0) - diag) / diag))
print('  max rel err of the uniform-offset identity =', werr)
print('  max rel deviation of E_x F_delta^2 from sum kappa E on general networks =', offd, '(nonzero: cross terms)')
