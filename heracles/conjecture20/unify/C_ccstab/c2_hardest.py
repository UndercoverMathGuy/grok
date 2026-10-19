# Where is the exchange inequality (RD) hardest, as a function of s, for a fixed network?
# n = 2: pi = 3^{1-s} (Note Thm q1n2), so D_F(s) = 3^{1-s} B_p Gamma_s exactly, and the RD slack of theta is
#   S_theta(s) = log D_F(s) - log gamma(theta;s) = log Gamma_s - (s-1) log 3 + log B_p - log gamma_1(theta) + (s-1) H_{1/s}(m).
# For an IF block union with clean profile yhat:  S(s) = log Gamma_s - log M(yhat) + (s-1) H_{1/s}(yhat)  (programme slack).
import numpy as np, math, sys
from sclib import *

def M(yh, p):
    K = ER.kap(p); return float((K @ yh).min())

def Lam(yh, s):
    yh = yh[yh > 0]; yh = yh / yh.sum(); return s * math.log((yh ** (1 / s)).sum())

out = []
for p in [5, 7, 11, 13, 17]:
    ss = np.round(np.linspace(1.0, 1.5, 51), 4)
    Gs = []; Hd = []; Ys = []
    for s in ss:
        Gam, maxs = programme_exact(p, s)
        Gs.append(Gam)
        hz = [renyi(ER.zshares(y, s), 1.0) for y in maxs]
        Hd.append(math.log(3) + min(hz)); Ys.append(maxs[int(np.argmin(hz))])
    Gs = np.array(Gs); Hd = np.array(Hd)
    dH = np.diff(Hd)
    print('p=%d: H^dec(s) = log3 + min H(z*): from %.4f to %.4f; #increases %d (max +%.2e), #decreases %d (max -%.3f)'
          % (p, Hd[0], Hd[-1], (dH > 1e-9).sum(), dH.max(), (dH < -1e-9).sum(), -dH.min()))
    # supports along s
    sup = [tuple(np.nonzero(y > 1e-9)[0] + 1) for y in Ys]
    ch = [(ss[i], sup[i - 1], sup[i]) for i in range(1, len(ss)) if sup[i] != sup[i - 1]]
    print('   support changes (s, before, after):', ch)
    # (a) decoupled maximisers at s0: argmin of slack over s
    print('   decoupled maximiser at s0 -> argmin_s slack, slack at s=1 and s=1.5:')
    for i0 in [5, 15, 25, 35, 45]:
        yh = Ys[i0] / Ys[i0].sum()
        S = np.array([math.log(Gs[k]) - math.log(M(yh, p)) + Lam(yh, ss[k]) for k in range(len(ss))])
        print('      s0=%.2f  argmin=%.2f  min=%.1e  S(1)=%.4f  S(1.5)=%.4f' % (ss[i0], ss[int(np.argmin(S))], S.min(), S[0], S[-1]))
    # (b) uniform IF network on F
    F = (p - 1) // 2; yh = np.ones(F) / F
    S = np.array([math.log(Gs[k]) - math.log(M(yh, p)) + Lam(yh, ss[k]) for k in range(len(ss))])
    print('   uniform IF network: argmin_s slack = %.2f (min %.4f, S(1.5)=%.4f)' % (ss[int(np.argmin(S))], S.min(), S[-1]))
    out.append((p, ss, Gs, Hd))
np.save('c2_programme.npy', np.array(out, dtype=object), allow_pickle=True)

# (c) generic networks: (5,2) and (7,2) random positive-margin perturbations of IF optima and the SPLIT violators
print('\n(c) perturbed IF networks and stored violators: argmin over s in [1,1.5] of the RD slack')
rng = np.random.default_rng(3)
for p in [5, 7, 11]:
    dat = [r for r in out if r[0] == p][0]; ss, Gs = dat[1], dat[2]
    B = Bp(p, 2)
    nets = []
    for s0 in [1.1, 1.3, 1.5]:
        Gm, ys = programme_exact(p, s0)
        base = G.if_network(p, 2, ys[0], 2.0)
        for t in range(6):
            eps = [0.02, 0.05, 0.1][t % 3]
            nets.append(('IF(s0=%.1f)+%.2f' % (s0, eps), Net(base.U + eps * rng.normal(size=base.U.shape) * np.abs(base.U).max(),
                                                               base.W + eps * rng.normal(size=base.W.shape) * np.abs(base.W).max(), 2.0)))
    try:
        d = np.load('/home/user/grok/heracles/conjecture20/unify/split/viol_p%d_n2_b2.npz' % p)
        for kk in range(4): nets.append(('viol%d' % kk, Net(d['U%d' % kk], d['W%d' % kk], 2.0)))
    except FileNotFoundError:
        pass
    for name, net in nets:
        h = net.margin()
        if h <= 0: continue
        a = masses(net); m = a / a.sum(); g1 = h / a.sum()
        S = np.array([math.log(Gs[k]) - (ss[k] - 1) * math.log(3) + math.log(B) - math.log(g1) + (ss[k] - 1) * renyi(m, 1 / ss[k]) for k in range(len(ss))])
        Hs = [renyi(np.where(m > 0, m ** (1 / ss[k]), 0) / (m ** (1 / ss[k])).sum(), 1.0) for k in range(len(ss))]
        k0 = int(np.argmin(S))
        print('  p=%2d %-16s argmin s=%.2f  minS=%.4f  S(1)=%.4f S(1.5)=%.4f  H(theta,s*)-H^dec(s*)=%+.3f' %
              (p, name, ss[k0], S.min(), S[0], S[-1], Hs[k0] - dat[3][k0]))
