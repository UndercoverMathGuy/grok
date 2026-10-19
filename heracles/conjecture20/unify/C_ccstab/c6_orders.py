# True orders along rays theta_0 + eps V from the decoupled (IF block-union) maximiser theta_0 at s = nu/2:
#   deficits Em_sig, dint, dprog-change, Renyi concentration  H_{1/s}(m_0) - H_{1/s}(m_eps),  and the RD-IF slack
#   S(eps) = log gamma_IF - log gamma(theta_eps; s).   Fitted log-log slopes over eps in [1e-4, 1e-2].
import numpy as np, math
from sclib import *

rng = np.random.default_rng(5)
def slope(es, vs):
    es = np.array(es); vs = np.abs(np.array(vs)); ok = vs > 1e-14
    if ok.sum() < 3: return float('nan')
    return float(np.polyfit(np.log(es[ok]), np.log(vs[ok]), 1)[0])

for (p, n, b) in [(5, 2, 2.0), (7, 2, 2.0), (11, 2, 2.0), (5, 3, 2.0)]:
    nu = n + 1; s = nu / b
    Gs, ys = programme_exact(p, s); y = ys[0]
    th0 = G.if_network(p, n, y, b)
    d0 = decomp(th0); H0 = renyi(d0['m'], 1 / s)
    gIF = nu ** (1 - s) * Bp(p, n) * Gs
    print('(p,n,b)=(%d,%d,%g): theta_0 = IF block union, gamma/gamma_IF=%.6f, dprog_0=%.4f' % (p, n, b, th0.gamma() / gIF, d0['dprog']))
    kinds = ['general', 'classC-aligned', 'phase-only', 'mass-only']
    for kind in kinds:
        sl = {k: [] for k in ['Em', 'dint', 'dprog', 'dH', 'S']}
        for trial in range(6):
            if kind == 'general':
                VU = rng.normal(size=th0.U.shape); VW = rng.normal(size=th0.W.shape)
            else:
                # perturb the block parameters: amplitudes / phases of each class-C neuron
                pass
            es = np.logspace(-4, -2, 7); rec = {k: [] for k in sl}
            for e in es:
                if kind == 'general':
                    net = Net(th0.U + e * VU * np.abs(th0.U).max(), th0.W + e * VW * np.abs(th0.W).max(), b)
                else:
                    r2 = np.random.default_rng(100 * trial + 7)
                    specs = []
                    for j in range(len(y)):
                        if y[j] <= 0: continue
                        lam = (2 ** n * y[j] / nu) ** (1.0 / nu)
                        for k in range(nu):
                            al = np.pi * k / nu
                            damp = r2.normal(size=nu) if kind in ('classC-aligned', 'mass-only') else np.zeros(nu)
                            dph = r2.normal(size=n) if kind in ('classC-aligned', 'phase-only') else np.zeros(n)
                            specs.append((j + 1, lam * np.exp(e * damp), al + e * dph))
                    net = classC_net(p, n, specs, b)
                d = decomp(net)
                rec['Em'].append(d['Em_sig']); rec['dint'].append(d['dint']); rec['dprog'].append(d['dprog'] - d0['dprog'])
                rec['dH'].append(H0 - renyi(d['m'], 1 / s)); rec['S'].append(math.log(gIF) - math.log(net.gamma()))
            for k in sl: sl[k].append(slope(es, rec[k]))
        print('   %-15s slopes: Em_sig %s | dint %s | dprog-change %s | Renyi drop %s | RD-IF slack %s' %
              (kind, *['%.2f..%.2f' % (np.nanmin(sl[k]), np.nanmax(sl[k])) for k in ['Em', 'dint', 'dprog', 'dH', 'S']]))
