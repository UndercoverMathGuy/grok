# Is Theorem L's conclusion gamma <= (1-X) gamma_IF valid without h > 0?  N_S (E_reduce) contains h <= 0 networks.
# Build S = {1}: a small aligned IF block on feature 1 plus a large 'ghost' pair on feature 2 (beta_2 = 0, E_2 > 0).
import numpy as np, math
from vlib import *
from v2_theoremL import ledger
from v3_ifmanifold import block_specs
rng = np.random.default_rng(3); worst = None
for (p, n, b) in [(5, 2, 2.0), (7, 2, 2.0), (5, 3, 2.0)]:
    nu = n + 1
    for t in range(200):
        specs, _ = block_specs(1, n, 2 * np.pi * np.arange(nu) / nu, rng.uniform(0.01, 1))
        amp = rng.uniform(0.5, 3) * np.ones(nu); ph1 = rng.uniform(0, 2 * np.pi, n); ph2 = rng.uniform(0, 2 * np.pi, n)
        specs += [(2, amp, ph1, ph1.sum()), (2, amp, ph2, ph2.sum() + np.pi)]
        net = classC(p, n, specs, b)
        if abs(net.atoms().sum(0)[1]) > 1e-9: continue
        d = ledger(net, [1])
        viol = d['gam'] - (1 - d['X']) * d['gIF']
        if d['X'] >= 0 and (worst is None or viol > worst[0]): worst = (viol / d['gIF'], (p, n, b), d['X'], d['h'])
print('max over X>=0 of [gamma - (1-X) gamma_IF]/gamma_IF = %.4f at %s (X = %.3f, h = %.3e)' % worst)
