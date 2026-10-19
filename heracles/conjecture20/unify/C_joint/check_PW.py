# (PW): Xi_min := nu^{1-r} min_x sum_l (Re Phi_l(x))_+^r / sum_l Q_l <= 1 ?  (FR-min bound relative to gamma_IF is Xi_min^s.)
import sys, glob, numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
from frlib import *
rows = []
files = sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p*_k*.npz'))
for f in files + [f'/home/user/grok/heracles/conjecture20/unify/split/viol_p{p}_n2_b2.npz' for p in [5, 7, 11]]:
    d = np.load(f); keys = [k for k in d.keys() if k.startswith('U')]
    for kU in keys:
        kW = 'W' + kU[1:]
        net = EW.Net(d[kU], d[kW]); p, n = net.p, net.n; nu = n+1
        b = float(d['b']) if 'b' in d else 2.0; s = nu/b; r = 1/s
        if net.margin() <= 0: continue
        mn, M = FR_terms(net, s); Q = clean_Q(net, s)
        Xmin = nu**(1-r)*mn/Q.sum(); g = net.gamma(s)/(nu**(1-s)*EW.Bp(p, n)*Gam(p, s))
        rows.append((f.split('/')[-1] + ('' if kU == 'U' else ':' + kU), g, Xmin**s))
for name, g, x in rows: print(f"  {name}: gamma/gamma_IF={g:.4f}  FR-min bound/gamma_IF = Xi_min^s = {x:.4f}")
print(f"max Xi_min^s over {len(rows)} stored networks: {max(x for _, _, x in rows):.5f}")
