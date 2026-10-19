# (FR) on (a) IF networks (equality expected), (b) stored local maxima (E_kkt), (c) p=3 winners via count vectors.
import sys, glob, numpy as np, itertools
from math import factorial
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_width'); sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import elib as EW
from cjlib import gamma_exact
from check_FR import FR_terms, clean_Q, Gam
print("(a) IF networks at programme allocations:")
for (p, n, s) in [(5, 2, 1.5), (7, 2, 1.3), (7, 3, 1.7), (11, 2, 1.4), (5, 3, 2.0)]:
    G, ys = gamma_exact(p, s); y = ys[0]; U, W = EW.if_network(p, n, y); net = EW.Net(U, W)
    r = 1/s; nu = n+1; h = net.margin(); mn, M = FR_terms(net, s); Q = clean_Q(net, s); live = Q > 1e-14
    Xi = nu**(1-r)*M[live]/Q[live]; avg = np.sum(Q[live]*Xi)/np.sum(Q[live]); gIF = nu**(1-s)*EW.Bp(p, n)*G
    print(f"   p={p} n={n} s={s}: gamma/gamma_IF={net.gamma(s)/gIF:.8f}  avgXi^s={avg**s:.8f}  Xi_l={np.round(Xi, 8)}")
print("(b) stored local maxima (E_kkt):")
for f in sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p*_k*.npz'))[:24]:
    d = np.load(f); keys = list(d.keys())
    U = d['U'] if 'U' in keys else d[keys[0]]; W = d['W'] if 'W' in keys else d[keys[1]]
    net = EW.Net(U, W); p, n = net.p, net.n; nu = n+1
    b = float(f.split('_b')[1].split('_k')[0]); s = nu/b; r = 1/s; G = Gam(p, s)
    h = net.margin()
    if h <= 0: continue
    mn, M = FR_terms(net, s); gIF = nu**(1-s)*EW.Bp(p, n)*G
    # class C test: frequency-pure up to tolerance?
    sp = net.split() if hasattr(net, 'split') else None
    print(f"   {f.split('/')[-1]}: gamma/gamma_IF={net.gamma(s)/gIF:.5f}  h={h:.4e} <= Gam(min_x)^s={G*mn**s:.4e} (ratio {h/(G*mn**s):.4f}) <= Gam(sumM)^s={G*M.sum()**s:.4e}")
