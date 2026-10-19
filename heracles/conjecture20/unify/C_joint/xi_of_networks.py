# Per-feature field-row efficiencies Xi_l = nu^{1-r} E_x (Re Phi_l)_+^r / Q_l and cheapness G_l on stored non-IF optima,
# to see which features FR treats as tame (Xi<=1) or wild (Xi>1).
import sys, glob, numpy as np
from frlib import *
for f in sorted(glob.glob('/home/user/grok/heracles/conjecture20/unify/E_kkt/lm_p*_k*.npz')) + ['/home/user/grok/heracles/conjecture20/unify/split/viol_p5_n2_b2.npz']:
    d = np.load(f)
    U, W = (d['U'], d['W']) if 'U' in d else (d['U0'], d['W0'])
    b = float(d['b']) if 'b' in d else 2.0
    net = EW.Net(U, W); p, n = net.p, net.n; nu = n+1; s = nu/b; r = 1/s
    if net.margin() <= 0: continue
    mn, M = FR_terms(net, s); Q = clean_Q(net, s); beta = net.clean(); G = Gam(p, s)
    gIF = nu**(1-s)*EW.Bp(p, n)*G; ratio = net.gamma(s)/gIF
    if ratio > 0.9999: continue
    live = Q > 1e-12*Q.max()
    Xi = np.where(live, nu**(1-r)*M/np.where(live, Q, 1), np.inf)
    Gl = np.where(live, np.abs(beta)/(nu**(1-s)*np.where(live, Q, 1)**s), 0)
    sh = Q/Q.sum()
    print(f"{f.split('/')[-1]} (b={b}): gamma/gIF={ratio:.4f}; per feature [cost share, cheapness G_l, Xi_l]: " +
          "; ".join(f"l={l+1}: [{sh[l]:.3f}, {Gl[l]:.3f}, {Xi[l]:.3f}]" for l in range(len(Q)) if sh[l] > 1e-6) +
          f";  M of non-features: {M[~live].sum()/M.sum() if (~live).any() else 0:.3f}")
