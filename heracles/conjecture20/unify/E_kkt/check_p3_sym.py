# KKT data of the certified p=3 concentrated winners (n=11,12,13), polished as slot-symmetric networks
# (SLSQP from mechanism_scripts/kkt/polish.py, used read-only).  A slot-symmetric KKT point of the symmetric
# problem is a KKT point of the full problem (symmetrise q over slot permutations).
# Checks the KKT structure theorem K1 at p=3 and records the entropy / critical-value data showing that the
# IF analogue of (E) fails at p=3 (concentrated winner: lower entropy AND higher value than the IF block).
import numpy as np, sys
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts/kkt')
from kkt import Net as SNet, gamma_IF
from polish import polish
from load import build1d
from math import cos, pi

for nn in [11, 12, 13]:
    net, p, n = build1d(f'/home/user/grok/heracles/conjecture20/oned_certified/p3_n{nn}.json')
    nu = n + 1; b = 2.0; s = nu / b
    gIF = gamma_IF(p, n, b, 1 - cos(2 * pi / p))
    g0 = net.gamma() / gIF
    polish(net, iters=300, verbose=False)
    c = net.cost() ** (-1 / b); net.U *= c; net.W *= c            # cost 1
    g1 = net.gamma() / gIF
    for rtol in [1e-8, 1e-6, 1e-5]:
        A, q, lam, res, h = net.kkt(rtol)
        if res < 1e-6: break
    P = net.cost()
    sq = n * (net.U ** 2).sum(1) + (net.W ** 2).sum(1)              # ||omega_i||^2
    a = sq ** (nu / 2); w = a ** (1 / s) / np.sum(a ** (1 / s)); H = -np.sum(w * np.log(w))
    # balance check: n ||u||^2 vs nu-th share: ||u_t||^2 = ||w||^2 = ||omega||^2/nu
    bal = np.max(np.abs((net.U ** 2).sum(1) - (net.W ** 2).sum(1)) / sq)
    # critical values with unit factors (q = multipliers on class representatives)
    Uu = net.U / np.linalg.norm(net.U, axis=1, keepdims=True); Wu = net.W / np.linalg.norm(net.W, axis=1, keepdims=True)
    unit = SNet(Uu, Wu, n, b); Gi = unit.neuron_gaps()
    t = np.array([sum(qq * Gi[k, d, i] for (k, d), qq in zip(A, q)) for i in range(net.m)])
    t_pred = nu ** (nu / 2) * (h / P) * a ** (-(s - 1) / s)
    lhs = np.log(nu ** (nu / 2) * h / P ** s); rhs = np.sum(w * np.log(t)) - (s - 1) * H
    print(f'p=3 n={n}: gamma/gIF cert {g0:.6f} polished {g1:.6f}; KKT res {res:.1e}, lambda {lam:.6e} vs s h/P {s*h/P:.6e}, |A| {len(A)}, balance err {bal:.1e}')
    print(f'   width {net.m}; H = {H:.5f} < log nu = {np.log(nu):.5f};  max|t/t_pred-1| = {np.max(np.abs(t/t_pred-1)):.2e};  identity diff {lhs-rhs:.2e}')
    tIF = nu ** (nu / 2) * 2 * (2 * p * nu) ** (-nu / 2) * (1 - cos(2 * pi / p))
    print(f'   E_w log t = {np.sum(w*np.log(t)):.5f}, IF block log t_IF = {np.log(tIF):.5f}: deficit {np.log(tIF)-np.sum(w*np.log(t)):.5f} < (s-1)(log nu - H) = {(s-1)*(np.log(nu)-H):.5f}')
