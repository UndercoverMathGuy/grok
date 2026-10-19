# Sanity checks of the KKT structure theorem (K1) on interference-free (IF) optima, and of the IF dual
# formulas: t_j = nu^{nu/2} B_p K_j(mu*), tau(q) = nu^{nu/2} B_p max_l K_l(mu), w_i ~ t_i^{-1/(s-1)}.
import numpy as np, sys
from klib import *

np.set_printoptions(precision=6, suppress=True)
cases = [(5, 2, 2.5), (5, 2, 2.0), (7, 2, 2.2), (5, 3, 2.5), (7, 3, 3.0)]
for (p, n, b) in cases:
    nu = n + 1; s = nu / b
    G, ys, zs = programme_exact(p, s)
    y = ys[0]
    net = if_network(p, n, y, b)
    c = net.cost(); net = Net(net.U * c ** (-1 / b), net.W * c ** (-1 / b), b)
    gam = net.gamma(); gIF = gamma_if(p, n, b, G)
    act, q, res, h = kkt_measure(net, rtol=1e-9)
    Q = q_full(net, act, q)
    t, cres, M = critical_values(net, Q)
    a = net.a(); w = net.shares(); H = net.H()
    # KKT predictions
    t_pred = nu ** (nu / 2) * (h / net.cost()) * a ** ((1 - s) / s)
    lhs = np.log(nu ** (nu / 2) * gam); rhs = np.sum(w * np.log(t)) - (s - 1) * H
    tau, _ = injective_norm(M, starts=60, iters=200)
    mu = Q.sum(0); mu = mu / mu.sum()                        # delta-marginal (offsets 1..p-1)
    K = kappa_mat(p).T @ mu                                   # K_l(mu) for l=1..F  (shape F)
    tau_unif = nu ** (nu / 2) * Bp(p, n) * K.max()
    # x-uniformity of q: total variation distance from p^{-n} mu
    tv = 0.5 * np.abs(Q - mu[None, :] / net.N).sum()
    print(f'(p,n,b)=({p},{n},{b}) s={s:.4f} gamma/gIF={gam/gIF:.8f}  KKT residual={res:.2e}  |act|={len(act)}  TV(q,unif x mu)={tv:.2e}')
    print(f'   max|t-t_pred|/t = {np.max(np.abs(t-t_pred)/t):.2e}   critical-point residual max = {cres.max():.2e}')
    print(f'   identity log(nu^(nu/2) gamma) = E_w log t - (s-1)H : {lhs:.10f} vs {rhs:.10f}  diff {lhs-rhs:.2e}')
    print(f'   H = {H:.6f}  log nu + H(z*) = {np.log(nu)+entropy(zs[0]):.6f}')
    print(f'   tau(q) = {tau:.8e}  nu^(nu/2) B_p max_l K_l(mu) = {tau_unif:.8e}   t/tau in [{t.min()/tau:.6f}, {t.max()/tau:.6f}]')
    print(f'   distinct t_i: {np.unique(np.round(t/tau, 8))}   K_l(mu)/K_max: {K/K.max()}')
