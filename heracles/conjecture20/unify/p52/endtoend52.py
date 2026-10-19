"""End-to-end numerical check of the (5,2) chain on actual networks with beta_2 = 0:
  gamma/gamma_IF <= [certificate bound h_cert]/(Gamma sqrt(F^2 + R^2/12))   with F = rigorous payment of the data,
and every intermediate inequality.  Networks: random, IF-perturbed, single-atom-plus-compensators, and networks
locally optimised for the margin under beta_2 = 0 (penalised), at s = 1.5, 1.25.
"""
import numpy as np
from scipy.optimize import minimize
from p52lib import *
from sanity52 import make_net, kill_beta2, check_LB_LC, own_data
from hs2 import elim_ratio
from cert_hs import cert_data, GAMMA
from pay_adapt import payment

rng = np.random.default_rng(7)


def torus_normalise(A, bT):
    """rotate (B, C, D) by (e^{i t1}, e^{i t2}, e^{i(t1+t2)}) to make B, C real >= 0; return alpha, m, phi"""
    B, C, D = bT
    t1 = -np.angle(B); t2 = -np.angle(C)
    Dn = D * np.exp(1j * (t1 + t2))
    return np.angle(A), abs(B), abs(C), abs(D), np.angle(Dn)


def opt_net(s, m=6, iters=300):
    """crude local optimisation of gamma under beta_2 = 0 (projection after each step)"""
    U = rng.normal(size=(m, 2, 5)); W = rng.normal(size=(m, 5)); W = kill_beta2(U, W)
    b = 3 / s

    def f(x):
        U_ = x[:m * 10].reshape(m, 2, 5); W_ = kill_beta2(U_, x[m * 10:].reshape(m, 5))
        g = gaps(U_, W_); c = cost(U_, W_, b)
        tau = 0.02 * np.abs(g).mean()
        sm = -tau * np.log(np.mean(np.exp(-(g - g.min()) / tau))) + g.min()
        return -sm / c ** s
    x = np.concatenate([U.ravel(), W.ravel()])
    res = minimize(f, x, method='Powell', options={'maxiter': iters, 'xtol': 1e-4, 'ftol': 1e-6})
    U_ = res.x[:m * 10].reshape(m, 2, 5); W_ = kill_beta2(U_, res.x[m * 10:].reshape(m, 5))
    return U_, W_


if __name__ == '__main__':
    worst = 0; rows = []
    nets = [make_net(k) for k in ['random', 'IFpert', 'atom'] * 6]
    # near-optimal single-feature networks from the GLOBAL agent's penalised optimiser (outputs on all frequencies)
    import sys as _s
    _s.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global')
    from w1_compensation import run as wrun
    for seed in range(3):
        (val, net), vals = wrun(5, 2, 2.0, (1,), 8, 3, seed=seed)
        if net is not None:
            nets.append((net.U.copy(), kill_beta2(net.U.copy(), net.W.copy())))
    for (U, W) in nets:
        c0 = cost(U, W, 2.0) ** 0.5
        U = U / c0 ** (1 / 3); W = W / c0 ** (1 / 3)
        A, bT, Phi = own_data(U, W)
        if abs(A) < 1e-9:
            continue
        for s in (1.5, 1.25):
            r = 1 / s; b = 3 / s
            h = gaps(U, W).min()
            gam = h / cost(U, W, b) ** s
            gIF = 3 ** (1 - s) * B5 * Gamma(s)
            _, _, R, P1 = check_LB_LC(U, W, s)
            sc = abs(A)
            al, m1, m2, m12, phi = torus_normalise(A, bT)
            m1 /= sc; m2 /= sc; m12 /= sc
            F = payment(np.array([m1, m2, m12]), np.array([m1, m2, m12]), phi + al, phi + al)   # s=3/2 payment (valid lower bound at every s<=3/2)
            # certificate at the normalised data (s = 3/2 objective); its bound applies at every s
            val, Gs = elim_ratio(al, m1, m2, m12, phi, F, deg=2, ret=True)
            cd = cert_data(Gs)
            a0 = complex(cd['a0'][0].hi, cd['a0'][1].hi)
            # bound on h for the actual (unnormalised) network: rotate certificate back = evaluate at normalised data
            hb = (np.real(np.exp(-1j * al) * complex(cd['a0'][0].lo, cd['a0'][1].lo)) + m1 * cd['aT'][0][0].lo + m2 * cd['aT'][1][0].lo
                  + m12 * np.real(np.exp(-1j * phi) * complex(cd['aT'][2][0].lo, cd['aT'][2][1].lo))) * sc + (R) * float(cd['n'].hi)
            triv = KA * np.cos(al) * sc
            hb = min(hb, triv)
            bound_ratio = hb / (Gamma(s) * np.sqrt((F * sc) ** 2 + 3 ** (2 - 2 * s) * R ** 2 / 4))
            ok = (gam / gIF <= bound_ratio + 1e-9) and (h <= hb + 1e-9 * abs(hb)) and (P1 >= F * sc * (1 - 1e-9))
            worst = max(worst, gam / gIF)
            rows.append((s, gam / gIF, bound_ratio, h, hb, P1, F * sc, ok))
    for row in rows:
        print("s=%.2f gamma/gIF=%.4f  bound=%.4f  h=%.5f<=hb=%.5f  P1=%.4f>=F=%.4f  ok=%s" % row)
    print("all ok:", all(r[-1] for r in rows), " max gamma/gIF observed:", round(worst, 4))
