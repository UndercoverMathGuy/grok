# C_ccstab library.  Imports the refereed GLOBAL network library and the E_reduce exact programme read-only.
#   decomp(net)        : the convex-plus-correction identity (C_cc/IDENTITY.md) and its slack-field form
#   slack_stats(net)   : moments of the slack field sigma(x,delta) = g(x,delta) - h >= 0
#   renyi(m, a)        : Renyi entropy of order a
#   fields(net)        : feature fields Phi_l(x), l in F, on Z_p^n (flattened x)
import sys, os, itertools, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'global'))
sys.path.insert(0, os.path.join(HERE, '..', 'E_reduce'))
import lib as G          # noqa: E402
import elib as ER        # noqa: E402

Bp, Net, kappa_mat = G.Bp, G.Net, G.kappa_mat


def renyi(m, a):
    m = np.asarray(m, float); m = m[m > 0]; m = m / m.sum()
    if abs(a - 1) < 1e-12: return float(-(m * np.log(m)).sum())
    return float(np.log((m ** a).sum()) / (1 - a))


def masses(net):
    return ((net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1)) ** (net.nu / 2)


def fields(net):
    """Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-lc}, l = 1..F; returns (N, F) complex."""
    p = net.p; F = (p - 1) // 2
    f = net.table().reshape(net.N, p)
    c = np.arange(p)
    out = np.zeros((net.N, F), complex)
    for l in range(1, F + 1):
        out[:, l - 1] = (2.0 / p) * np.exp(2j * np.pi * l * net.Y / p) * (f @ np.exp(-2j * np.pi * l * c / p))
    return out


def decomp(net):
    """IDENTITY: gamma(theta;1) = B_p Gamma_1 (1 - Em_sig - dprog - dint).  Also the slack-field form:
       N := B_p Gamma_1 sum a;  dint*N = min_delta sbar_sym(delta);  dprog*N = mean_delta sbar - min_delta sbar_sym."""
    p, n, nu = net.p, net.n, net.nu
    B = Bp(p, n); G1 = p / (p - 1)
    a = masses(net); m = a / a.sum()
    g = net.gaps(); h = g.min()
    at = net.atoms(); beta = at.sum(0)
    sig = 1 - np.real(at).sum(1) / (B * a)
    K = kappa_mat(p)[:(p - 1) // 2]
    clean_min = (K @ np.real(beta)).min()
    Nn = B * G1 * a.sum()
    dprog = (G1 * np.real(beta).sum() - clean_min) / Nn
    dint = (clean_min - h) / Nn
    # slack field
    sl = g - h                                   # (N, p-1), delta = 1..p-1
    sbar = sl.mean(0)
    sbar_sym = 0.5 * (sbar + sbar[::-1])         # delta <-> p - delta
    dint_s = sbar_sym.min() / Nn
    dprog_s = (sbar.mean() - sbar_sym.min()) / Nn
    gam1 = h / a.sum()
    return dict(gamma1=gam1, ratio1=gam1 / (B * G1), Em_sig=float((m * sig).sum()), dprog=dprog, dint=dint,
                dprog_slack=dprog_s, dint_slack=dint_s, m=m, sig=sig, h=h, Nn=Nn)


def slack_energy(net):
    """Check: sum_{l in F} E_x |Phi_l - E Phi_l|^2  =  (2/p^2) E_x[ p sum_delta tau^2 - (sum_delta tau)^2 ],
       tau = sigma - E_x sigma (x-centred slack)."""
    p = net.p
    Phi = fields(net); f = Phi - Phi.mean(0)
    lhs = (np.abs(f) ** 2).mean(0).sum()
    g = net.gaps(); tau = g - g.mean(0)
    rhs = (2.0 / p ** 2) * (p * (tau ** 2).sum(1) - tau.sum(1) ** 2).mean()
    return lhs, rhs


def gamma_s(net, s):
    """gamma(theta;s) for the same network at exponent b = nu/s (W1)."""
    a = masses(net); m = a / a.sum()
    return net.margin() / a.sum() * math.exp(-(s - 1) * renyi(m, 1.0 / s))


def programme_exact(p, s, S=None):
    """Exact Gamma^{(S)}_s(p) via E_reduce vertex enumeration (small p)."""
    Gam, maxs, _ = ER.programme_vertices(p, s, S=S)
    return Gam, maxs


def classC_net(p, n, specs, b):
    """specs: list of (l, amps[n+1], phis[n]); neuron u_t = amp_t cos(2 pi l x/p + phi_t), w = amp_n cos(2 pi l c/p + sum phi).
       Atoms are then real positive (aligned); xi_it = exp(-2 i phi_t)."""
    x = np.arange(p); U = []; W = []
    for (l, amps, phis) in specs:
        U.append([amps[t] * np.cos(2 * np.pi * l * x / p + phis[t]) for t in range(n)])
        W.append(amps[n] * np.cos(2 * np.pi * l * x / p + np.sum(phis)))
    return Net(np.array(U), np.array(W), b)
