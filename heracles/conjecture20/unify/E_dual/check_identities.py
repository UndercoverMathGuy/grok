"""Checks for D1-D3 (Renyi-Legendre identities, KL form of the free energy, dust, product structure).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy --with mpmath python check_identities.py"""
import numpy as np, mpmath as mp
from elib import G, nu_costs, shares, shannon, renyi, kl

mp.mp.dps = 40
rng = np.random.default_rng(1)

print('(A) cost identity  s log sum a^{1/s} = log sum a + (s-1) H_{1/s}(a/sum a)   [50-digit]')
worst = 0
for trial in range(200):
    m = int(rng.integers(1, 30)); s = mp.mpf(1 + 3 * rng.random())
    a = [mp.mpf(float(x)) for x in rng.exponential(1.0, m) ** rng.uniform(0.2, 4)]
    Z1 = sum(a); p = [x / Z1 for x in a]
    lhs = s * mp.log(sum(x ** (1 / s) for x in a))
    Hr = mp.log(sum(q ** (1 / s) for q in p)) / (1 - 1 / s)
    worst = max(worst, abs(lhs - mp.log(Z1) - (s - 1) * Hr))
print('   max |error| =', mp.nstr(worst, 3))

print('(B,C) free energy F = log g(s) + (s-1)H(w_s) = log g(1) + KL(w_s||w_1);  Gibbs: log g(s) = min_pi[...]')
errB = 0; minC = np.inf; eqC = 0
for (p, n, m, j) in [(5, 2, 6, None), (7, 2, 9, None), (7, 3, 12, None), (11, 2, 15, None)]:
    Gam, y, _ = G.programme(p, 1.4)
    net0 = G.if_network(p, n, y)
    for trial in range(15):
        U = net0.U + 0.05 * rng.standard_normal(net0.U.shape) * rng.random()
        W = net0.W + 0.05 * rng.standard_normal(net0.W.shape) * rng.random()
        # random rescaling of neurons (keeps h>0 only approximately; rescale table-preserving: scale u_1 by c, w by 1/c)
        net = G.Net(U, W, 2.0)
        h = net.margin()
        if h <= 0: continue
        a = nu_costs(net); p1 = a / a.sum()
        for s in [1.2, 1.5, (n + 1) / 2.0]:
            lg1 = np.log(h) - np.log(a.sum()); lgs = np.log(h) - s * np.log((a ** (1 / s)).sum())
            w = shares(a, s)
            F = lgs + (s - 1) * shannon(w)
            errB = max(errB, abs(F - (lg1 + kl(w, p1))))
            for k in range(30):
                pi = rng.dirichlet(np.ones(len(a)) * rng.uniform(0.2, 3))
                val = lg1 + kl(pi, p1) - (s - 1) * shannon(pi)
                minC = min(minC, val - lgs)
            eqC = max(eqC, abs(lg1 + kl(w, p1) - (s - 1) * shannon(w) - lgs))
print('   max |F - (log g1 + KL)| =', errB, ';  min over random pi of [Gibbs value - log g(s)] =', minC,
      '(>=0);  |at pi=w_s| =', eqC)

print('(D) dust: add M cancelling pairs (T,-T) of nu-mass eps each; table unchanged')
p, n = 7, 2; Gam, y, _ = G.programme(p, 1.5); net = G.if_network(p, n, y); s = 1.5
a0 = nu_costs(net); h = net.margin()
Z0 = (a0 ** (1 / s)).sum(); c = 0.05 * Z0          # dust s-mass fixed at 5% of the network's
for M in [0, 10, 10 ** 3, 10 ** 6, 10 ** 9, 10 ** 12]:
    if M == 0:
        w = shares(a0, s); lgs = np.log(h) - s * np.log(Z0); H = shannon(w)
    else:
        eps = (c / (2 * M)) ** s                       # nu-mass of each dust neuron (total nu-mass 2M eps -> 0)
        Z = Z0 + c; w0 = a0 ** (1 / s) / Z; wd = c / Z / (2 * M)
        H = -(w0 * np.log(w0)).sum() - c / Z * np.log(wd)
        lgs = np.log(h) - s * np.log(Z)
    print('   M=%14d  total dust nu-mass=%.1e  log gamma=%.6f  H=%.4f  F=%.4f' % (M, 0 if M == 0 else 2 * M * eps, lgs, H, lgs + (s - 1) * H))

print('(E) decoupled IF network = product: p_Theta = yhat (x) uniform_nu, Renyi additive; programme in Renyi form')
for (p, n, s) in [(7, 2, 1.3), (11, 2, 1.45), (11, 3, 1.8), (13, 2, 1.5)]:
    Gam, y, _ = G.programme(p, s); net = G.if_network(p, n, y, b=(n + 1) / s)
    a = nu_costs(net); pp = a / a.sum(); r = 1 / s
    yh = y[y > 0] / y.sum()
    Hr_net = renyi(pp, r); Hr_prod = np.log(n + 1) + renyi(yh, r)
    K = G.kappa_mat(p)[:(p - 1) // 2]
    M = (K @ (y / y.sum())).min()
    lg = np.log(net.margin()) - s * np.log((a ** r).sum())
    pred = np.log(G.Bp(p, n) * M) - (s - 1) * Hr_prod
    progR = np.log(M) - (s - 1) * renyi(yh, r)
    print('   p=%d n=%d s=%.2f  H_r(net)-H_r(prod)=%.1e  log gamma - [log(B M(yhat)) - (s-1)H_r]=%.1e  '
          'log Gamma_s - [log M - (s-1)H_r(yhat)]=%.1e' % (p, n, s, Hr_net - Hr_prod, lg - pred, np.log(Gam) - progR))
