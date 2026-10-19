"""KKT/dual-tensor check at the decoupled IF optimum: q = uniform(x) x mu*(delta), mu* the programme dual.
Verifies stationarity grad E_q g = lam grad cost, t_i = <E_q g, unit direction of neuron i> = h nu^{nu/2} a_i^{r-1}
(unit cost), and F = log gamma + (s-1) H = E_w log t - (nu/2) log nu  (GLOBAL R14 route 4 / BRIEF_E sec. 2)."""
import numpy as np
from scipy.optimize import linprog
from elib import G, nu_costs, shares, shannon
for (p, n, s) in [(7, 2, 1.3), (11, 2, 1.5), (7, 3, 1.8), (13, 2, 1.4)]:
    nu = n + 1; b = nu / s; r = 1 / s; F = (p - 1) // 2
    Gam, y, _ = G.programme(p, s); net = G.if_network(p, n, y, b)
    sc = net.cost() ** (-1 / b); net = G.Net(net.U * sc, net.W * sc, b)        # unit b-cost
    K = G.kappa_mat(p)[:F]; spt = y > 1e-9
    # programme dual: mu >= 0 on delta=1..F, sum mu = 1, K_j(mu) = lam y_j^{r-1} (j in spt), <= (j not in spt)
    # solve: find mu, lam with A_eq [K^T mu - lam y^{r-1}] = 0 on spt
    yr = np.where(spt, y, 1) ** (r - 1)
    nvar = F + 1
    Aeq = np.zeros((spt.sum() + 1, nvar)); beq = np.zeros(spt.sum() + 1)
    for k, j in enumerate(np.nonzero(spt)[0]):
        Aeq[k, :F] = K[:, j]; Aeq[k, F] = -yr[j]
    Aeq[-1, :F] = 1; beq[-1] = 1
    cov = K @ y; binding = cov <= cov.min() * (1 + 1e-7)            # complementary slackness: mu on binding offsets only
    bnds = [(0, None) if binding[d] else (0, 0) for d in range(F)] + [(0, None)]
    res = linprog(np.zeros(nvar), A_eq=Aeq, b_eq=beq, bounds=bnds, method='highs')
    mu = res.x[:F]
    lam = np.zeros((net.N, p - 1))
    for dlt in range(1, F + 1):                       # symmetric in +-delta, uniform in x
        lam[:, dlt - 1] += mu[dlt - 1] / 2 / net.N; lam[:, p - dlt - 1] += mu[dlt - 1] / 2 / net.N
    gU, gW = net.grad_lin(lam); cU, cW = net.grad_cost()
    g = np.concatenate([gU.ravel(), gW.ravel()]); c = np.concatenate([cU.ravel(), cW.ravel()])
    lm = (g @ c) / (c @ c); stat = np.linalg.norm(g - lm * c) / np.linalg.norm(g)
    h = net.margin(); a = nu_costs(net); w = shares(a, s)
    # t_i = E_q g of neuron i alone at unit factor norms
    t = []
    for i in range(net.m):
        Ui = net.U[i:i + 1] / np.linalg.norm(net.U[i], axis=1)[None, :, None]; Wi = net.W[i:i + 1] / np.linalg.norm(net.W[i])
        gi = G.Net(Ui, Wi, b).gaps(); t.append((lam * gi).sum())
    t = np.array(t)
    pred = h * nu ** (nu / 2) * a ** (r - 1)
    Fv = np.log(h) + (s - 1) * shannon(w)
    print('p=%d n=%d s=%.2f: stationarity residual %.1e; max|t/pred-1| %.1e; F - [E_w log t - nu/2 log nu] = %.1e; '
          'max t / max_i t = 1, spread of t: %.3f..%.3f' % (p, n, s, stat, np.max(np.abs(t / pred - 1)),
          Fv - ((w * np.log(t)).sum() - nu / 2 * np.log(nu)), t.min() / t.max(), 1.0))
