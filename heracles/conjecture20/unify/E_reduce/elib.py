# E_reduce library. Imports the refereed GLOBAL network library read-only (unify/global/lib.py) and adds:
#  - exact weighted programme Gamma^lambda_s(p) by vertex enumeration (small F),
#  - cost-share entropy H(theta,s), output-frequency usage vector E(theta),
#  - weighted-margin optimiser (gaps sum_l lambda_l G_l, standard cost).
import sys, os, itertools, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'global'))
import lib as G   # noqa: E402
from scipy.optimize import minimize

Bp, Net, freq_proj, if_network, if_block, split, balance, prune = G.Bp, G.Net, G.freq_proj, G.if_network, G.if_block, G.split, G.balance, G.prune


def kap(p):
    F = (p - 1) // 2
    d = np.arange(1, F + 1)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)        # rows delta=1..F (delta ~ -delta), cols j=1..F


def programme_vertices(p, s, lam=None, S=None, tol=1e-9):
    """Exact Gamma^{lam,(S)}_s(p) = max{ min_delta sum_l lam_l y_l kappa_l(delta) : y>=0, spt y in S, sum y^{1/s} <= 1 }.
    Equivalent to 1/min{ (sum v^{1/s})^s : K_lam v >= 1, v >= 0, spt v in S } (concave minimisation -> vertices).
    Returns (Gamma, list of maximisers y (normalised), all vertices with values)."""
    F = (p - 1) // 2; r = 1.0 / s
    lam = np.ones(F) if lam is None else np.asarray(lam, float)
    S = list(range(F)) if S is None else [j - 1 for j in S]
    K = kap(p)[:, S] * lam[S][None, :]               # F x |S|
    k = len(S); verts = []
    # vertices of {v>=0, K v >= 1}: choose k tight constraints among F rows + k coordinate planes
    rows = [('r', i) for i in range(F)] + [('c', j) for j in range(k)]
    for comb in itertools.combinations(rows, k):
        A = np.zeros((k, k)); bb = np.zeros(k)
        for q, (typ, idx) in enumerate(comb):
            if typ == 'r': A[q] = K[idx]; bb[q] = 1
            else: A[q, idx] = 1; bb[q] = 0
        if abs(np.linalg.det(A)) < 1e-12: continue
        v = np.linalg.solve(A, bb)
        if np.any(v < -tol) or np.any(K @ v < 1 - tol): continue
        v = np.clip(v, 0, None)
        verts.append(v)
    best = []
    for v in verts:
        val = np.sum(v[v > 0] ** r) ** (-s)
        best.append((val, v))
    best.sort(key=lambda z: -z[0])
    Gam = best[0][0]
    maxs = []
    for val, v in best:
        if val > Gam * (1 - 1e-9):
            y = np.zeros(F); y[S] = v / np.sum(v[v > 0] ** r) ** s
            if not any(np.allclose(y, z, atol=1e-8) for z in maxs): maxs.append(y)
    return Gam, maxs, best


def zshares(y, s):
    z = np.where(y > 0, np.abs(y) ** (1.0 / s), 0.0); return z / z.sum()


def Hent(w):
    w = np.asarray(w, float); w = w[w > 0]; w = w / w.sum(); return float(-(w * np.log(w)).sum())


def neuron_sq(net):
    return (net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1)


def shares(net, s=None):
    s = net.s if s is None else s; b = net.nu / s
    a = neuron_sq(net) ** (b / 2); return a / a.sum()


def entropy(net, s=None):
    return Hent(shares(net, s))


def Evec(net, s=None):
    """E_l = sum_i w_i ||P_l w_i||^2 / ||omega_i||^2  (l = 1..F); nu*E sums to 1 for balanced, output-DC-free nets."""
    p = net.p; F = (p - 1) // 2; w = shares(net, s); sq = neuron_sq(net)
    out = []
    for l in range(1, F + 1):
        Pl = freq_proj(p, l)
        out.append(np.sum(w * ((net.W @ Pl) ** 2).sum(1) / np.where(sq > 0, sq, 1)))
    return np.array(out)


def out_weight_matrix(p, lam):
    F = (p - 1) // 2; M = freq_proj(p, 0).copy()
    for l in range(1, F + 1): M += lam[l - 1] * freq_proj(p, l)
    return M


def wmargin(net, lam):
    """weighted margin h_lam = min_{x,delta} sum_l lam_l G_l(x,delta) = margin of the net with outputs W @ Lambda."""
    Lm = out_weight_matrix(net.p, lam)
    return Net(net.U, net.W @ Lm, net.b).margin()


def soft_w(x, m, n, p, b, tau, Lm):
    U, W = G.unpack(x, m, n, p)
    netc = Net(U, W, b); c = netc.cost(); s = netc.s
    netg = Net(U, W @ Lm, b)
    g = netg.gaps() * c ** (-s)
    gmin = g.min(); z = np.exp(-(g - gmin) / tau); Z = z.sum(); lamq = z / Z
    Fv = gmin - tau * np.log(Z)
    gU, gWp = netg.grad_lin(lamq)
    gW = gWp @ Lm.T
    cU, cW = netc.grad_cost()
    Eg = (lamq * g).sum()
    dU = c ** (-s) * gU - s * Eg / c * cU
    dW = c ** (-s) * gW - s * Eg / c * cW
    return Fv, G.pack(dU, dW)


def optimise_w(U, W, b, lam, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5), iters=3000):
    m, n, p = U.shape; Lm = out_weight_matrix(p, lam)
    x = G.pack(U, W)
    sc = Bp(p, n) * (n + 1) ** (1 - (n + 1) / b)
    for tau in taus:
        res = minimize(lambda z: tuple(-v for v in soft_w(z, m, n, p, b, tau * sc, Lm)), x, jac=True,
                       method='L-BFGS-B', options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-12})
        x = res.x
    U, W = G.unpack(x, m, n, p)
    net = Net(U, W, b); c = net.cost(); k = c ** (-1.0 / b)
    return Net(U * k, W * k, b)


def wgamma(net, lam):
    return wmargin(net, lam) / net.cost() ** net.s
