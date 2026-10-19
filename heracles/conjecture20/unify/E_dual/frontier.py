"""Entropy-margin frontiers at fixed s: all networks vs decoupled networks (copies of one single feature, scaled).
Maximises J = log softmin(g / P^s) - lam * H(theta, s) for a scan of lam >= 0, from many starts.
Usage: python frontier.py p n b mode m lam1,lam2,... nstarts seed   (mode = all | dec)
Each line of output: mode lam m H gamma/gamma_IF (exact margin) Kr"""
import sys, numpy as np
from scipy.optimize import minimize
from elib import G, nu_costs, shares, shannon, renyi

p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
mode, m = sys.argv[4], int(sys.argv[5])
lams = [float(v) for v in sys.argv[6].split(',')]
nst, seed = int(sys.argv[7]), int(sys.argv[8])
nu = n + 1; s = nu / b; F = (p - 1) // 2
Gam, ystar, _ = G.programme(p, s); gIF = G.gamma_if(p, n, b, Gam)
rng = np.random.default_rng(seed)
OUT1 = G.freq_proj(p, 0) + G.freq_proj(p, 1)          # single-feature outputs: span{1, cos, sin} at frequency 1


def perm_mat(j):
    P = np.zeros((p, p))
    for a in range(p): P[a, (j * a) % p] = 1.0         # (P v)(a) = v(j a)
    return P


PERMS = [perm_mat(j) for j in range(1, F + 1)]
MU = [1.0]


def ent_grad(U, W):
    c = ((U ** 2).sum((1, 2)) + (W ** 2).sum(1)) ** (b / 2)
    C = c.sum(); w = c / C; H = -(w * np.log(w + 1e-300)).sum()
    S = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
    coef = -(b / C) * S ** (b / 2 - 1) * (np.log(w + 1e-300) + H)
    return H, coef[:, None, None] * U, coef[:, None] * W


def J_net(U, W, tau, lam):
    """value and gradient of log softmin(g/P^s) - lam H for the network (U, W)."""
    m_ = U.shape[0]
    x = G.pack(U, W)
    Fv, gr = G.softgamma(x, m_, n, p, b, tau)
    H, hU, hW = ent_grad(U, W)
    gU, gW = G.unpack(gr, m_, n, p)
    e0 = 1e-3 * gIF                                     # C^1 extension of log below e0
    if Fv > e0:
        lv, dl = np.log(Fv), 1.0 / Fv
    else:
        lv, dl = np.log(e0) + (Fv - e0) / e0, 1.0 / e0
    if lam >= 0:
        return lv - lam * H, (dl * gU - lam * hU, dl * gW - lam * hW)
    eta = -lam; ex = max(0.0, H - eta); mu = MU[0]
    return lv - mu * ex ** 2, (dl * gU - 2 * mu * ex * hU, dl * gW - 2 * mu * ex * hW)


def build_dec(x):
    """x = (theta_1 params, log c_2..c_F): union of theta_1 (outputs projected to frequency 1) and relabelled copies."""
    U1, W1 = G.unpack(x[:m * (n + 1) * p], m, n, p); W1 = W1 @ OUT1
    lc = x[m * (n + 1) * p:]
    Us = [U1]; Ws = [W1]
    for k in range(1, F):
        c = np.exp(lc[k - 1]); P = PERMS[k]
        Us.append(c * U1 @ P.T); Ws.append(c * W1 @ P.T)
    return U1, W1, np.concatenate(Us), np.concatenate(Ws)


def J_dec(x, tau, lam):
    U1, W1, U, W = build_dec(x)
    val, (gU, gW) = J_net(U, W, tau, lam)
    gU1 = gU[:m].copy(); gW1 = gW[:m].copy(); glc = []
    for k in range(1, F):
        c = np.exp(x[m * (n + 1) * p + k - 1]); P = PERMS[k]
        sl = slice(k * m, (k + 1) * m)
        gU1 += c * gU[sl] @ P; gW1 += c * gW[sl] @ P
        glc.append((gU[sl] * U[sl]).sum() + (gW[sl] * W[sl]).sum())
    gW1 = gW1 @ OUT1
    return val, np.concatenate([G.pack(gU1, gW1), np.array(glc)])


def run_one(x0, lam, fun):
    x = x0
    sched = [(t, 1.0) for t in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]]
    if lam < 0:
        sched = [(t, mu) for t, mu in zip([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-5, 1e-5],
                                          [1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])]
    for tau_rel, mu in sched:
        tau = tau_rel * gIF; MU[0] = mu

        def f(z):
            v, g = fun(z, tau, lam)
            return -v, -g
        res = minimize(f, x, jac=True, method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
        x = res.x
    return x


def J_all(z, tau, lam):
    v, g = J_net(*G.unpack(z, m, n, p), tau, lam)
    return v, (G.pack(*g) if isinstance(g, tuple) else np.zeros_like(z))


def report(U, W, lam, tag):
    net = G.Net(U, W, b); h = net.margin()
    a = nu_costs(net); keep = a > 1e-12 * a.max(); a = a[keep]
    w = shares(a, s); H = shannon(w)
    g = h / ((a ** (1 / s)).sum()) ** s
    print('%s lam=%.3f m=%d H=%.5f gamma/gIF=%.6f Kr=%.4f' % (tag, lam, m, H, g / gIF, np.exp(renyi(a / a.sum(), 1 / s))),
          flush=True)
    return H, g / gIF


def start_all():
    # perturbed IF block(s) on random supports, padded to width m
    U0, W0 = G.if_network(p, n, ystar * rng.uniform(0.3, 1.0, F), b).U, G.if_network(p, n, ystar, b).W
    net = G.if_network(p, n, ystar * rng.uniform(0.3, 1.0, F), b)
    U0, W0 = net.U, net.W
    if U0.shape[0] >= m:
        idx = rng.choice(U0.shape[0], m, replace=False); U0, W0 = U0[idx], W0[idx]
    else:
        k = m - U0.shape[0]
        U0 = np.concatenate([U0, 0.3 * rng.standard_normal((k, n, p))]); W0 = np.concatenate([W0, 0.3 * rng.standard_normal((k, p))])
    sc = rng.uniform(0.0, 0.4)
    return G.pack(U0 + sc * rng.standard_normal(U0.shape), W0 + sc * rng.standard_normal(W0.shape))


VIOL = np.load('../split/viol_p5_n2_b2.npz') if (p, n) == (5, 2) else None


def start_viol():
    k = int(rng.integers(0, 3)); U = VIOL['U%d' % k]; W = VIOL['W%d' % k]
    sc = rng.uniform(0.0, 0.05) * np.sqrt((U ** 2).mean())
    U = U + sc * rng.standard_normal(U.shape); W = W + sc * rng.standard_normal(W.shape)
    if m > U.shape[0]:
        U = np.concatenate([U, 1e-3 * rng.standard_normal((m - U.shape[0], n, p))]); W = np.concatenate([W, 1e-3 * rng.standard_normal((m - W.shape[0], p))])
    return G.pack(U, W)


def start_dec():
    U, W = G.if_block(p, n, 1, 1.0)
    if m > nu:
        U = np.concatenate([U, 0.3 * rng.standard_normal((m - nu, n, p))]); W = np.concatenate([W, 0.3 * rng.standard_normal((m - nu, p))])
    elif m < nu:
        idx = rng.choice(nu, m, replace=False); U, W = U[idx], W[idx]
    sc = rng.uniform(0.0, 0.4)
    U = U + sc * rng.standard_normal(U.shape); W = W + sc * rng.standard_normal(W.shape)
    return np.concatenate([G.pack(U, W), np.log(rng.uniform(0.2, 1.2, F - 1))])


for lam in lams:
    for t in range(nst):
        if mode in ('all', 'allv'):
            x = run_one(start_all() if mode == 'all' else start_viol(), lam, J_all)
            U, W = G.unpack(x, m, n, p)
        else:
            x = run_one(start_dec(), lam, J_dec)
            _, _, U, W = build_dec(x)
        report(U, W, lam, mode)
