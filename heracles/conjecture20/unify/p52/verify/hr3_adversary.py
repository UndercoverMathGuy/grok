"""hr3: adversarial maximisation of gamma/gamma_IF over GENERAL networks (DC, all frequencies) at (p,n)=(5,2) with
beta_2 = 0 imposed exactly (equality constraints), and at (7,2) with beta_3 = 0 (two features) or beta_2=beta_3=0.
SLSQP with analytic Jacobians; epigraph form max t s.t. g(x,d) >= t, cost = 1.
Usage: python hr3_adversary.py p kill(comma) b m nstarts seed"""
import sys
import itertools
import numpy as np
from scipy.optimize import minimize

p = int(sys.argv[1]); kill = [int(v) for v in sys.argv[2].split(',')]; b = float(sys.argv[3])
m = int(sys.argv[4]); NS = int(sys.argv[5]); seed = int(sys.argv[6])
rng = np.random.default_rng(seed)
n = 2; nu = 3; s = nu / b
X = np.array(list(itertools.product(range(p), repeat=2))); Y = X.sum(1) % p
a = np.arange(p)
Bp = 2 * (2 * p * nu) ** (-nu / 2)
F = list(range(1, (p - 1) // 2 + 1))


def Gamma(s):
    # programme value by brute force over the simplex boundary (small p): maximise min_d sum kappa_j(d) y_j, sum y^r = 1
    from scipy.optimize import minimize as mz
    kap = np.array([[1 - np.cos(2 * np.pi * j * d / p) for j in F] for d in range(1, p)])
    best = 0
    for trial in range(200):
        z = rng.random(len(F))
        def neg(z):
            y = np.abs(z); y = y / np.sum(y ** (1 / s)) ** s
            return -np.min(kap @ y)
        r = mz(neg, z, method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
        best = max(best, -r.fun)
    return best


GS = Gamma(s)
gIF = 3 ** (1 - s) * Bp * GS
rhoK = np.exp(-2j * np.pi * np.outer(a, a) / p)    # rhoK[k, a] = rho^{-k a}


def unpack(z):
    U1 = z[:m * p].reshape(m, p); U2 = z[m * p:2 * m * p].reshape(m, p); W = z[2 * m * p:3 * m * p].reshape(m, p)
    return U1, U2, W


def gaps_and_jac(z):
    U1, U2, W = unpack(z)
    P1 = U1[:, X[:, 0]]; P2 = U2[:, X[:, 1]]           # m x N
    G = []; J = []
    for d in range(1, p):
        dw = W[:, Y] - W[:, (Y - d) % p]                 # m x N
        g = np.sum(dw * P1 * P2, axis=0)
        G.append(g)
        N = len(X)
        jac = np.zeros((N, 3 * m * p + 1))
        for i in range(m):
            # d/dU1[i, a]
            jac[np.arange(N), i * p + X[:, 0]] += dw[i] * P2[i]
            jac[np.arange(N), m * p + i * p + X[:, 1]] += dw[i] * P1[i]
            jac[np.arange(N), 2 * m * p + i * p + Y] += P1[i] * P2[i]
            jac[np.arange(N), 2 * m * p + i * p + (Y - d) % p] -= P1[i] * P2[i]
        G.append(None); J.append(jac)
        G.pop()
    return np.concatenate(G), np.concatenate(J)


def con_g(zt):
    z, t = zt[:-1], zt[-1]
    g, _ = gaps_and_jac(z)
    return g - t


def con_g_jac(zt):
    z = zt[:-1]
    _, J = gaps_and_jac(z)
    J = J.copy(); J[:, -1] = -1.0
    return J


def cost(z):
    U1, U2, W = unpack(z)
    nr2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
    return np.sum(nr2 ** (b / 2)), nr2


def con_cost(zt):
    return np.array([cost(zt[:-1])[0] - 1.0])


def con_cost_jac(zt):
    U1, U2, W = unpack(zt[:-1]); _, nr2 = cost(zt[:-1])
    f = b * nr2 ** (b / 2 - 1)
    g = np.concatenate([(f[:, None] * U1).ravel(), (f[:, None] * U2).ravel(), (f[:, None] * W).ravel(), [0.0]])
    return g[None, :]


def betas(z, j):
    U1, U2, W = unpack(z)
    u1 = U1 @ rhoK[j]; u2 = U2 @ rhoK[j]; w = W @ rhoK[j]
    return 2 * p ** -3 * np.conj(w) * u1 * u2, u1, u2, w


def con_beta(zt):
    outv = []
    for j in kill:
        bi, *_ = betas(zt[:-1], j); B = bi.sum()
        outv += [B.real, B.imag]
    return np.array(outv)


def con_beta_jac(zt):
    rows = []
    for j in kill:
        bi, u1, u2, w = betas(zt[:-1], j)
        c = 2 * p ** -3
        dU1 = c * (np.conj(w) * u2)[:, None] * rhoK[j][None, :]
        dU2 = c * (np.conj(w) * u1)[:, None] * rhoK[j][None, :]
        dW = c * (u1 * u2)[:, None] * np.conj(rhoK[j])[None, :]
        g = np.concatenate([dU1.ravel(), dU2.ravel(), dW.ravel(), [0]])
        rows += [g.real, g.imag]
    return np.array(rows)


def h_of(z):
    return gaps_and_jac(z)[0].min()


def init(kind):
    if kind == 0:
        z = rng.normal(size=3 * m * p)
    else:   # IF block of feature 1 (3 neurons) + small noise on the rest
        z = rng.normal(size=3 * m * p) * 0.05
        U1, U2, W = [v.copy() for v in unpack(z)]
        ph = rng.uniform(0, 2 * np.pi)
        for i in range(min(3, m)):
            sh = 2 * np.pi * i / 3
            U1[i] += np.cos(2 * np.pi * a / p + ph + sh); U2[i] += np.cos(2 * np.pi * a / p + sh)
            W[i] += np.cos(2 * np.pi * a / p + ph + 2 * sh)
        z = np.concatenate([U1.ravel(), U2.ravel(), W.ravel()])
    z = z / cost(z)[0] ** (1 / b)
    return z


best = -np.inf; bestz = None
for st in range(NS):
    z0 = init(st % 2)
    zt0 = np.concatenate([z0, [h_of(z0)]])
    cons = [dict(type='ineq', fun=con_g, jac=con_g_jac), dict(type='eq', fun=con_cost, jac=con_cost_jac),
            dict(type='eq', fun=con_beta, jac=con_beta_jac)]
    try:
        res = minimize(lambda zt: -zt[-1], zt0, jac=lambda zt: np.concatenate([np.zeros(len(zt) - 1), [-1.0]]),
                       constraints=cons, method='SLSQP', options=dict(maxiter=600, ftol=1e-12))
    except Exception as ex:
        continue
    z = res.x[:-1]
    # exact re-evaluation of gamma (cost renormalised), residual beta
    c, _ = cost(z); z = z / c ** (1 / b)
    gam = h_of(z)
    resid = max(abs(betas(z, j)[0].sum()) for j in kill)
    feats = [abs(betas(z, j)[0].sum()) for j in F]
    ratio = gam / gIF
    if resid < 1e-9 and ratio > best:
        best = ratio; bestz = z
    print(f"start {st}: gamma/gIF = {ratio:.5f}  resid beta_killed = {resid:.1e}  |beta_j| = {np.round(feats, 5)}", flush=True)
print(f"p={p} kill={kill} b={b} s={s:.4f} m={m}: Gamma_s={GS:.6f}; BEST gamma/gamma_IF (feasible) = {best:.5f}")
