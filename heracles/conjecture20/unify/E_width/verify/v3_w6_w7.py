import numpy as np, sys, itertools
from vlib import *
from fractions import Fraction as Fr
rng = np.random.default_rng(303)
# Rotfel'd/McCarthy for arbitrary (non-normal) matrices, 0<r<=1, and Weyl majorant
w = -np.inf
for t in range(3000):
    d = rng.integers(2, 7); k = rng.integers(2, 6); r = rng.uniform(0.2, 1)
    Xs = [rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)) for _ in range(k)]
    if t % 2: Xs = [np.outer(rng.normal(size=d) + 1j*rng.normal(size=d), rng.normal(size=d) + 1j*rng.normal(size=d)) for _ in range(k)]
    S = sum(Xs)
    lhs = np.sum(np.linalg.svd(S, compute_uv=False) ** r); rhs = sum(np.sum(np.linalg.svd(X, compute_uv=False) ** r) for X in Xs)
    wl = np.sum(np.abs(np.linalg.eigvals(S)) ** r)
    w = max(w, lhs / rhs - 1, wl / lhs - 1)
print(f"Rotfel'd + Weyl majorant: max relative excess {w:.2e} (<= ~1e-15)")

def chain(xi):
    N, n = xi.shape
    return np.concatenate([np.ones((N, 1), complex), np.cumprod(xi, 1)], 1)

def w6_check(beta, xi, s):
    n = xi.shape[1]; nu = n + 1; r = 1 / s
    rho = np.abs(beta); keep = rho > 1e-300; beta = beta[keep]; xi = xi[keep]; rho = rho[keep]
    A = beta.sum(); L = rho.sum(); Q = np.sum(rho ** r)
    m = chain(xi)
    H = np.einsum('l,la,lb->ab', beta, m, np.conj(m))
    lam = np.abs(np.linalg.eigvals(H)); lam = np.where(lam < 1e-9 * lam.max(), 0, lam); Lam = lam.sum() / nu; mu = lam / lam.sum()
    ea = max(np.sum(lam ** r) / (nu ** r * Q) - 1, Lam * np.exp((s - 1) * renyi(mu, r)) / Q ** s - 1,
             abs(A) * np.exp((s - 1) * renyi(mu, 1)) / Q ** s - 1, abs(A) / Lam - 1)
    pl = beta / rho
    u = m.copy(); u[:, 1:] *= pl[:, None]                  # TeX definition: u_{l,t} = p_l prod_{u<=t} xi (t>=1)
    K = np.einsum('l,la,lb->ab', rho, u, np.conj(u))
    ev = np.clip(np.linalg.eigvalsh(K), 0, None); ev = np.where(ev < 1e-9 * ev.max(), 0, ev); pi = ev / ev.sum()
    eb = L * np.exp((s - 1) * renyi(pi, r)) / Q ** s - 1
    esh = max(renyi(pi, a) - renyi(rho / L, a) for a in [0.25, r, 1.0, 2.0, 5.0])
    al = A / abs(A); eps = np.sqrt(max(2 * L * (L - abs(A)), 0))
    ec = 0; off = 0
    for tt in range(1, nu):
        ec = max(ec, abs(K[tt, 0] - np.sum(beta * np.prod(xi[:, :tt], 1))) / L)
        off += abs(K[tt, 0]) ** 2
        for tp in range(1, tt):
            ec = max(ec, (abs(K[tt, tp] - np.conj(al) * np.sum(beta * np.prod(xi[:, tp:tt], 1))) - eps) / L)
            off += abs(K[tt, tp]) ** 2
    Gn = abs(A) / (nu ** (1 - s) * Q ** s)
    ed = Gn / ((abs(A) / L) * (1 + 2 * off / (nu * L ** 2)) ** (s - 1)) - 1
    # aligned version
    return ea, eb, esh, ec, ed

W = np.full(5, -np.inf)
for n in range(1, 9):
    nu = n + 1
    for t in range(150):
        N = rng.integers(1, 3 * nu)
        kind = t % 4
        if kind == 0: beta = rng.normal(size=N) + 1j * rng.normal(size=N)
        elif kind == 1: beta = rng.exponential(size=N) * np.exp(1j * 0.3 * rng.normal(size=N))
        elif kind == 2: beta = rng.exponential(size=N) * np.exp(1j * np.pi * rng.integers(0, 2, N))   # real, cancelling
        else: beta = rng.exponential(size=N) + 0j                                                       # aligned
        xi = np.exp(2j * np.pi * rng.random((N, n)))
        if t % 5 == 0: xi = np.exp(2j * np.pi * rng.integers(0, 3, (N, n)) / 3)   # roots of unity
        for s in [1.0, 1.25, (1 + nu / 2) / 2, nu / 2]:
            W = np.maximum(W, w6_check(beta, xi, s))
print("W6 worst: (a) %.1e (b) %.1e Schur-Horn %.1e (c twist, /L) %.1e (c gain) %.1e" % tuple(W))
# aligned: G <= (1 + (2/nu) sum_I |beta_I|^2/|A|^2)^{s-1} <= exp(sum_I |beta_I|^2/|A|^2)
wa = -np.inf
for n in range(1, 10):
    nu = n + 1
    for t in range(200):
        N = rng.integers(1, 3 * nu); beta = rng.exponential(size=N) * np.exp(0.7j)
        xi = np.exp(2j * np.pi * rng.random((N, n)) * rng.uniform(0, 1))
        A = beta.sum()
        SI = sum(abs(np.sum(beta * np.prod(xi[:, a:b], 1))) ** 2 for a in range(n) for b in range(a + 1, n + 1)) / abs(A) ** 2
        for s in [1.0, 1.3, nu / 2]:
            G = abs(A) / (nu ** (1 - s) * np.sum(np.abs(beta) ** (1 / s)) ** s)
            wa = max(wa, G / (1 + 2 * SI / nu) ** (s - 1) - 1, (1 + 2 * SI / nu) ** (s - 1) / np.exp(SI) - 1)
print(f"W6(c) aligned chain of bounds: max excess {wa:.2e}")

# ---------------- W7 on networks (shared neurons, DC, all frequencies), with H_j = max(H(mu_j), H(rho_j/L_j))
def w7_slack(U, W, s, h=None):
    m, n, p = U.shape; nu = n + 1; r = 1 / s; F = (p - 1) // 2
    if h is None: h = margin(U, W)
    if h <= 0: return None
    Pb = np.sum(sqnorm(U, W) ** (nu / s / 2))
    lg = np.log(h) - s * np.log(Pb)
    Aabs = []; Hs = []; S = []
    for j in range(1, F + 1):
        beta, xi = atoms(U, W, j)
        A = beta.sum()
        if abs(A) < 1e-12 * max(1, np.abs(beta).sum()): continue
        keep = np.abs(beta) > 1e-14 * np.abs(beta).max()
        beta = beta[keep]; xi = xi[keep]
        m_ = chain(xi); H = np.einsum('l,la,lb->ab', beta, m_, np.conj(m_))
        lam = np.abs(np.linalg.eigvals(H)); lam = np.where(lam < 1e-11 * lam.max(), 0, lam)
        Hj = max(renyi(lam, r), renyi(np.abs(beta), r))
        Aabs.append(abs(A)); Hs.append(Hj); S.append(j)
    Aabs = np.array(Aabs); Hs = np.array(Hs)
    Sh = np.sum(Aabs ** r); q = Aabs ** r / Sh
    Ht = np.log(np.sum(q * np.exp((1 - r) * Hs))) / (1 - r) if r < 1 else np.sum(q * Hs)
    G, _ = programme(p, s, S)
    Dp = np.log(G * Sh ** s / h)
    lgif = np.log(nu ** (1 - s) * Bp(p, n) * G)
    return lgif + (s - 1) * (np.log(nu) - Ht) - Dp - lg, Dp

worst = np.inf; cnt = 0; minDp = np.inf
for (p, n) in [(5, 2), (7, 2), (5, 3), (3, 3), (7, 3)]:
    F = (p - 1) // 2; nu = n + 1
    for trial in range(16):
        s = rng.uniform(1, nu / 2)
        G, ys = programme(p, s); y = ys[0]
        Us, Ws = [], []
        for j in range(1, F + 1):
            if y[j - 1] <= 0: continue
            lam = (2 ** n * y[j - 1] / nu) ** (1 / nu)
            U0, W0 = if_block(p, n, j, lam); Us.append(U0); Ws.append(W0)
        U = np.concatenate(Us); W = np.concatenate(Ws)
        if trial % 3 == 1 and U.shape[0] >= 2 * nu:      # share neurons: merge block 1 and block 2 neuron-wise
            U = np.concatenate([U[:nu] + U[nu:2 * nu], U[2 * nu:]]); W = np.concatenate([W[:nu] + W[nu:2 * nu], W[2 * nu:]])
        eps = [0.0, 0.03, 0.1, 0.2][trial % 4]
        U = U + eps * rng.normal(size=U.shape) + eps * rng.normal(size=(U.shape[0], n, 1))   # DC
        W = W + eps * rng.normal(size=W.shape)
        out = w7_slack(U, W, s)
        if out is None: continue
        worst = min(worst, out[0]); minDp = min(minDp, out[1]); cnt += 1
print(f"W7 on {cnt} networks (incl. shared neurons, DC): min slack {worst:.2e} (>= 0); min Delta' {minDp:.2e} (>=0)")

sys.path.insert(0, '/home/user/grok/heracles/conjecture20')
import exact_certificate as ec
for n in [11, 13]:
    d = ec.NETWORKS[n]; p = 3; nu = n + 1; s = nu / 2
    Uf = [[Fr(a) for a in row] for row in d['u']]; Wf = [[Fr(a) for a in row] for row in d['w']]
    h = None
    for N0 in range(n + 1):
        for N1 in range(n + 1 - N0):
            N2 = n - N0 - N1; yv = (N1 + 2 * N2) % 3
            z = [Uf[i][0] ** N0 * Uf[i][1] ** N1 * Uf[i][2] ** N2 for i in range(len(Uf))]
            f = [sum(Wf[i][c] * z[i] for i in range(len(Uf))) for c in range(3)]
            for c in range(3):
                if c != yv and (h is None or f[yv] - f[c] < h): h = f[yv] - f[c]
    u = np.array([[float(a) for a in r] for r in d['u']]); w = np.array([[float(a) for a in r] for r in d['w']])
    U = np.repeat(u[:, None, :], n, axis=1)
    out = w7_slack(U, w, s, h=float(h))
    Pb = np.sum(n * (u ** 2).sum(1) + (w ** 2).sum(1)); g = float(h) / Pb ** s
    gif = nu ** (1 - s) * Bp(3, n) * 1.5
    print(f"  p=3 n={n}: gamma/gamma_IF = {g/gif:.6f}; W7 slack = {out[0]:.4f} (>=0); Delta' = {out[1]:.4f}")
