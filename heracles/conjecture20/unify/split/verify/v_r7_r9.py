"""Referee checks of SPLIT R7 (span transfer), R8 (universal bound with live split costs), R9 (reduction; b = nu case),
including an explicit counterexample to the 'equality iff g_R = 0 on every binding offset' clause of R7(a).
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy python v_r7_r9.py
"""
import itertools, numpy as np
from scipy.optimize import minimize
from vlib import *

rng = np.random.default_rng(777)
lines = []
def rep(name, val, ok):
    lines.append(('PASS ' if ok else 'FAIL ') + f'{name}   [{val:+.4e}]')


def span_mod_p(vecs, p, n):
    V = {(0,) * n}; frontier = list(V)
    gens = [tuple(int(x) for x in v) for v in vecs if any(v)]
    while frontier:
        new = []
        for v in frontier:
            for g in gens:
                z = tuple((v[t] + g[t]) % p for t in range(n))
                if z not in V:
                    V.add(z); new.append(z)
        frontier = new
    return V


# ---------------- R7 main inequality on random networks
worst = -np.inf; worst_offV = -np.inf
for (p, n, m) in [(5, 2, 3), (7, 2, 3), (5, 3, 2), (7, 3, 2)]:
    for kind in ['gauss', 'sparse', 'heavydc']:
        for trial in range(3):
            U, W = rand_net(rng, m, n, p, kind)
            # make the split interference live on few slot-sets sometimes: kill some sign-flipped content by sparsity
            f = table(U, W); Us, Ws = split(U, W); fs = table(Us, Ws)
            for d in range(1, p):
                gs = ghat(fs, d); gt = ghat(f, d)
                V = span_mod_p(np.argwhere(np.abs(gs) > 1e-12), p, n)
                mask = np.zeros(gs.shape, bool)
                for v in V: mask[v] = True
                comb = np.real(np.fft.ifftn(np.where(mask, gt, 0)) * gt.size)      # g_split + g_R^V
                worst = max(worst, gaps(f)[..., d].min() - comb.min())
rep('R7 min g_theta(.,d) - min(g_split + g_R^V)(.,d)  (<=0)', worst, worst <= 1e-9)

# ---------------- R7(a): IF cores with mixing: h(theta) <= h(split)
def if_block(p, n, j, lam=1.0, ph=0.0):
    nu = n + 1; a = np.arange(p)
    U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for l in range(nu):
        al = np.pi * l / nu + ph
        for t in range(n):
            U[l, t] = lam * np.cos(2 * np.pi * j * a / p + al)
        W[l] = lam * np.cos(2 * np.pi * j * a / p + n * al)
    return U, W

worst = -np.inf
for (p, n) in [(5, 2), (7, 2), (5, 3), (7, 3)]:
    nu = n + 1
    for trial in range(15):
        blocks = [if_block(p, n, j, rng.uniform(.5, 1.5), rng.uniform(0, 6.3)) for j in F_(p)]
        U = sum(B[0][rng.permutation(nu)] for B in blocks); W = sum(B[1] for B in blocks)   # glue (shared neurons)
        U = U + rng.uniform(0, .6) * rng.normal(size=(nu, n, 1))          # input DC (keeps own data)
        W = W + rng.uniform(0, .6) * rng.normal(size=(nu, 1))              # output DC
        f = table(U, W)
        _, bT = own_from_table(f)
        if max(abs(v) for v in bT.values()) > 1e-9:
            continue                                                      # gluing may create interference; skip
        worst = max(worst, margin_f(f) - margin_f(table(*split(U, W))))
rep('R7a IF (beta_T = 0) mixed networks: h(theta) - h(split) (<=0)', worst, worst <= 1e-9)

# ---------------- R7(a) equality clause: explicit counterexamples at p = 5, n = 2
p, n = 5, 2
Uc, Wc = if_block(p, n, 1)                      # IF single feature, frequency 1
fc = table(Uc, Wc)
E = gaps(fc)[..., 1:].mean((0, 1))              # E_x g_core(., d), d = 1..4 (constant in x)
beta = own_from_table(fc)[0][1].real
e = np.zeros(p); e[2] = e[3] = -1.0              # e(c - y): e(0) = e(+-1) = 0, e(+-2) = -1
xs = np.indices((p, p)); y = xs.sum(0) % p
r = np.cos(2 * np.pi * (xs[0] - xs[1]) / p)      # character (1,-1): not in {0, 2j 1_T}
def realise(A):
    R = np.zeros((p, p, p))
    for c in range(p):
        R[..., c] = A * r * e[(c - y) % p]
    # realise R as a sum of rank-one product neurons: R = sum_a delta_a (x) M_a,  M_a = SVD
    Us, Ws = [], []
    for a0 in range(p):
        Ua, S, Vt = np.linalg.svd(R[a0])
        for k in range(p):
            if S[k] > 1e-14:
                u1 = np.zeros(p); u1[a0] = 1.0
                Us.append([u1, Ua[:, k] * S[k]]); Ws.append(Vt[k])
    return np.array(Us), np.array(Ws), R
for A, label in [(1.3 * (kappa(1, 2, p) - kappa(1, 1, p)) * beta, "'if' direction"),
                 ((kappa(1, 2, p) - kappa(1, 1, p)) * beta, "'only if' with theta-binding offsets")]:
    Ur, Wr, R = realise(A)
    U = np.concatenate([Uc, Ur]); W = np.concatenate([Wc, Wr])
    f = table(U, W)
    assert np.abs(f - fc - R).max() < 1e-10
    _, bT = own_from_table(f)
    Us, Ws = split(U, W); fs = table(Us, Ws)
    gs = gaps(fs); gt = gaps(f)
    dirmin_s = gs[..., 1:].min((0, 1)); dirmin_t = gt[..., 1:].min((0, 1))
    bind_s = [d for d in range(1, p) if dirmin_s[d - 1] < dirmin_s.min() + 1e-12]
    bind_t = [d for d in range(1, p) if dirmin_t[d - 1] < dirmin_t.min() + 1e-12]
    gR_on = {d: np.abs(gt[..., d] - gs[..., d]).max() for d in range(1, p)}
    print(f"R7a clause test ({label}): max|beta_T(theta)| = {max(abs(v) for v in bT.values()):.2e}, "
          f"|table(split) - core| = {np.abs(fs - fc).max():.2e}")
    print(f"   split binding offsets {bind_s}, theta binding offsets {bind_t}; max|g_R(.,d)| by d: "
          + ', '.join(f'{d}:{v:.3f}' for d, v in gR_on.items()))
    print(f"   h(theta) = {margin_f(f):.6f}   h(split) = {margin_f(fs):.6f}")
    if label == "'if' direction":
        ok = all(gR_on[d] < 1e-12 for d in bind_s) and margin_f(f) < margin_f(fs) - 1e-6
        rep("R7a 'equality IF g_R=0 on split-binding offsets' is FALSE: counterexample found", margin_f(f) - margin_f(fs), ok)
    else:
        ok = abs(margin_f(f) - margin_f(fs)) < 1e-9 and any(gR_on[d] > 1e-3 for d in bind_t)
        rep("R7a 'equality ONLY IF g_R=0 on theta-binding offsets' is FALSE: counterexample found", margin_f(f) - margin_f(fs), ok)

# ---------------- R8: h <= B_p min_d sum_j (P_j^live)^s kappa_j(d) <= B_p Gamma_s P_live^s, random + adversarial
def r8_ratio(U, W, b):
    m, n, p = U.shape; nu = n + 1; s = nu / b
    Pl = P_live_by_freq(U, W, b)
    rhs = Bp(p, n) * min(sum(Pl[k] ** s * kappa(k, d, p) for k in Pl) for d in range(1, p))
    return margin(U, W) / rhs
worst = -np.inf
for (p, n) in [(3, 2), (5, 2), (7, 2), (5, 3), (3, 3)]:
    for b in [2.0, 2.5, float(n + 1)]:
        m = 3
        for trial in range(4):
            z0 = rng.normal(size=m * n * p + m * p)
            def obj(z):
                U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
                G = gaps(table(U, W))[..., 1:].ravel(); sc = np.abs(G).mean() + 1e-12; tau = 0.01
                smin = G.min() - tau * sc * np.log(np.exp(-(G - G.min()) / (tau * sc)).sum())
                nu = n + 1; s = nu / b
                Pl = P_live_by_freq(U, W, b)
                rhs = Bp(p, n) * min(sum(Pl[k] ** s * kappa(k, d, p) for k in Pl) for d in range(1, p))
                return -smin / (rhs + 1e-300)
            res = minimize(obj, z0, method='L-BFGS-B', options=dict(maxiter=200))
            U = res.x[:m * n * p].reshape(m, n, p); W = res.x[m * n * p:].reshape(m, p)
            worst = max(worst, r8_ratio(U, W, b))
rep('R8 adversarial max h / (B_p min_d sum (P_j^live)^s kappa_j(d))  (<=1)', worst, worst <= 1 + 1e-9)

# ---------------- R9(c) (b = nu): gamma/(B_p Gamma_1) <= P_live/P for every network; ascent drives P_live/P -> 1
p, n = 5, 2; nu = 3; b = 3.0
G1, _ = programme(p, 1.0)
worst = -np.inf; finals = []
for trial in range(6):
    m = 5
    z0 = rng.normal(size=m * n * p + m * p)
    def objg(z, tau=0.005):
        U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
        G = gaps(table(U, W))[..., 1:].ravel(); sc = np.abs(G).mean() + 1e-12
        smin = G.min() - tau * sc * np.log(np.exp(-(G - G.min()) / (tau * sc)).sum())
        return -smin / cost(U, W, b)
    res = minimize(objg, z0, method='L-BFGS-B', options=dict(maxiter=3000))
    U = res.x[:m * n * p].reshape(m, n, p); W = res.x[m * n * p:].reshape(m, p)
    g = margin(U, W) / cost(U, W, b)
    ratio = g / (Bp(p, n) * G1); live = sum(P_live_by_freq(U, W, b).values()) / cost(U, W, b)
    worst = max(worst, ratio - live); finals.append((ratio, live))
rep('R9c b=nu: gamma/(B_p Gamma_1) - P_live/P (<=0)', worst, worst <= 1e-9)
print('R9c ascent at (5,2,b=3): (gamma/gamma*, P_live/P) =', [(round(a, 4), round(c, 4)) for a, c in finals])

for x in lines:
    print(x)
