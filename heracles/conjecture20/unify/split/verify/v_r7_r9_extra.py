"""Follow-ups: (1) R7(a) equality clause under the 'theta-binding offsets' reading (exact tie construction);
(2) R9(c) at b = nu: ascent from an IF network plus mixing noise drives P_live/P -> 1; (3) R8 ratio digits.
Run: OMP_NUM_THREADS=1 uv run --with numpy --with scipy python v_r7_r9_extra.py
"""
import numpy as np
from scipy.optimize import minimize
from vlib import *

rng = np.random.default_rng(4242)
p, n = 5, 2


def if_block(p, n, j, lam=1.0):
    nu = n + 1; a = np.arange(p)
    U = np.zeros((nu, n, p)); W = np.zeros((nu, p))
    for l in range(nu):
        al = np.pi * l / nu
        for t in range(n):
            U[l, t] = lam * np.cos(2 * np.pi * j * a / p + al)
        W[l] = lam * np.cos(2 * np.pi * j * a / p + n * al)
    return U, W

# (1) tie construction
Uc, Wc = if_block(p, n, 1); fc = table(Uc, Wc)
beta = own_from_table(fc)[0][1].real
e = np.zeros(p); e[2] = e[3] = -1.0
xs = np.indices((p, p)); y = xs.sum(0) % p
r = np.cos(2 * np.pi * (xs[0] - xs[1]) / p)
A = (kappa(1, 2, p) - kappa(1, 1, p)) * beta / (-r.min())
R = np.zeros((p, p, p))
for c in range(p):
    R[..., c] = A * r * e[(c - y) % p]
Us_, Ws_ = [], []
for a0 in range(p):
    Ua, S, Vt = np.linalg.svd(R[a0])
    for k in range(p):
        if S[k] > 1e-14:
            u1 = np.zeros(p); u1[a0] = 1.0
            Us_.append([u1, Ua[:, k] * S[k]]); Ws_.append(Vt[k])
U = np.concatenate([Uc, np.array(Us_)]); W = np.concatenate([Wc, np.array(Ws_)])
f = table(U, W); fs = table(*split(U, W))
gt = gaps(f); gs = gaps(fs)
dm_t = gt[..., 1:].min((0, 1)); dm_s = gs[..., 1:].min((0, 1))
print('directional minima theta:', np.round(dm_t, 12), ' split:', np.round(dm_s, 12))
print('h(theta) - h(split) = %.3e ; max|g_R| on d=2,3: %.3f ; theta-binding offsets: %s' % (
    margin_f(f) - margin_f(fs), np.abs(gt[..., 2] - gs[..., 2]).max(),
    [d for d in range(1, p) if dm_t[d - 1] < dm_t.min() + 1e-9]))
print('max |beta_T(theta)| = %.2e (theta is IF)' % max(abs(v) for v in own_from_table(f)[1].values()))

# (2) b = nu = 3 ascent from IF network (programme maximiser y = (1/2,1/2) at s = 1) plus mixing noise
b = 3.0; s = 1.0; nu = 3
G1, ystar = programme(p, 1.0)
Ubs, Wbs = [], []
for j, yj in zip(F_(p), [0.5, 0.5]):
    Ub, Wb = if_block(p, n, j, lam=yj ** (1 / nu))
    Ubs.append(Ub); Wbs.append(Wb)
U0 = np.concatenate(Ubs); W0 = np.concatenate(Wbs)
g0 = margin(U0, W0) / cost(U0, W0, b) / (Bp(p, n) * G1)
print('IF network at b=3: gamma/(B_p Gamma_1) = %.6f' % g0)
m = len(W0)
def objg(z, tau):
    U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
    G = gaps(table(U, W))[..., 1:].ravel(); sc = np.abs(G).mean() + 1e-12
    smin = G.min() - tau * sc * np.log(np.exp(-(G - G.min()) / (tau * sc)).sum())
    return -smin / cost(U, W, b)
for trial in range(4):
    z = np.concatenate([U0.ravel(), W0.ravel()]) + 0.25 * rng.normal(size=m * n * p + m * p)
    U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
    st = (margin(U, W) / cost(U, W, b) / (Bp(p, n) * G1), sum(P_live_by_freq(U, W, b).values()) / cost(U, W, b))
    for tau in [0.03, 0.01, 0.003, 0.001]:
        z = minimize(objg, z, args=(tau,), method='L-BFGS-B', options=dict(maxiter=800)).x
    U = z[:m * n * p].reshape(m, n, p); W = z[m * n * p:].reshape(m, p)
    g = margin(U, W) / cost(U, W, b) / (Bp(p, n) * G1); live = sum(P_live_by_freq(U, W, b).values()) / cost(U, W, b)
    print(f'R9c trial {trial}: start (gamma/gamma*, P_live/P) = ({st[0]:.4f}, {st[1]:.4f}) -> after ascent ({g:.6f}, {live:.6f}); '
          f'R8 at s=1 requires gamma/gamma* <= P_live/P: {g <= live + 1e-12}')
