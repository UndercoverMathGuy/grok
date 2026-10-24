"""Referee-3 adversary: try to beat the certified bounds with actual networks.

Maximise h subject to cost = 1 (b = 3/s) and beta_k = 0 (k = 2 at p = 5; k = 3 at p = 7) with SLSQP, starting from
IF blocks + extra random neurons (with DC and every frequency) + noise.  Report the best gamma/gamma_IF and the number of
features of the optimum.  Certified claims: (5,2): <= 0.995 for s in [1, 1.5];  (7,2) two features: <= 0.9995 for s <= 1.182.
Usage: python ref3_adversary.py p s nstarts seed
"""
import sys, time
import numpy as np
from scipy.optimize import minimize

p = int(sys.argv[1]); s = float(sys.argv[2]); NST = int(sys.argv[3]); SEED = int(sys.argv[4])
rng = np.random.default_rng(SEED)
b = 3 / s
rho = np.exp(2j * np.pi / p)
X1, X2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); Y = (X1 + X2) % p
DEL = list(range(1, p))
kill_j = 2 if p == 5 else 3
feats = [1] if p == 5 else [1, 2]
Bp = 2 * (6 * p) ** -1.5
Gam = 2.5 * 2 ** -s if p == 5 else 3.5 * 3 ** -s
gIF = 3 ** (1 - s) * Bp * Gam
xs = np.arange(p)
Fmat = rho ** (-np.outer(xs, xs))


def unpack(z, m):
    return z[:m * p].reshape(m, p), z[m * p:2 * m * p].reshape(m, p), z[2 * m * p:3 * m * p].reshape(m, p)


def gaps(U1, U2, W):
    f = np.einsum('ic,ia,ib->abc', W, U1, U2)
    return np.stack([f[X1, X2, Y] - f[X1, X2, (Y - d) % p] for d in DEL], -1)


def beta(U1, U2, W, j):
    return np.sum((2 / p ** 3) * np.conj((W @ Fmat)[:, j]) * (U1 @ Fmat)[:, j] * (U2 @ Fmat)[:, j])


def if_block(j, a):
    out = []
    for k in range(3):
        u = np.cos(2 * np.pi * j * xs / p - np.pi * k / 3)
        uh = np.sum(u * rho ** (-j * xs))
        t = np.conj(a * p ** 3 / (2 * uh * uh))
        w = (2 / p) * np.real(t * rho ** (j * xs))
        out.append((u, u, w))
    return out


def start():
    neurons = []
    for j in feats:
        a = 1.0 if j == 1 else 2 * np.cos(np.pi / 7) * rng.uniform(0.8, 1.2)
        neurons += if_block(j, a)
    for e in range(rng.integers(1, 4)):
        sc = rng.uniform(0.05, 0.6)
        neurons.append(tuple(sc * rng.normal(size=p) for _ in range(3)))
    U1 = np.array([n[0] for n in neurons]); U2 = np.array([n[1] for n in neurons]); W = np.array([n[2] for n in neurons])
    U1 = U1 + 0.05 * rng.normal(size=U1.shape); U2 = U2 + 0.05 * rng.normal(size=U2.shape); W = W + 0.05 * rng.normal(size=W.shape)
    nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
    P = np.sum(nrm2 ** (b / 2))
    c = P ** (-1 / b)
    return U1 * c, U2 * c, W * c


def run_one():
    U1, U2, W = start()
    m = U1.shape[0]
    z0 = np.concatenate([U1.ravel(), U2.ravel(), W.ravel(), [gaps(U1, U2, W).min()]])
    cons = [{'type': 'ineq', 'fun': lambda z: (gaps(*unpack(z[:-1], m)) - z[-1]).ravel()},
            {'type': 'eq', 'fun': lambda z: np.array([1.0 - np.sum((((unpack(z[:-1], m)[0] ** 2).sum(1) + (unpack(z[:-1], m)[1] ** 2).sum(1)
                                                                       + (unpack(z[:-1], m)[2] ** 2).sum(1)) + 1e-14) ** (b / 2))])},
            {'type': 'eq', 'fun': lambda z: (lambda v: np.array([v.real, v.imag]) * 1e3)(beta(*unpack(z[:-1], m), kill_j))}]
    res = minimize(lambda z: -z[-1], z0, method='SLSQP', constraints=cons, options=dict(maxiter=500, ftol=1e-12))
    U1, U2, W = unpack(res.x[:-1], m)
    bk = beta(U1, U2, W, kill_j)
    h = gaps(U1, U2, W).min()
    nrm2 = (U1 ** 2).sum(1) + (U2 ** 2).sum(1) + (W ** 2).sum(1)
    P = np.sum(nrm2 ** (b / 2))
    bs = [abs(beta(U1, U2, W, j)) for j in range(1, (p - 1) // 2 + 1)]
    return h / P ** s / gIF, abs(bk), bs, m


t0 = time.time()
res = []
for k in range(NST):
    try:
        res.append(run_one())
    except Exception as ex:
        print("err", ex)
res.sort(key=lambda r: -r[0])
for r in res[:5]:
    print(f"p={p} s={s} ratio={r[0]:.5f} |beta_kill|={r[1]:.1e} |beta_j|={np.round(r[2],5).tolist()} width={r[3]}")
print(f"p={p} s={s}: best over {len(res)} starts = {res[0][0]:.5f}   ({time.time()-t0:.0f}s)")
